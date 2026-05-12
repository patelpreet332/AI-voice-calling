import { WebSocket } from "ws";
import { VAD } from "./vad.js";
import { transcribe } from "./stt.js";
import { LLMSession } from "./llm.js";
import { synthesize } from "./tts.js";
import { mulawToLinear16k, pcmToMulawBase64 } from "./audio-utils.js";

/** Size of each mulaw chunk sent back to Twilio */
const TWILIO_CHUNK_SIZE = 160;
/** Piper TTS output sample rate */
const PIPER_SAMPLE_RATE = 22050;

/** Greeting message */
const GREETING_TEXT = "Hello! Pikachu here, Thanks for taking the call. How can I help you today?";

export class CallSession {
  private streamSid: string;
  private twilioWs: WebSocket;
  private vad: VAD;
  private llm: LLMSession;
  private isActive: boolean = true;
  private isAiSpeaking: boolean = false;
  private stopCurrentAudio: boolean = false;
  private callSid: string | null = null;
  private lastAiSpeechEndTime: number = 0;

  constructor(streamSid: string, twilioWs: WebSocket) {
    this.streamSid = streamSid;
    this.twilioWs = twilioWs;

    // Improved VAD settings
    this.vad = new VAD({
  speechThreshold: 300,           // slightly higher
  silenceDurationMs: 900,        // increased (was 1200)
  minSpeechDurationMs: 450,       // increased
  maxSpeechDurationMs: 10000,
    });

    this.llm = new LLMSession();

    this.vad.onUtterance((pcmAudio: Buffer) => {
      this.handleUtterance(pcmAudio);
    });

    console.log(`🎯 [SESSION] Created for stream: ${streamSid}`);
    setTimeout(() => this.sendGreeting(), 600);
  }

  setCallSid(callSid: string): void {
    this.callSid = callSid;
  }

  /**
   * Process incoming audio from Twilio
   */
  processMedia(base64Payload: string): void {
    if (!this.isActive) return;

    const pcm16k = mulawToLinear16k(base64Payload);
    const now = Date.now();

    // Cooldown period after AI finishes speaking (reduces echo)
    if (now - this.lastAiSpeechEndTime < 850) {
      return;
    }

    if (this.isAiSpeaking) {
      const rms = this.calculateRMS(pcm16k);

      // Strong interruption threshold
      if (rms > 680) {
        console.log("⚡ [INTERRUPT] User interrupted AI!");
        this.interrupt();
      } else {
        // Feed with higher threshold while AI is speaking
        this.vad.processAudioWithThreshold(pcm16k, 800);
        return;
      }
    }

    this.vad.processAudio(pcm16k);
  }

  private calculateRMS(pcmBuffer: Buffer): number {
    const samples = new Int16Array(
      pcmBuffer.buffer,
      pcmBuffer.byteOffset,
      pcmBuffer.byteLength / 2,
    );
    if (samples.length === 0) return 0;
    let sumSquares = 0;
    for (let i = 0; i < samples.length; i++) {
      sumSquares += samples[i] * samples[i];
    }
    return Math.sqrt(sumSquares / samples.length);
  }

  private interrupt(): void {
    this.stopCurrentAudio = true;
    this.isAiSpeaking = false;
    this.sendClear();
  }

  private sendClear(): void {
    if (this.twilioWs.readyState === WebSocket.OPEN) {
      this.twilioWs.send(
        JSON.stringify({
          event: "clear",
          streamSid: this.streamSid,
        })
      );
    }
  }

  /**
   * Handle complete user utterance
   */
  private async handleUtterance(pcmAudio: Buffer): Promise<void> {
    if (!this.isActive) return;

    this.vad.setProcessing(true);

    try {
      const startTime = Date.now();

      console.log("📝 [STT] Transcribing...");
      const sttResult = await transcribe(pcmAudio);

      if (!sttResult.text || sttResult.text.trim() === "") {
        console.log("⚠️  [STT] Empty transcription, skipping");
        this.vad.setProcessing(false);
        return;
      }

      console.log(`👤 [USER] "${sttResult.text}" (${sttResult.language})`);

      // === INTERRUPTION COMMANDS ===
      const lower = sttResult.text.toLowerCase().trim();
      if (["stop", "shut up", "bas", "chup", "band kar", "enough", "quiet", "rok"].some(cmd => lower.includes(cmd))) {
        console.log("🛑 [STOP COMMAND] User asked to stop");
        this.interrupt();
        this.vad.setProcessing(false);
        return;
      }

      // === LLM + TTS ===
      console.log("🤖 [LLM] Thinking...");
      const llmStart = Date.now();
      let isFirstSentence = true;

      for await (const sentence of this.llm.chatStream(sttResult.text, sttResult.language)) {
        if (!this.isActive || this.stopCurrentAudio) {
          break;
        }

        if (sentence) {
          if (isFirstSentence) {
            console.log(`🤖 [AI] First sentence: "${sentence}" (${Date.now() - llmStart}ms)`);
            isFirstSentence = false;
          } else {
            console.log(`🤖 [AI] "${sentence}"`);
          }

          const ttsStart = Date.now();
          const ttsPcm = await synthesize(sentence, sttResult.language);
          console.log(`🔊 [TTS] Done (${Date.now() - ttsStart}ms)`);

          await this.sendAudioToTwilio(ttsPcm);
        }
      }

      const totalTime = Date.now() - startTime;
      console.log(`✅ [PIPELINE] Total: ${totalTime}ms`);
    } catch (err: any) {
      console.error("❌ [PIPELINE] Error:", err.message);
    } finally {
      this.vad.setProcessing(false);
    }
  }

  private async sendAudioToTwilio(pcm22kBuffer: Buffer): Promise<void> {
    this.isAiSpeaking = true;
    this.stopCurrentAudio = false;

    const fullMulawBase64 = pcmToMulawBase64(pcm22kBuffer, PIPER_SAMPLE_RATE);
    const mulawBuffer = Buffer.from(fullMulawBase64, "base64");

    for (let offset = 0; offset < mulawBuffer.length; offset += TWILIO_CHUNK_SIZE) {
      if (this.stopCurrentAudio || !this.isActive || this.twilioWs.readyState !== WebSocket.OPEN) {
        console.log("🛑 [TWILIO] Audio transmission interrupted");
        this.stopCurrentAudio = false;
        break;
      }

      const chunk = mulawBuffer.subarray(
        offset,
        Math.min(offset + TWILIO_CHUNK_SIZE, mulawBuffer.length)
      );

      this.twilioWs.send(
        JSON.stringify({
          event: "media",
          streamSid: this.streamSid,
          media: { payload: chunk.toString("base64") },
        })
      );
    }

    this.isAiSpeaking = false;
    this.lastAiSpeechEndTime = Date.now();

    this.twilioWs.send(
      JSON.stringify({
        event: "mark",
        streamSid: this.streamSid,
        mark: { name: `utterance-${Date.now()}` },
      })
    );
  }

  private async sendGreeting(): Promise<void> {
    if (!this.isActive) return;

    try {
      console.log("👋 [GREETING] Synthesizing greeting...");
      this.vad.setProcessing(true);

      const ttsPcm = await synthesize(GREETING_TEXT);
      await this.sendAudioToTwilio(ttsPcm);

      this.llm.addAssistantMessage(GREETING_TEXT);
      console.log("👋 [GREETING] Sent successfully");
    } catch (err: any) {
      console.error("❌ [GREETING] Failed:", err.message);
    } finally {
      this.vad.setProcessing(false);
    }
  }

  destroy(): void {
    this.isActive = false;
    this.vad.flush();
    this.vad.reset();
    this.llm.clear();
    console.log(`🔴 [SESSION] Destroyed for stream: ${this.streamSid}`);
  }
}