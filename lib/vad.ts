/**
 * Simple energy-based Voice Activity Detection (VAD)
 *
 * Buffers PCM 16kHz audio and detects speech boundaries
 * by monitoring RMS energy levels. When silence is detected
 * after speech, the buffered audio is emitted for processing.
 */

export interface VADConfig {
  /** RMS threshold to consider a frame as speech (0-32767 range for 16-bit) */
  speechThreshold: number;
  /** Duration of silence (ms) after speech to trigger end-of-utterance */
  silenceDurationMs: number;
  /** Minimum speech duration (ms) to avoid processing noise bursts */
  minSpeechDurationMs: number;
  /** Maximum speech duration (ms) to force-flush very long utterances */
  maxSpeechDurationMs: number;
  /** Sample rate of the incoming audio */
  sampleRate: number;
}

const DEFAULT_CONFIG: VADConfig = {
  // Tuned to mirror python-stt/local_test.py behavior (MIN_SPEECH_SEC/SILENCE_SEC/MAX_AUDIO_SEC)
  // Note: Twilio phone audio benefits from slightly shorter end-of-utterance silence.
  speechThreshold: 300,
  silenceDurationMs: 650,
  minSpeechDurationMs: 650,
  maxSpeechDurationMs: 20000,
  sampleRate: 16000,
};

export class VAD {
  private config: VADConfig;
  private audioBuffer: Buffer[] = [];
  private totalBytes: number = 0;
  private isSpeaking: boolean = false;
  private silenceStart: number = 0;
  private speechStart: number = 0;
  private onSpeechEnd: ((audio: Buffer) => void) | null = null;

  // Track if we're currently processing (to avoid overlapping STT calls)
  private isProcessing: boolean = false;
  // When an utterance ends while we're processing, keep the latest one instead of dropping it.
  private pendingUtterance: Buffer | null = null;

  constructor(config: Partial<VADConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Register callback for when a complete utterance is detected
   */
  onUtterance(callback: (audio: Buffer) => void): void {
    this.onSpeechEnd = callback;
  }

  /**
   * Set processing state — when true, incoming audio is still buffered
   * but won't trigger new utterance callbacks
   */
  setProcessing(state: boolean): void {
    this.isProcessing = state;
    if (!state && this.pendingUtterance && this.onSpeechEnd) {
      const audio = this.pendingUtterance;
      this.pendingUtterance = null;
      this.onSpeechEnd(audio);
    }
  }

  /**
   * Feed PCM 16-bit audio data into the VAD
   */
  processAudio(pcmBuffer: Buffer): void {
    const rms = this.calculateRMS(pcmBuffer);
    const now = Date.now();

    if (rms > this.config.speechThreshold) {
      // Speech detected
      if (!this.isSpeaking) {
        this.isSpeaking = true;
        this.speechStart = now;
        console.log("🗣️  [VAD] Speech started");
      }
      this.silenceStart = 0;
      this.audioBuffer.push(pcmBuffer);
      this.totalBytes += pcmBuffer.length;

      // Check max duration
      const speechDuration = now - this.speechStart;
      if (speechDuration >= this.config.maxSpeechDurationMs) {
        console.log("⏱️  [VAD] Max speech duration reached, flushing");
        this.flushBuffer();
      }
    } else {
      // Silence
      if (this.isSpeaking) {
        // Still buffer silence for a bit (natural speech has pauses)
        this.audioBuffer.push(pcmBuffer);
        this.totalBytes += pcmBuffer.length;

        if (this.silenceStart === 0) {
          this.silenceStart = now;
        }

        const silenceDuration = now - this.silenceStart;
        if (silenceDuration >= this.config.silenceDurationMs) {
          const speechDuration = now - this.speechStart;
          if (speechDuration >= this.config.minSpeechDurationMs) {
            console.log(
              `⏹️  [VAD] Speech ended (${speechDuration}ms speech, ${silenceDuration}ms silence)`,
            );
            this.flushBuffer();
          } else {
            console.log("⚠️  [VAD] Too short, discarding");
            this.resetBuffer();
          }
        }
      }
    }
  }

  /**
   * Force flush any remaining audio (e.g., on call end)
   */
  flush(): void {
    if (this.isSpeaking && this.audioBuffer.length > 0) {
      this.flushBuffer();
    }
  }

  /**
   * Reset state for a new call
   */
  reset(): void {
    this.resetBuffer();
    this.isProcessing = false;
  }

  private flushBuffer(): void {
    if (this.isProcessing) {
      console.log(
        "⏳ [VAD] Still processing previous utterance, skipping...",
      );
      // Do not drop the user's speech entirely; keep the latest completed utterance.
      this.pendingUtterance = Buffer.concat(this.audioBuffer);
      this.resetBuffer();
      return;
    }

    const combined = Buffer.concat(this.audioBuffer);
    this.resetBuffer();

    if (this.onSpeechEnd && combined.length > 0) {
      this.onSpeechEnd(combined);
    }
  }

  private resetBuffer(): void {
    this.audioBuffer = [];
    this.totalBytes = 0;
    this.isSpeaking = false;
    this.silenceStart = 0;
    this.speechStart = 0;
  }
  processAudioWithThreshold(pcmBuffer: Buffer, customThreshold: number): void {
  const originalThreshold = this.config.speechThreshold;
  this.config.speechThreshold = customThreshold;
  this.processAudio(pcmBuffer);
  this.config.speechThreshold = originalThreshold; // restore
}

  /**
   * Calculate Root Mean Square of PCM 16-bit audio
   */
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
}
