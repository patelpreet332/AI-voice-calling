
export interface VADConfig {

  speechThreshold: number;

  silenceDurationMs: number;

  minSpeechDurationMs: number;

  maxSpeechDurationMs: number;

  sampleRate: number;
}

const DEFAULT_CONFIG: VADConfig = {
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

  private isProcessing: boolean = false;
  private pendingUtterance: Buffer | null = null;

  constructor(config: Partial<VADConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }


  onUtterance(callback: (audio: Buffer) => void): void {
    this.onSpeechEnd = callback;
  }


  setProcessing(state: boolean): void {
    this.isProcessing = state;
    if (!state && this.pendingUtterance && this.onSpeechEnd) {
      const audio = this.pendingUtterance;
      this.pendingUtterance = null;
      this.onSpeechEnd(audio);
    }
  }


  processAudio(pcmBuffer: Buffer): void {
    const rms = this.calculateRMS(pcmBuffer);
    const now = Date.now();

    if (rms > this.config.speechThreshold) {

      if (!this.isSpeaking) {
        this.isSpeaking = true;
        this.speechStart = now;
        console.log("🗣️  [VAD] Speech started");
      }
      this.silenceStart = 0;
      this.audioBuffer.push(pcmBuffer);
      this.totalBytes += pcmBuffer.length;

      const speechDuration = now - this.speechStart;
      if (speechDuration >= this.config.maxSpeechDurationMs) {
        console.log("⏱️  [VAD] Max speech duration reached, flushing");
        this.flushBuffer();
      }
    } else {
      if (this.isSpeaking) {

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


  flush(): void {
    if (this.isSpeaking && this.audioBuffer.length > 0) {
      this.flushBuffer();
    }
  }


  reset(): void {
    this.resetBuffer();
    this.isProcessing = false;
  }

  private flushBuffer(): void {
    if (this.isProcessing) {
      console.log(
        "⏳ [VAD] Still processing previous utterance, skipping...",
      );

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
  this.config.speechThreshold = originalThreshold;
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
}
