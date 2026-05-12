import { mulaw } from "alawmulaw";

/**
 * Audio conversion utilities for Twilio ↔ Local pipeline
 *
 * Twilio sends/receives: μ-law encoded, 8kHz, mono
 * Local pipeline needs:  PCM 16-bit signed LE, 16kHz, mono
 */

/**
 * Decode base64 μ-law 8kHz payload from Twilio → PCM Int16 @ 16kHz
 */
export function mulawToLinear16k(base64Payload: string): Buffer {
  // Step 1: base64 → raw μ-law bytes
  const mulawBuf = Buffer.from(base64Payload, "base64");

  // Step 2: μ-law → PCM 16-bit (still 8kHz)
  const pcm8k: Int16Array = mulaw.decode(mulawBuf);

  // Step 3: Upsample 8kHz → 16kHz (simple linear interpolation)
  const pcm16k = upsample8kTo16k(pcm8k);

  // Step 4: Int16Array → Buffer
  return Buffer.from(pcm16k.buffer, pcm16k.byteOffset, pcm16k.byteLength);
}

/**
 * Convert PCM Int16 @ 16kHz → μ-law base64 payload for Twilio
 */
export function linear16kToMulawBase64(pcm16kBuffer: Buffer): string {
  // Step 1: Buffer → Int16Array
  const pcm16k = new Int16Array(
    pcm16kBuffer.buffer,
    pcm16kBuffer.byteOffset,
    pcm16kBuffer.byteLength / 2,
  );

  // Step 2: Downsample 16kHz → 8kHz (take every other sample)
  const pcm8k = downsample16kTo8k(pcm16k);

  // Step 3: PCM → μ-law
  const mulawBytes: Uint8Array = mulaw.encode(pcm8k);

  // Step 4: → base64
  return Buffer.from(mulawBytes).toString("base64");
}

/**
 * Convert PCM Int16 at ANY sample rate → μ-law 8kHz base64 for Twilio
 * Used for Piper TTS which outputs at 22050 Hz
 */
export function pcmToMulawBase64(
  pcmBuffer: Buffer,
  sourceSampleRate: number,
): string {
  // Step 1: Buffer → Int16Array
  const samples = new Int16Array(
    pcmBuffer.buffer,
    pcmBuffer.byteOffset,
    pcmBuffer.byteLength / 2,
  );

  // Step 2: Resample from source rate → 8kHz
  const pcm8k = resample(samples, sourceSampleRate, 8000);

  // Step 3: PCM → μ-law
  const mulawBytes: Uint8Array = mulaw.encode(pcm8k);

  // Step 4: → base64
  return Buffer.from(mulawBytes).toString("base64");
}

/**
 * Upsample Int16Array from 8kHz → 16kHz using linear interpolation
 */
function upsample8kTo16k(samples: Int16Array): Int16Array {
  const upsampled = new Int16Array(samples.length * 2);
  for (let i = 0; i < samples.length; i++) {
    const current = samples[i];
    const next = i < samples.length - 1 ? samples[i + 1] : current;
    upsampled[i * 2] = current;
    upsampled[i * 2 + 1] = Math.round((current + next) / 2); // interpolated
  }
  return upsampled;
}

/**
 * Downsample Int16Array from 16kHz → 8kHz (take every other sample)
 */
function downsample16kTo8k(samples: Int16Array): Int16Array {
  const downsampled = new Int16Array(Math.floor(samples.length / 2));
  for (let i = 0; i < downsampled.length; i++) {
    downsampled[i] = samples[i * 2];
  }
  return downsampled;
}

/**
 * Generic linear-interpolation resampler
 * Converts PCM Int16 from any source rate to any target rate
 */
function resample(
  samples: Int16Array,
  fromRate: number,
  toRate: number,
): Int16Array {
  if (fromRate === toRate) return samples;

  const ratio = fromRate / toRate;
  const outputLength = Math.floor(samples.length / ratio);
  const output = new Int16Array(outputLength);

  for (let i = 0; i < outputLength; i++) {
    const srcIndex = i * ratio;
    const srcFloor = Math.floor(srcIndex);
    const srcCeil = Math.min(srcFloor + 1, samples.length - 1);
    const fraction = srcIndex - srcFloor;

    // Linear interpolation between two nearest samples
    output[i] = Math.round(
      samples[srcFloor] * (1 - fraction) + samples[srcCeil] * fraction,
    );
  }

  return output;
}

/**
 * Create a proper WAV file buffer from raw PCM data
 */
export function createWavBuffer(
  pcmData: Buffer,
  sampleRate: number = 16000,
  channels: number = 1,
  bitDepth: number = 16,
): Buffer {
  const byteRate = sampleRate * channels * (bitDepth / 8);
  const blockAlign = channels * (bitDepth / 8);
  const dataSize = pcmData.length;

  const header = Buffer.alloc(44);
  header.write("RIFF", 0);
  header.writeUInt32LE(36 + dataSize, 4);
  header.write("WAVE", 8);
  header.write("fmt ", 12);
  header.writeUInt32LE(16, 16); // fmt chunk size
  header.writeUInt16LE(1, 20); // PCM format
  header.writeUInt16LE(channels, 22);
  header.writeUInt32LE(sampleRate, 24);
  header.writeUInt32LE(byteRate, 28);
  header.writeUInt16LE(blockAlign, 32);
  header.writeUInt16LE(bitDepth, 34);
  header.write("data", 36);
  header.writeUInt32LE(dataSize, 40);

  return Buffer.concat([header, pcmData]);
}

/**
 * Strip WAV header (44 bytes) from a WAV file buffer, returning raw PCM
 */
export function stripWavHeader(wavBuffer: Buffer): Buffer {
  // Find the "data" chunk — standard WAV has it at byte 36
  // but some encoders add extra chunks, so search for it
  for (let i = 0; i < wavBuffer.length - 8; i++) {
    if (
      wavBuffer[i] === 0x64 && // 'd'
      wavBuffer[i + 1] === 0x61 && // 'a'
      wavBuffer[i + 2] === 0x74 && // 't'
      wavBuffer[i + 3] === 0x61 // 'a'
    ) {
      const dataSize = wavBuffer.readUInt32LE(i + 4);
      return wavBuffer.subarray(i + 8, i + 8 + dataSize);
    }
  }
  // Fallback: assume standard 44-byte header
  return wavBuffer.subarray(44);
}