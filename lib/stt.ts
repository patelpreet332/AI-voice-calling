import axios from "axios";
import FormData from "form-data";
import { createWavBuffer } from "./audio-utils.js";

/**
 * Speech-to-Text client using Hybrid Pipeline (Python FastAPI)
 *
 * Pipeline:
 *   1. Whisper medium → language detection
 *   2. English → Whisper medium
 *   3. Indian language → IndicConformer-600M
 *
 * Sends PCM audio to the Python STT service running on localhost:8000
 */

const STT_URL = process.env.STT_URL || "http://localhost:8000";

export interface STTResult {
  text: string;
  language: string;
  confidence: number;
  duration: number;
  engine?: string;
  pipeline_time?: number;
  detect_time?: number;
}

/**
 * Transcribe PCM 16kHz audio buffer using the hybrid STT pipeline
 */
export async function transcribe(pcmBuffer: Buffer): Promise<STTResult> {
  // Wrap raw PCM in a WAV container (STT service expects WAV format)
  const wavBuffer = createWavBuffer(pcmBuffer, 16000, 1, 16);

  const form = new FormData();
  form.append("file", wavBuffer, {
    filename: "audio.wav",
    contentType: "audio/wav",
  });

  const response = await axios.post(`${STT_URL}/transcribe`, form, {
    headers: form.getHeaders(),
    timeout: 150000,
  });

  return {
    text: response.data.text || "",
    language: response.data.language || "en",
    confidence: response.data.confidence || 0,
    duration: response.data.duration || 0,
    engine: response.data.engine || "unknown",
    pipeline_time: response.data.pipeline_time || 0,
    detect_time: response.data.detect_time || 0,
  };
}