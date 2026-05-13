import axios from "axios";
import FormData from "form-data";
import { createWavBuffer } from "./audio-utils.js";

/**
 * Speech-to-Text client using local Faster-Whisper (Python FastAPI)
 *
 * Sends PCM audio to the Python STT service running on localhost:8000
 */

const STT_URL = process.env.STT_URL || "http://localhost:8000";

export interface STTResult {
  text: string;
  language: string;
  confidence: number;
  duration: number;
}

/**
 * Transcribe PCM 16kHz audio buffer using Faster-Whisper
 */
export async function transcribe(pcmBuffer: Buffer): Promise<STTResult> {
  // Wrap raw PCM in a WAV container (Whisper expects WAV format)
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
  };
}