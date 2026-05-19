import axios from "axios";
import FormData from "form-data";
import { createWavBuffer } from "./audio-utils.js";



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

export interface TranscribeOptions {

  hintLanguage?: string;
}


export async function transcribe(
  pcmBuffer: Buffer,
  opts: TranscribeOptions = {},
): Promise<STTResult> {

  const wavBuffer = createWavBuffer(pcmBuffer, 16000, 1, 16);

  const form = new FormData();
  form.append("file", wavBuffer, {
    filename: "audio.wav",
    contentType: "audio/wav",
  });

  const hint = (opts.hintLanguage || "").trim().toLowerCase().slice(0, 2);
  if (hint) {
    form.append("hint_language", hint);
  }

  const response = await axios.post(`${STT_URL}/transcribe`, form, {
    headers: form.getHeaders(),
    timeout: 150000,
  });

  const durationSec = pcmBuffer.length / 2 / 16000;

  return {
    text: response.data.text || "",
    language: response.data.language || "en",
    confidence: response.data.confidence || 0,
    duration: response.data.duration || durationSec,
    engine: response.data.engine || "unknown",
    pipeline_time: response.data.pipeline_time || 0,
    detect_time: response.data.detect_time || 0,
  };
}
