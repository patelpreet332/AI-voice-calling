import { NextRequest, NextResponse } from "next/server";
import axios from "axios";
import FormData from "form-data";

export async function POST(req: NextRequest) {
  try {
    const form = await req.formData();
    const file = form.get("file") as File;

    if (!file) {
      return NextResponse.json({ error: "No file uploaded" }, { status: 400 });
    }

    // Convert to buffer
    const bytes = await file.arrayBuffer();
    const buffer = Buffer.from(bytes);

    const data = new FormData();
    data.append("file", buffer, {
      filename: "audio.webm",
      contentType: "audio/webm",
    });

    // ✅ FIXED MODEL
    data.append("model", "whisper-large-v3-turbo");

    // Optional but improves accuracy
    data.append("language", "en");

    const response = await axios.post(
      "https://api.groq.com/openai/v1/audio/transcriptions",
      data,
      {
        headers: {
          Authorization: `Bearer ${process.env.GROQ_API_KEY}`,
          ...data.getHeaders(),
        },
      }
    );

    return NextResponse.json({ text: response.data.text });
  } catch (error: any) {
    console.error("TRANSCRIBE ERROR:", error?.response?.data || error.message);

    return NextResponse.json(
      {
        error: "Transcription failed",
        details: error?.response?.data || error.message,
      },
      { status: 500 }
    );
  }
}