import { NextRequest, NextResponse } from "next/server";
import axios from "axios";

export async function POST(req: NextRequest) {
  try {
    const { text, language } = await req.json();

    if (!text || text.trim() === "") {
      return NextResponse.json(
        { error: "Empty transcription" },
        { status: 400 }
      );
    }

    const response = await axios.post(
      "https://api.groq.com/openai/v1/chat/completions",
      {
        model: "llama-3.1-8b-instant", // ✅ FIXED MODEL
        messages: [
          {
            role: "system",
            content: `You are a helpful multilingual AI assistant that give answer in short, crisp and concise way. But you only and only have conversation about Ben 10. if someone ask you about other topic you just divert topic to Ben 10 . you have all Ben 10 knoledge. if someone force you to have conversation about other than Ben 10 you just say them sorry and tell them to ask about Ben 10. Always respond in ${language}.`,
          },
          {
            role: "user",
            content: text,
          },
        ],
        temperature: 0.7, // ✅ important
      },
      {
        headers: {
          Authorization: `Bearer ${process.env.GROQ_API_KEY}`,
          "Content-Type": "application/json",
        },
      }
    );

    return NextResponse.json({
      reply: response.data.choices[0].message.content,
    });
  } catch (error: any) {
    console.error("AI ERROR:", error?.response?.data || error.message);

    return NextResponse.json(
      {
        error: "AI failed",
        details: error?.response?.data || error.message,
      },
      { status: 500 }
    );
  }
}