import express, { Request, Response } from "express";
import { createServer } from "http";
import { WebSocketServer, WebSocket } from "ws";
import twilio from "twilio";
import multer from "multer";
import axios from "axios";
import FormData from "form-data";
import { mulaw } from "alawmulaw";
import "dotenv/config";

// ── Local AI pipeline imports ──
import { CallSession } from "./lib/call-session.js";
import { checkPiperReady } from "./lib/tts.js";

// ── ElevenLabs import (preserved — not used for local pipeline) ──
import { connectToElevenLabs } from "./lib/elevenlabs.js";

const PORT = process.env.PORT || 3000;

const app = express();
const server = createServer(app); 

const wss = new WebSocketServer({ noServer: true });

// ✅ Upgrade handler
server.on("upgrade", (req, socket, head) => {
  if (req.url === "/media-stream") {
    wss.handleUpgrade(req, socket, head, (ws) => {
      wss.emit("connection", ws, req);
    });
  } else {
    socket.destroy();
  }
});


const twilioClient = twilio(
  process.env.TWILIO_ACCOUNT_SID,
  process.env.TWILIO_AUTH_TOKEN,
);

const upload = multer();

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// ═══════════════════════════════════════════════════════
// ✅ API Routes (preserved — existing functionality)
// ═══════════════════════════════════════════════════════

app.post("/api/ai", async (req: Request, res: Response) => {
  try {
    const { text, language } = req.body;

    if (!text || text.trim() === "") {
      return res.status(400).json({ error: "Empty transcription" });
    }

    const response = await axios.post(
      "https://api.groq.com/openai/v1/chat/completions",
      {
        model: "llama-3.1-8b-instant",
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
        temperature: 0.7,
      },
      {
        headers: {
          Authorization: `Bearer ${process.env.GROQ_API_KEY}`,
          "Content-Type": "application/json",
        },
      },
    );

    res.json({ reply: response.data.choices[0].message.content });
  } catch (error: any) {
    console.error("AI ERROR:", error?.response?.data || error.message);
    res.status(500).json({
      error: "AI failed",
      details: error?.response?.data || error.message,
    });
  }
});

app.post(
  "/api/transcribe",
  upload.single("file"),
  async (req: Request, res: Response) => {
    try {
      const file = req.file;

      if (!file) {
        return res.status(400).json({ error: "No file uploaded" });
      }

      const data = new FormData();
      data.append("file", file.buffer, {
        filename: "audio.webm",
        contentType: "audio/webm",
      });

      data.append("model", "whisper-large-v3-turbo");
      data.append("language", "en");

      const response = await axios.post(
        "https://api.groq.com/openai/v1/audio/transcriptions",
        data,
        {
          headers: {
            Authorization: `Bearer ${process.env.GROQ_API_KEY}`,
            ...data.getHeaders(),
          },
        },
      );

      res.json({ text: response.data.text });
    } catch (error: any) {
      console.error(
        "TRANSCRIBE ERROR:",
        error?.response?.data || error.message,
      );
      res.status(500).json({
        error: "Transcription failed",
        details: error?.response?.data || error.message,
      });
    }
  },
);

// ═══════════════════════════════════════════════════════
// ✅ Twilio webhook — outbound call connects here
// ═══════════════════════════════════════════════════════

app.post("/voice", (req: Request, res: Response) => {
  console.log("🔔 [TWILIO] Webhook hit");

  const response = new twilio.twiml.VoiceResponse();

  const domain = process.env.NGROK_URL!.replace("https://", "");
  const streamUrl = `wss://${domain}/media-stream`;

  console.log("🔌 Stream URL:", streamUrl);

  const connect = response.connect();
  connect.stream({ url: streamUrl });

  res.type("text/xml");
  res.send(response.toString());
});

// ═══════════════════════════════════════════════════════
// ✅ WebSocket handler — Local AI Pipeline
//    Twilio mulaw 8kHz → PCM 16kHz → STT → LLM → TTS → mulaw 8kHz → Twilio
// ═══════════════════════════════════════════════════════

wss.on("connection", (twilioWs: WebSocket) => {
  let callSession: CallSession | null = null;

  console.log("📞 [WS] New Twilio WebSocket connection");

  twilioWs.on("message", (data: string) => {
    try {
      const message = JSON.parse(data);

      switch (message.event) {
        case "connected":
          console.log("📞 [TWILIO] Media stream connected");
          break;

        case "start":
          const streamSid = message.start.streamSid;
          console.log("🎙️ [TWILIO] Call started:", streamSid);
          console.log("📋 [TWILIO] Call SID:", message.start.callSid);
          console.log(
            "📋 [TWILIO] Tracks:",
            message.start.mediaFormat?.encoding,
            message.start.mediaFormat?.sampleRate,
            message.start.mediaFormat?.channels,
          );

          // Create a new CallSession for this call
          callSession = new CallSession(streamSid, twilioWs);
          if (message.start.callSid) {
            callSession.setCallSid(message.start.callSid);
          }
          break;

        case "media":
          const payload = message.media?.payload;
          if (!payload || !callSession) return;

          // Feed Twilio audio into the local pipeline
          callSession.processMedia(payload);
          break;

        case "mark":
          console.log(
            "✔️  [TWILIO] Mark received:",
            message.mark?.name,
          );
          break;

        case "stop":
          console.log("🛑 [TWILIO] Call ended");
          callSession?.destroy();
          callSession = null;
          break;
      }
    } catch (err) {
      console.error("❌ [WS] Parse error:", err);
    }
  });

  twilioWs.on("close", () => {
    console.log("🔴 [WS] Twilio WebSocket closed");
    callSession?.destroy();
    callSession = null;
  });

  twilioWs.on("error", (err) => {
    console.error("❌ [WS] WebSocket error:", err);
    callSession?.destroy();
    callSession = null;
  });
});

// ═══════════════════════════════════════════════════════
// ✅ OLD ElevenLabs WebSocket handler (preserved for reference)
//    Uncomment and use connectToElevenLabs if switching back
// ═══════════════════════════════════════════════════════

/*
function setupElevenLabs(
  elevenLabsWs: WebSocket,
  twilioWs: WebSocket,
  streamSid: string,
  onReady: () => void,
) {
  elevenLabsWs.on("message", (data: string) => {
    try {
      const message = JSON.parse(data);

      switch (message.type) {
        case "audio":
          if (message.audio_event?.audio_base_64) {
            twilioWs.send(
              JSON.stringify({
                event: "media",
                streamSid,
                media: {
                  payload: message.audio_event.audio_base_64,
                },
              }),
            );
          }
          break;

        case "user_transcript":
          console.log(
            "👤 User:",
            message.user_transcription_event?.user_transcript,
          );
          break;

        case "agent_response":
          console.log("🤖 AI:", message.agent_response_event?.agent_response);
          break;

        case "conversation_initiation_metadata":
          console.log("✅ ElevenLabs ready");
          onReady();
          break;

        case "error":
          console.error("❌ ElevenLabs error:", message);
          break;
      }
    } catch (err) {
      console.error("❌ ElevenLabs parse error:", err);
    }
  });
}
*/

// ═══════════════════════════════════════════════════════
// ✅ Outbound call trigger
// ═══════════════════════════════════════════════════════

async function makeCall(to: string) {
  const call = await twilioClient.calls.create({
    to,
    from: process.env.TWILIO_PHONE_NUMBER!,
    url: `${process.env.NGROK_URL}/voice`,
  });

  console.log("📞 Call SID:", call.sid);
  return call.sid;
}

app.get("/make-call", async (_, res) => {
  try {
    const sid = await makeCall(process.env.USER_PHONE_NUMBER!);
    res.send(`Call started: ${sid}`);
  } catch (err: any) {
    console.error("❌ Failed to make call:", err.message);
    res.status(500).send(`Call failed: ${err.message}`);
  }
});

// ═══════════════════════════════════════════════════════
// ✅ Server startup with preflight checks
// ═══════════════════════════════════════════════════════

server.listen(PORT, () => {
  // Check Piper TTS is available
  const piperReady = checkPiperReady();

  console.log(`
🚀 SERVER READY — Local AI Voice Pipeline
═══════════════════════════════════════════
Local:      http://localhost:${PORT}
Webhook:    ${process.env.NGROK_URL}/voice
Make Call:  http://localhost:${PORT}/make-call
═══════════════════════════════════════════
Pipeline:   Twilio → VAD → Faster-Whisper → Groq → Piper → Twilio
Piper TTS:  ${piperReady ? "✅ Ready" : "❌ NOT FOUND"}
STT Server: http://localhost:8000 (start python-stt/main.py)
═══════════════════════════════════════════
`);
});