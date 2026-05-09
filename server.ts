import "dotenv/config";
import express, { Request, Response } from "express";
import { createServer } from "http";
import { WebSocketServer, WebSocket } from "ws";
import twilio from "twilio";
import multer from "multer";
import axios from "axios";
import FormData from "form-data";
import { connectToElevenLabs } from "./lib/elevenlabs";
import { mulaw } from "alawmulaw";

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

// ✅ API Routes
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

// ✅ Twilio webhook
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

// ✅ WebSocket handler
wss.on("connection", (twilioWs: WebSocket) => {
  let streamSid: string | null = null;
  let elevenLabsWs: WebSocket | null = null;
  let elevenLabsReady = false;

  twilioWs.on("message", (data: string) => {
    try {
      const message = JSON.parse(data);

      switch (message.event) {
        case "connected":
          console.log("📞 Twilio connected");
          break;

        case "start":
          streamSid = message.start.streamSid;
          console.log("🎙️ Call started:", streamSid);

          elevenLabsWs = connectToElevenLabs(
            process.env.ELEVENLABS_AGENT_ID!,
            process.env.ELEVENLABS_API_KEY!,
          );

          if (streamSid) {
            setupElevenLabs(elevenLabsWs, twilioWs, streamSid, () => {
              elevenLabsReady = true;
            });
          }
          break;

        case "media":
          const payload = message.media?.payload;
          if (!payload) return;

          if (elevenLabsWs?.readyState === WebSocket.OPEN && elevenLabsReady) {
            // 🔥 STEP 1: base64 → buffer
            const mulawBuffer = Buffer.from(payload, "base64");

            // 🔥 STEP 2: μ-law → PCM16
            const pcm16 = mulaw.decode(mulawBuffer);

            // 🔥 STEP 3: Upsample 8k → 16k
            const upsampled = new Int16Array(pcm16.length * 2);
            for (let i = 0; i < pcm16.length; i++) {
              const sample = pcm16[i];
              upsampled[i * 2] = sample;
              upsampled[i * 2 + 1] = sample;
            }

            // 🔥 STEP 4: convert to buffer
            const pcmBuffer = Buffer.from(upsampled.buffer);

            // 🔥 STEP 5: send to ElevenLabs
            elevenLabsWs.send(
              JSON.stringify({
                user_audio_chunk: pcmBuffer.toString("base64"),
              }),
            );
          }
          break;

        case "stop":
          console.log("🛑 Call ended");
          elevenLabsWs?.close();
          break;
      }
    } catch (err) {
      console.error("❌ Twilio parse error:", err);
    }
  });

  twilioWs.on("close", () => {
    console.log("🔴 Twilio WS closed");
    elevenLabsWs?.close();
  });
});

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

// ✅ Call trigger
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
  const sid = await makeCall(process.env.USER_PHONE_NUMBER!);
  res.send(`Call started: ${sid}`);
});

server.listen(PORT, () => {
  console.log(`
🚀 SERVER READY
-------------------------
Local: http://localhost:${PORT}
Webhook: ${process.env.NGROK_URL}/voice
-------------------------
`);
});
