import WebSocket from "ws";

export function connectToElevenLabs(
  agentId: string,
  apiKey: string
): WebSocket {
  const url = `wss://api.elevenlabs.io/v1/convai/conversation?agent_id=${agentId}`;

  console.log("🔗 [ELEVENLABS] Connecting...");

  const ws = new WebSocket(url, {
    headers: {
      "xi-api-key": apiKey,
    },
  });

  ws.on("open", () => {
    console.log("✅ ElevenLabs connected");

    ws.send(
      JSON.stringify({
        type: "conversation_initiation_client_data",
      })
    );
  });

  ws.on("message", (data) => {
    const msg = JSON.parse(data.toString());

    if (msg.type === "conversation_initiation_metadata") {
      console.log("✅ ElevenLabs READY");
    }
  });

  ws.on("error", (err) => {
    console.error("❌ ElevenLabs error:", err);
  });

  ws.on("close", () => {
    console.log("🔌 ElevenLabs closed");
  });

  return ws;
}