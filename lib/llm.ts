import axios from "axios";

interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

const SYSTEM_PROMPT = `You are a friendly, natural-sounding AI voice assistant in a real-time phone conversation.

Core behavior:
- Speak like a human, not like a chatbot.
- Keep responses short: 1–2 sentences max unless absolutely necessary.
- Use simple, conversational language.

Conversation style:
- Use natural fillers occasionally like "Okay", "Got it", "Sure".
- Avoid sounding robotic or overly formal.
- Slightly vary phrasing to feel natural and unscripted.

Language handling:
- Always reply in the english language.

Understanding user input:
- Speech-to-text may contain errors; infer meaning from context.
- Prioritize intent over exact words.
- If something is unclear, ask a short clarification question.

Turn-taking:
- Do not interrupt the user.
- Assume the user may pause mid-sentence; wait before responding.
- If user says "stop", "enough", or similar → acknowledge briefly and stop.

Error handling:
- If you don’t know something, say so briefly and honestly.
- Do not guess or hallucinate information.

Response constraints:
- No markdown, no bullet points, no special formatting.
- No long explanations unless explicitly asked.
- Avoid repeating the same phrases frequently.

Tone:
- Friendly, calm, and helpful.
- Slightly informal, like talking to a real person on a call.

Goal:
- Help the user efficiently while sounding natural and easy to talk to.
`;

export class LLMSession {
  private history: ChatMessage[] = [];
  private model: string = "llama-3.1-8b-instant";

  constructor() {
    this.history.push({ role: "system", content: SYSTEM_PROMPT });
  }

  /**
   * Streaming response with language awareness
   */
  async *chatStream(userMessage: string, detectedLanguage: string = "en"): AsyncGenerator<string, void, unknown> {
    this.history.push({ role: "user", content: userMessage });

    if (this.history.length > 21) {
      const system = this.history[0];
      this.history = [system, ...this.history.slice(-20)];
    }

    const response = await axios.post(
      "https://api.groq.com/openai/v1/chat/completions",
      {
        model: this.model,
        messages: this.history,
        temperature: 0.75,
        max_tokens: 280,
        stream: true,
      },
      {
        headers: {
          Authorization: `Bearer ${process.env.GROQ_API_KEY}`,
          "Content-Type": "application/json",
        },
        responseType: "stream",
        timeout: 15000,
      }
    );

    let fullReply = "";
    let sentenceBuffer = "";
    const sentenceEndRegex = /[.?!।]\s*$/;

    for await (const chunk of response.data) {
      const lines = chunk.toString("utf8").split("\n").filter((line: string) => line.trim() !== "");

      for (const line of lines) {
        if (line === "data: [DONE]") continue;
        if (line.startsWith("data: ")) {
          try {
            const data = JSON.parse(line.slice(6));
            if (data.choices[0].delta?.content) {
              const text = data.choices[0].delta.content;
              fullReply += text;
              sentenceBuffer += text;

              if (sentenceEndRegex.test(sentenceBuffer)) {
                yield sentenceBuffer.trim();
                sentenceBuffer = "";
              }
            }
          } catch (e) {
            // Ignore partial JSON
          }
        }
      }
    }

    if (sentenceBuffer.trim().length > 0) {
      yield sentenceBuffer.trim();
    }

    this.history.push({ role: "assistant", content: fullReply });
  }

  addAssistantMessage(message: string): void {
    this.history.push({ role: "assistant", content: message });
  }

  clear(): void {
    const system = this.history[0];
    this.history = [system];
  }
}