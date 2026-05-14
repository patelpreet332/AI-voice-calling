import axios from "axios";

interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

const SYSTEM_PROMPT = `You are a friendly, natural Indian voice assistant in a real-time phone call.

Core rules:
- Speak exactly like a helpful human friend — warm, casual, never robotic.
- Keep every reply to 1-2 short sentences max. Be crisp.
- Use natural fillers: "Haan", "Theek hai", "Achha", "Okay" etc.
- Vary your phrasing naturally.

Language rules:
- Always reply in the same language (or mix) the user is using right now.
- For Hindi/Marathi: Use proper Devanagari.
- For Gujarati, Tamil, Telugu etc.: Write phonetically in Devanagari so the Hindi voice can pronounce it naturally (e.g., "Tamne Gujarati ma baat karvi che?").
- Handle Hinglish/code-mixing perfectly — it's normal.

STT is sometimes imperfect — understand intent, not exact words.
If unclear, ask one short clarification question.

Never be verbose. Never use lists or markdown.
If user says stop/bas/chup/enough — politely stop and confirm.

Tone: Friendly, calm, helpful, slightly playful.`;

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

    // Embed invisible per-turn language directive to guarantee flawless multilingual behavior
    const apiMessages = [...this.history];
    const lastIdx = apiMessages.length - 1;
    apiMessages[lastIdx] = {
      role: "user",
      content: `${userMessage}\n\n[System directive: Respond in language '${detectedLanguage}'. If this is an Indian regional language (e.g. gu, te, ta, kn, ml, bn, pa), you MUST output the response written entirely in the Devanagari script so the TTS engine can speak it.]`
    };

    const response = await axios.post(
      "https://api.groq.com/openai/v1/chat/completions",
      {
        model: this.model,
        messages: apiMessages,
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