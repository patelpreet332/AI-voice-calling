import axios from "axios";

interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

// Keep aligned with python-stt/local_test.py
const SYSTEM_PROMPT = `You are a real-time  voice assistant speaking over a phone call.
Keep replies natural, human-like and brief.
Keep your response short and to the point, like a helpful friend on a call but should include all user answer.
Always reply in same language as user.
Use casual spoken tone.
If unclear, ask short clarification.
Dont ever point yourself with name.
`;

export class LLMSession {
  private history: ChatMessage[] = [];
  private model: string = "llama-3.1-8b-instant";

  constructor() {
    this.history.push({ role: "system", content: SYSTEM_PROMPT });
  }

  /**
   * Streaming response similar to python-stt/local_test.py:
   * flush chunks when buffer grows or punctuation ends.
   */
  async *chatStream(
    userMessage: string,
    detectedLanguage: string = "en",
  ): AsyncGenerator<string, void, unknown> {
    const lang = (detectedLanguage || "en").toLowerCase().slice(0, 2);
    this.history.push({
      role: "user",
      // Strong per-turn constraint to prevent drifting into Hindi/English.
      content:
        `${userMessage}\n\n` +
        `[Reply ONLY in language code '${lang}'. Keep it short, spoken, natural. ` +
        `Do NOT translate to another language.`
    });

    // Mirror local_test.py: keep last ~10 turns for speed/latency.
    if (this.history.length > 11) {
      const system = this.history[0];
      this.history = [system, ...this.history.slice(-10)];
    }

    const response = await axios.post(
      "https://api.groq.com/openai/v1/chat/completions",
      {
        model: this.model,
        messages: this.history,
        // Lower temperature improves instruction-following and language adherence.
        temperature: 0.35,
        max_tokens: 150,
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
    let buffer = "";
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
              buffer += text;

              // Low latency: yield as soon as we have enough characters or punctuation.
              if (buffer.length >= 60 || sentenceEndRegex.test(buffer)) {
                yield buffer;
                buffer = "";
              }
            }
          } catch (e) {
            // Ignore partial JSON
          }
        }
      }
    }

    if (buffer.length > 0) {
      yield buffer;
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
