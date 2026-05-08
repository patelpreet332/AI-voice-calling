"use client";

import { useState, useRef } from "react";

type Message = {
  role: "user" | "ai";
  text: string;
};

export default function Home() {
  const [isActive, setIsActive] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [liveText, setLiveText] = useState("");

  const recognitionRef = useRef<any>(null);
  const isActiveRef = useRef(false);
  const isSpeakingRef = useRef(false);

  // ================= START =================
  const startListening = () => {
    const SpeechRecognition =
      (window as any).SpeechRecognition ||
      (window as any).webkitSpeechRecognition;

    if (!SpeechRecognition) {
      alert("Speech Recognition not supported");
      return;
    }

    const recognition = new SpeechRecognition();

    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = "en-IN";

    recognition.onresult = async (event: any) => {
      // 🚫 IGNORE if AI is speaking
      if (isSpeakingRef.current) return;

      let interim = "";
      let final = "";

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript;

        if (event.results[i].isFinal) final += transcript;
        else interim += transcript;
      }

      setLiveText(interim);

      if (final) {
        addMessage("user", final);
        setLiveText("");

        // 🔥 Stop listening BEFORE AI responds
        recognition.stop();

        const aiRes = await fetch("/api/ai", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: final }),
        });

        const aiData = await aiRes.json();

        if (aiData.reply) {
          addMessage("ai", aiData.reply);
          await speak(aiData.reply);
        }

        // 🔁 Resume listening AFTER speaking
        if (isActiveRef.current) {
          recognition.start();
        }
      }
    };

    recognition.onerror = (e: any) => {
      console.error("Speech error:", e);
    };

    recognition.onend = () => {
      // 🔁 Restart ONLY if active AND not speaking
      if (isActiveRef.current && !isSpeakingRef.current) {
        try {
          recognition.start();
        } catch {}
      }
    };

    recognition.start();
    recognitionRef.current = recognition;

    setIsActive(true);
    isActiveRef.current = true;
  };

  // ================= STOP =================
  const stopAll = () => {
    isActiveRef.current = false;
    setIsActive(false);

    // 🛑 stop recognition
    try {
      recognitionRef.current?.stop();
    } catch {}

    // 🛑 stop speaking
    speechSynthesis.cancel();
    isSpeakingRef.current = false;
    setIsSpeaking(false);

    // 🧹 clear UI
    setLiveText("");
    setMessages([]);
  };

  // ================= MESSAGES =================
  const addMessage = (role: "user" | "ai", text: string) => {
    setMessages((prev) => [...prev, { role, text }]);
  };

  // ================= SPEAK =================
  const speak = (text: string) => {
    return new Promise<void>((resolve) => {
      speechSynthesis.cancel();

      const utterance = new SpeechSynthesisUtterance(text);

      isSpeakingRef.current = true;
      setIsSpeaking(true);

      utterance.onend = () => {
        isSpeakingRef.current = false;
        setIsSpeaking(false);
        resolve();
      };

      speechSynthesis.speak(utterance);
    });
  };

  // ================= TOGGLE =================
  const handleToggle = () => {
    if (isActive) stopAll();
    else startListening();
  };

  return (
    <div className="min-h-screen bg-black text-white flex flex-col items-center p-6">
      <h1 className="text-3xl mb-4">AI Voice Assistant</h1>

      <button
        onClick={handleToggle}
        className={`px-6 py-3 rounded-lg ${
          isActive ? "bg-red-600" : "bg-blue-600"
        }`}
      >
        {isActive ? "Stop Conversation" : "Start Conversation"}
      </button>

      {/* STATUS */}
      <div className="mt-4">
        {isActive && <p>🎤 Listening...</p>}
        {isSpeaking && <p>🗣️ Speaking...</p>}
      </div>

      {/* LIVE TEXT */}
      {liveText && (
        <div className="mt-4 bg-yellow-600 p-3 rounded-lg max-w-md w-full">
          <b>Listening:</b> {liveText}
        </div>
      )}

      {/* CHAT */}
      <div className="mt-6 w-full max-w-md space-y-3">
        {messages.map((m, i) => (
          <div
            key={i}
            className={`p-3 rounded-lg ${
              m.role === "user" ? "bg-gray-700" : "bg-green-600"
            }`}
          >
            <b>{m.role === "user" ? "You" : "AI"}:</b> {m.text}
          </div>
        ))}
      </div>
    </div>
  );
}