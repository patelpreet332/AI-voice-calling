#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║  LOCAL VOICE AGENT — End-to-End Terminal Test               ║
║                                                             ║
║  Pipeline:                                                  ║
║  🎤 Mic → VAD → Whisper (lang detect)                      ║
║       → English?  → Whisper (transcribe)                    ║
║       → Indian?   → IndicConformer (transcribe)             ║
║  → Groq LLM (streaming) → Piper TTS → 🔊 Speaker          ║
╚══════════════════════════════════════════════════════════════╝

Usage:
    python local_test.py                 # Interactive voice loop
    python local_test.py --file test.wav # Transcribe a file (no LLM/TTS)
"""

import os
import sys
import time
import logging
import numpy as np
import sounddevice as sd
import webrtcvad
import requests
import json
from pathlib import Path
from dotenv import load_dotenv

# ==================== PATHS & ENV ====================

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent

env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    load_dotenv(env_path)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("LOCAL-AGENT")

# ==================== CONFIG ====================

SAMPLE_RATE = 16000
CHANNELS = 1
VAD_MODE = 3                # Most aggressive
SILENCE_THRESHOLD = 1.5     # Seconds of silence = end of speech
CHUNK_DURATION = 0.03       # 30ms chunks

INDIC_LANGS = {"hi", "gu", "te", "ta", "kn", "ml", "pa", "mr"}

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = """You are a friendly, natural Indian voice assistant in a real-time phone call.

Core rules:
- Speak exactly like a helpful human friend — warm, casual, never robotic.
- Keep every reply to short, simple and to the point.
- Vary your phrasing naturally.

Language rules:
- Always reply in the same language (or mix) the user is using right now.

STT is sometimes imperfect — understand intent, not exact words.
If unclear, ask one short clarification question.

Never be verbose. Never use lists or markdown.
If user says stop/bas/chup/enough — politely stop and confirm.

Tone: Friendly, calm, helpful, slightly playful."""

# ==================== MODEL LOADING ====================

whisper_model = None
indic_model = None
piper_voices = {}

def load_all_models():
    """Load Whisper, IndicConformer, and Piper TTS models."""
    global whisper_model, indic_model, piper_voices

    # ── Whisper (language detection + English transcription) ──
    print("\n🧠 Loading Whisper small...")
    start = time.time()
    from faster_whisper import WhisperModel
    whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
    print(f"✅ Whisper loaded ({time.time() - start:.1f}s)")

    # ── IndicConformer (Indian language transcription) ──
    print("🧠 Loading IndicConformer-600M...")
    start = time.time()
    try:
        import torch
        from transformers import AutoModel
        indic_model = AutoModel.from_pretrained(
            "ai4bharat/indic-conformer-600m-multilingual",
            trust_remote_code=True
        )
        print(f"✅ IndicConformer loaded ({time.time() - start:.1f}s)")
    except Exception as e:
        print(f"⚠️  IndicConformer failed to load: {e}")
        print("   Indian languages will fallback to Whisper")
        indic_model = None

    # ── Piper TTS ──
    print("🧠 Loading Piper TTS voices...")
    try:
        from piper.voice import PiperVoice
        piper_dir = PROJECT_ROOT / "piper"

        en_path = piper_dir / "en_US-lessac-medium.onnx"
        hi_path = piper_dir / "hi_IN-priyamvada-medium.onnx"

        if en_path.exists():
            piper_voices["en"] = PiperVoice.load(str(en_path))
            print(f"✅ Piper EN loaded")
        if hi_path.exists():
            piper_voices["hi"] = PiperVoice.load(str(hi_path))
            print(f"✅ Piper HI loaded")

        if not piper_voices:
            print("⚠️  No Piper voices found — TTS will be disabled")
    except Exception as e:
        print(f"⚠️  Piper TTS failed: {e}")


# ==================== VAD (VOICE ACTIVITY DETECTION) ====================

vad = webrtcvad.Vad(VAD_MODE)

def is_speech(frame: bytes) -> bool:
    try:
        return vad.is_speech(frame, SAMPLE_RATE)
    except:
        return False

def record_with_vad() -> np.ndarray:
    """
    Records audio from mic using WebRTC VAD.
    Returns float32 numpy array at 16kHz (ready for Whisper/IndicConformer).
    """
    print("\n🎤 Listening... Speak now")

    buffer = []
    silence_frames = 0
    recording = False

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype='int16',
        blocksize=int(SAMPLE_RATE * CHUNK_DURATION)
    )
    stream.start()

    try:
        while True:
            audio_chunk, _ = stream.read(stream.blocksize)
            frame = audio_chunk.tobytes()

            if is_speech(frame):
                if not recording:
                    print("🗣️  Speech detected...")
                    recording = True
                silence_frames = 0
                buffer.extend(audio_chunk.flatten())
            else:
                if recording:
                    silence_frames += 1
                    buffer.extend(audio_chunk.flatten())

                    if silence_frames > int(SILENCE_THRESHOLD / CHUNK_DURATION):
                        print("⏹️  Speech ended")
                        break
    finally:
        stream.stop()
        stream.close()

    if not buffer:
        return np.array([], dtype=np.float32)

    # Convert int16 → float32 for Whisper/IndicConformer
    audio_int16 = np.array(buffer, dtype=np.int16)
    audio_float32 = audio_int16.astype(np.float32) / 32768.0
    duration = len(audio_float32) / SAMPLE_RATE
    print(f"✅ Recorded {duration:.1f}s")

    return audio_float32


# ==================== HYBRID STT ====================

def detect_language(audio: np.ndarray) -> tuple:
    """
    Use Whisper to detect language from audio.
    Returns (language_code, confidence).
    """
    segments, info = whisper_model.transcribe(
        audio,
        language=None,
        beam_size=1,
        best_of=1,
        temperature=0.0,
        vad_filter=True,
    )

    # Must consume at least one segment to populate info
    for _ in segments:
        break

    lang = info.language or "en"
    prob = float(info.language_probability)
    return lang, prob


def transcribe_whisper(audio: np.ndarray, lang: str = None) -> dict:
    """Transcribe using Whisper (for English or fallback)."""
    segments, info = whisper_model.transcribe(
        audio,
        language=lang,
        beam_size=5,
        temperature=0.0,
        vad_filter=True,
    )

    text = " ".join([s.text.strip() for s in segments]).strip()
    return {
        "text": text,
        "language": info.language or lang or "en",
        "engine": "whisper"
    }


def transcribe_indic(audio: np.ndarray, lang: str) -> dict:
    """Transcribe using IndicConformer (for Indian languages)."""
    import torch

    if indic_model is None:
        logger.warning("IndicConformer not loaded → falling back to Whisper")
        return transcribe_whisper(audio, lang)

    try:
        audio_tensor = torch.from_numpy(audio).float()
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)  # [1, samples]

        with torch.no_grad():
            output = indic_model(audio_tensor, lang)

        text = " ".join(output) if isinstance(output, list) else str(output)

        return {
            "text": text.strip(),
            "language": lang,
            "engine": "indic-conformer"
        }
    except Exception as e:
        logger.error(f"IndicConformer failed: {e}")
        return transcribe_whisper(audio, lang)


def hybrid_transcribe(audio: np.ndarray, forced_lang: str = None) -> dict:
    """
    Full hybrid pipeline:
    - If forced_lang is set: skip detection, use that language directly
    - Otherwise: Whisper detects language, then routes accordingly
    """
    t0 = time.time()

    if forced_lang:
        # Language forced by user — skip detection
        lang = forced_lang
        prob = 1.0
        detect_time = 0.0
        logger.info(f"🔒 Language forced: {lang} (skipping detection)")
    else:
        # Step 1: Auto-detect language with Whisper
        lang, prob = detect_language(audio)
        detect_time = time.time() - t0

    # Step 2: Route to correct engine
    if lang in INDIC_LANGS:
        result = transcribe_indic(audio, lang)
    else:
        result = transcribe_whisper(audio, lang)

    total_time = time.time() - t0

    result["detected_lang"] = lang
    result["confidence"] = round(prob, 3)
    result["detect_time"] = round(detect_time, 2)
    result["total_time"] = round(total_time, 2)

    return result


# ==================== LLM (GROQ) ====================

conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]

def chat_with_llm(user_text: str, language: str) -> str:
    """
    Send user text to Groq LLM and get streaming response.
    Returns the full reply text.
    """
    global conversation_history

    # Add hidden language directive
    augmented_text = (
        f"{user_text}\n\n"
        f"[System directive: Respond in language '{language}'. "
        f"If this is an Indian regional language (gu, te, ta, kn, ml, pa), "
        f"write the response in Devanagari script so the TTS engine can speak it.]"
    )

    conversation_history.append({"role": "user", "content": augmented_text})

    # Trim history (keep system + last 20 turns)
    if len(conversation_history) > 21:
        system = conversation_history[0]
        conversation_history = [system] + conversation_history[-20:]

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": conversation_history,
                "temperature": 0.75,
                "max_tokens": 280,
                "stream": True,
            },
            stream=True,
            timeout=15,
        )
        response.raise_for_status()

        full_reply = ""
        print("🤖 AI: ", end="", flush=True)

        for line in response.iter_lines():
            line = line.decode("utf-8").strip()
            if not line or line == "data: [DONE]":
                continue
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    token = data["choices"][0]["delta"].get("content", "")
                    if token:
                        full_reply += token
                        print(token, end="", flush=True)
                except (json.JSONDecodeError, KeyError):
                    pass

        print()  # newline after streaming
        conversation_history.append({"role": "assistant", "content": full_reply})
        return full_reply

    except Exception as e:
        logger.error(f"LLM error: {e}")
        return "Sorry, I couldn't process that."


# ==================== TTS (PIPER) ====================

def speak(text: str, language: str = "en"):
    """
    Synthesize text with Piper TTS and play through speakers.
    Routes: English → en voice, anything else → hi voice (Devanagari).
    """
    if not piper_voices:
        print("⚠️  No TTS voices loaded — skipping playback")
        return

    # Choose voice: English uses EN, everything else uses HI
    if language == "en":
        voice = piper_voices.get("en", piper_voices.get("hi"))
    else:
        voice = piper_voices.get("hi", piper_voices.get("en"))

    if not voice:
        print("⚠️  No suitable voice found")
        return

    try:
        # Collect all audio chunks
        audio_chunks = []
        for chunk in voice.synthesize(text):
            audio_chunks.append(chunk.audio_int16_bytes)

        if not audio_chunks:
            return

        audio_bytes = b"".join(audio_chunks)
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16)

        print("🔊 Speaking...")
        # Piper outputs at 22050 Hz
        sd.play(audio_np, samplerate=22050, blocking=True)
        print("✅ Done speaking\n")

    except Exception as e:
        logger.error(f"TTS playback error: {e}")


# ==================== FILE MODE ====================

def test_file(filepath: str):
    """Transcribe a single audio file (no LLM/TTS)."""
    import soundfile as sf

    path = Path(filepath)
    if not path.exists():
        print(f"❌ File not found: {filepath}")
        return

    print(f"\n📁 Processing: {path.name}")

    audio, sr = sf.read(str(path), dtype="float32")

    # Resample to 16kHz if needed
    if sr != SAMPLE_RATE:
        import torchaudio
        import torch
        audio_tensor = torch.from_numpy(audio).float()
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        audio_tensor = resampler(audio_tensor)
        audio = audio_tensor.squeeze(0).numpy()

    # Mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    result = hybrid_transcribe(audio)

    print("\n" + "=" * 60)
    print("  TRANSCRIPTION RESULT")
    print("=" * 60)
    print(f"  📝 Text     : {result['text']}")
    print(f"  🌐 Language : {result['language']} (detected: {result['detected_lang']}, conf: {result['confidence']:.1%})")
    print(f"  ⚙️  Engine   : {result['engine']}")
    print(f"  ⏱️  Time     : {result['total_time']}s (detect: {result['detect_time']}s)")
    print("=" * 60 + "\n")


# ==================== INTERACTIVE VOICE LOOP ====================

def voice_loop(forced_lang: str = None):
    """
    Full end-to-end voice conversation loop:
    Mic → VAD → Hybrid STT → Groq LLM → Piper TTS → Speaker
    """
    print("\n" + "=" * 60)
    print("  🚀 LOCAL VOICE AGENT — Ready!")
    if forced_lang:
        print(f"  🔒 Language: {forced_lang} (forced, no auto-detect)")
    else:
        print("  🌐 Language: Auto-detect")
    print("  Pipeline: Mic → Whisper/IndicConformer → Groq → Piper")
    print("  Press Ctrl+C to exit")
    print("=" * 60)

    if not GROQ_API_KEY:
        print("\n⚠️  GROQ_API_KEY not set! LLM will not work.")
        print("   Set it in .env or export GROQ_API_KEY=...")

    turn = 0

    while True:
        try:
            turn += 1
            audio = record_with_vad()

            # Skip very short audio (< 0.5s)
            if len(audio) < SAMPLE_RATE * 0.5:
                print("⚠️  Too short, ignoring...")
                continue

            # ── STT ──
            print("📝 Transcribing...")
            stt_start = time.time()
            result = hybrid_transcribe(audio, forced_lang=forced_lang)
            stt_time = time.time() - stt_start

            text = result["text"]
            lang = result["language"]
            engine = result["engine"]

            if not text.strip():
                print("⚠️  Empty transcription, try again...")
                continue

            print(f"\n{'─' * 50}")
            print(f"👤 You [{lang} | {engine} | {stt_time:.1f}s]: {text}")
            print(f"{'─' * 50}")

            # ── Check for exit commands ──
            lower = text.lower().strip()
            if any(cmd in lower for cmd in ["quit", "exit", "bye", "goodbye", "band karo", "bye bye"]):
                farewell = "Bye! Take care!" if lang == "en" else "अच्छा, बाय बाय! ख्याल रखना!"
                print(f"🤖 AI: {farewell}")
                speak(farewell, lang)
                print("\n👋 Conversation ended.")
                break

            # ── LLM ──
            llm_start = time.time()
            reply = chat_with_llm(text, lang)
            llm_time = time.time() - llm_start

            if not reply.strip():
                continue

            print(f"   ⏱️  LLM: {llm_time:.1f}s")

            # ── TTS ──
            tts_start = time.time()
            speak(reply, lang)
            tts_time = time.time() - tts_start

            # ── Pipeline summary ──
            total = stt_time + llm_time + tts_time
            print(f"   📊 Pipeline: STT={stt_time:.1f}s | LLM={llm_time:.1f}s | TTS={tts_time:.1f}s | Total={total:.1f}s")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error in voice loop: {e}")
            time.sleep(0.5)


# ==================== MAIN ====================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Local Voice Agent — Hybrid STT + LLM + TTS",
        usage="%(prog)s [lang] [--file FILE]"
    )
    parser.add_argument(
        "lang",
        nargs="?",
        default=None,
        help="Force language (e.g. hi, gu, ta, en). Omit for auto-detect."
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        default=None,
        help="Audio file to transcribe (skips LLM/TTS)"
    )

    args = parser.parse_args()

    forced_lang = args.lang

    print("\n" + "═" * 60)
    print("  🎙️  LOCAL VOICE AGENT")
    if forced_lang:
        print(f"  🔒 Language: {forced_lang} (forced)")
        if forced_lang in INDIC_LANGS:
            print(f"  ⚙️  Engine: IndicConformer (skipping Whisper detect)")
        else:
            print(f"  ⚙️  Engine: Whisper")
    else:
        print("  🌐 Language: Auto-detect (Whisper → route)")
    print("  Groq LLM | Piper TTS")
    print("═" * 60)

    # Load all models
    load_all_models()

    if args.file:
        # File mode — just transcribe, no conversation
        test_file(args.file)
    else:
        # Interactive voice loop
        voice_loop(forced_lang=forced_lang)


if __name__ == "__main__":
    main()