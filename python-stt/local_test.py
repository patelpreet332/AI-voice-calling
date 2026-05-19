#!/usr/bin/env python3

import os, sys, time, json, logging, threading, queue
import numpy as np
import sounddevice as sd
import webrtcvad
import requests
from pathlib import Path
from dotenv import load_dotenv



ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
load_dotenv(PROJECT_ROOT / ".env")

SAMPLE_RATE = 16000
CHANNELS = 1

VAD_MODE = 1
CHUNK_DURATION = 0.03
MIN_SPEECH_SEC = 0.8
SILENCE_SEC = 1.0
MAX_AUDIO_SEC = 20

INDIC_LANGS = {"hi", "gu", "te", "ta"}
VALID_LANGS = {"en", "hi", "te", "ta", "gu"}

FORCE_LANG = None

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = """You are a real-time voice assistant speaking over a phone call.
Keep replies natural, human-like and brief.
Keep your response short and to the point, like a helpful friend on a call but should include all user answer.
Always reply in same language as user.
Use casual spoken tone.
If unclear, ask short clarification.
Never say you are an AI.
"""



logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("VOICE")



audio_queue = queue.Queue()
text_queue = queue.Queue()
reply_queue = queue.Queue()

conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
conversation_lock = threading.Lock()

session = requests.Session()

whisper_model = None
indic_model = None
piper_voices = {}


def warmup():
    log.info("[WARMUP] starting")

    dummy = np.zeros(SAMPLE_RATE, dtype=np.float32)

    whisper_model.transcribe(dummy)
    if indic_model:
        import torch
        indic_model(torch.zeros(1, SAMPLE_RATE), "hi")
    if "en" in piper_voices:
        for _ in piper_voices["en"].synthesize("hello"):
            break

    log.info("[WARMUP] done")

def load_models():
    global whisper_model, indic_model, piper_voices

    from faster_whisper import WhisperModel
    whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
    log.info("[INIT] Whisper ready")

    try:
        from transformers import AutoModel
        indic_model = AutoModel.from_pretrained(
            "ai4bharat/indic-conformer-600m-multilingual",
            trust_remote_code=True
        )
        log.info("[INIT] Indic ready")
    except Exception as e:
        log.error(f"[INIT] Indic failed: {e}")
        indic_model = None

    from piper.voice import PiperVoice

    voices = {
        "en": "en_US-lessac-medium.onnx",
        "hi": "hi_IN-priyamvada-medium.onnx",
        "te": "te_IN-padmavathi-medium.onnx",
        "gu": "gu_epoch229.onnx",
    }

    for lang, file in voices.items():
        path = PROJECT_ROOT / "piper" / file
        if path.exists():
            piper_voices[lang] = PiperVoice.load(str(path))
            log.info(f"[TTS] Loaded {lang}")



vad = webrtcvad.Vad(VAD_MODE)

def is_speech(frame):
    try:
        return vad.is_speech(frame, SAMPLE_RATE)
    except:
        return False

def mic_worker():
    while True:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype='int16',
            blocksize=int(SAMPLE_RATE * CHUNK_DURATION)
        )
        stream.start()

        buffer = []
        silence = 0
        speech_frames = 0

        min_frames = int(MIN_SPEECH_SEC / CHUNK_DURATION)
        silence_frames = int(SILENCE_SEC / CHUNK_DURATION)

        log.info("🎤 Listening...")

        while True:
            chunk, _ = stream.read(stream.blocksize)
            frame = chunk.tobytes()

            if is_speech(frame):
                buffer.extend(chunk.flatten())
                speech_frames += 1
                silence = 0
            else:
                if speech_frames > min_frames:
                    silence += 1
                    buffer.extend(chunk.flatten())
                    if silence > silence_frames:
                        break
                else:
                    buffer = []
                    speech_frames = 0

        stream.stop()
        stream.close()

        if not buffer:
            continue

        audio = np.array(buffer, dtype=np.int16).astype(np.float32) / 32768.0

        if len(audio) > SAMPLE_RATE * MAX_AUDIO_SEC:
            audio = audio[:SAMPLE_RATE * MAX_AUDIO_SEC]

        audio_queue.put(audio)



def stt_worker():
    while True:
        audio = audio_queue.get()
        start = time.time()


        if FORCE_LANG:
            lang = FORCE_LANG

            if lang in INDIC_LANGS and indic_model:
                import torch
                audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
                with torch.no_grad():
                    out = indic_model(audio_tensor, lang)
                text = " ".join(out) if isinstance(out, list) else str(out)
                engine = "indic"

            else:
                segments, _ = whisper_model.transcribe(
                    audio,
                    language=lang,
                    beam_size=1,
                    temperature=0.0
                )
                text = "".join(s.text for s in segments)
                engine = "whisper"


        else:
            segments, info = whisper_model.transcribe(
                audio,
                beam_size=1,
                temperature=0.0
            )

            lang = info.language or "en"
            if lang not in VALID_LANGS:
                lang = "hi"

            if lang in INDIC_LANGS and indic_model:
                import torch
                audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
                with torch.no_grad():
                    out = indic_model(audio_tensor, lang)
                text = " ".join(out) if isinstance(out, list) else str(out)
                engine = "indic"
            else:
                text = "".join(s.text for s in segments)
                engine = "whisper"

        text = text.strip()

        if not text or len(text) < 2:
            continue

        log.info(f"[USER] ({engine}/{lang}) {time.time()-start:.2f}s → {text}")
        text_queue.put((text, lang))



def llm_worker():
    while True:
        text, lang = text_queue.get()
        start = time.time()

        user_input = f"{text}\n\n[Reply in {lang}, short spoken.]"

        with conversation_lock:
            conversation.append({"role": "user", "content": user_input})
            conversation[:] = conversation[-10:]

        res = session.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": GROQ_MODEL,
                "messages": conversation,
                "temperature": 0.6,
                "max_tokens": 150,
                "stream": True,
            },
            stream=True,
        )

        buffer = ""
        full = ""
        first_token = None

        for line in res.iter_lines():
            if not line:
                continue

            line = line.decode("utf-8")

            if line.startswith("data: "):
                if "[DONE]" in line:
                    break

                data = json.loads(line[6:])
                token = data["choices"][0]["delta"].get("content", "")

                if token:
                    if first_token is None:
                        first_token = time.time()

                    buffer += token
                    full += token
                    if len(buffer) > 60 or buffer.endswith((".", "?", "!")):
                        reply_queue.put((buffer, lang, False))
                        buffer = ""

        if buffer:
            reply_queue.put((buffer, lang, True))

        total = time.time() - start
        first_latency = (first_token - start) if first_token else 0

        log.info(f"[LLM] first={first_latency:.2f}s total={total:.2f}s")
        log.info(f"[ASSISTANT] → {full}")
        with conversation_lock:
            conversation.append({"role": "assistant", "content": full})



def tts_worker():
    stream = sd.OutputStream(samplerate=22050, channels=1, dtype='int16')
    stream.start()

    buffer = ""
    current_lang = "en"

    while True:
        text, lang, is_final = reply_queue.get()

        current_lang = lang
        buffer += text

        should_flush = (
            len(buffer) > 80 or
            buffer.endswith((".", "?", "!")) or
            is_final
        )

        if not should_flush:
            continue

        voice = piper_voices.get(current_lang[:2], piper_voices.get("hi"))
        if not voice:
            buffer = ""
            continue

        log.info(f"[TTS] speaking chunk → {buffer}")

        for chunk in voice.synthesize(buffer):
            audio = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
            stream.write(audio)

        buffer = ""



def main():
    global FORCE_LANG

    if len(sys.argv) > 1:
        FORCE_LANG = sys.argv[1]

    load_models()
    warmup()

    threading.Thread(target=mic_worker, daemon=True).start()
    threading.Thread(target=stt_worker, daemon=True).start()
    threading.Thread(target=llm_worker, daemon=True).start()
    threading.Thread(target=tts_worker, daemon=True).start()

    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()