from faster_whisper import WhisperModel
from fastapi import FastAPI, UploadFile, HTTPException
import numpy as np
import soundfile as sf
import io
import logging
import os
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from piper.voice import PiperVoice

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Multi-Language STT & TTS")

# ========================== MODELS ==========================
model = WhisperModel(
    "small",           # Better for Hindi than "small"
    device="cpu",
    compute_type="int8"
)

# Allowed languages
ALLOWED_LANGS = {"en", "hi", "gu"}

# ======================= PIPER TTS MODELS =======================
PIPER_MODELS = {}

def load_piper_model(model_path: str, name: str):
    try:
        logger.info(f"Loading Piper model: {name} from {model_path}")
        voice = PiperVoice.load(model_path)
        PIPER_MODELS[name] = voice
        logger.info(f"✅ {name} TTS loaded successfully")
        return voice
    except Exception as e:
        logger.error(f"❌ Failed to load {name}: {e}")
        return None

# Load English and Hindi models
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "piper"))

EN_MODEL_PATH = os.path.join(BASE_DIR, "en_US-lessac-medium.onnx")
# HI_MODEL_PATH = os.path.join(BASE_DIR, "hi_IN-pratham-medium.onnx")   # Change filename if different

piper_en = load_piper_model(EN_MODEL_PATH, "en")
# piper_hi = load_piper_model(HI_MODEL_PATH, "hi")

# Fallback to English if Hindi not available
DEFAULT_PIPER = piper_en

class TTSRequest(BaseModel):
    text: str
    language: str = "en"   # New: accept language

@app.post("/tts")
async def generate_tts(req: TTSRequest):
    if not req.language:
        req.language = "en"
    
    voice = PIPER_MODELS.get(req.language, DEFAULT_PIPER)
    
    if not voice:
        logger.warning(f"Voice for '{req.language}' not loaded, using English fallback")
        voice = DEFAULT_PIPER

    def audio_stream():
        for chunk in voice.synthesize(req.text):
            yield chunk.audio_int16_bytes

    return StreamingResponse(audio_stream(), media_type="audio/pcm")

@app.post("/transcribe")
async def transcribe(file: UploadFile):
    try:
        file_bytes = await file.read()
        audio_stream = io.BytesIO(file_bytes)

        audio_data, sample_rate = sf.read(audio_stream, dtype="float32")

        # Better language detection for multilingual
        segments, info = model.transcribe(
            audio_data,
            language="en",           # Auto detect
            beam_size=1,
            best_of=1,
            temperature=0.0,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=400,
                threshold=0.5
            ),
            word_timestamps=False
        )

        text = " ".join([seg.text.strip() for seg in segments]).strip()

        detected_lang = info.language or "en"
        confidence = float(info.language_probability)

        logger.info(f"Raw detected: {detected_lang} | Confidence: {confidence:.3f} | Text: {text[:80]}...")

        # === Improved Language Logic ===
        final_lang = detected_lang

        if detected_lang not in ALLOWED_LANGS:
            if confidence < 0.70 or detected_lang == "unknown":
                # Fallback logic
                has_devanagari = any(ord(c) > 127 for c in text)
                final_lang = "hi" if has_devanagari else "en"
                logger.warning(f"Unknown lang {detected_lang} → fallback to {final_lang}")
            else:
                final_lang = "en"

        # Force Hindi if strong Hindi indicators
        if "hi" in text.lower() or any(ord(c) in range(0x0900, 0x097F) for c in text):
            final_lang = "hi"

        return {
            "text": text,
            "language": final_lang,
            "confidence": round(confidence, 3),
            "duration": round(info.duration, 2),
            "raw_detected": detected_lang
        }

    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        raise HTTPException(500, detail=str(e))