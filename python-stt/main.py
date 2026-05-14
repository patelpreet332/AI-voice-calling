from faster_whisper import WhisperModel
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import soundfile as sf
import numpy as np
import io
import time
import logging
import torch
import os

from piper.voice import PiperVoice

# ==================== LOGGING ====================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HYBRID-STT")

app = FastAPI(title="Dynamic STT + TTS (Whisper + IndicConformer)")

# ==================== CONFIG ====================
INDIC_LANGS = {"hi", "gu", "te", "ta", "kn", "ml", "pa"}

# ==================== MODELS ====================

logger.info("🧠 Loading Whisper...")
whisper = WhisperModel("small", device="cpu", compute_type="int8")
logger.info("✅ Whisper loaded")

indic_model = None

def load_indic():
    global indic_model
    try:
        from transformers import AutoModel
        logger.info("🧠 Loading IndicConformer...")
        indic_model = AutoModel.from_pretrained(
            "ai4bharat/indic-conformer-600m-multilingual",
            trust_remote_code=True
        )
        logger.info("✅ IndicConformer loaded")
    except Exception as e:
        logger.warning(f"⚠️ IndicConformer failed: {e}")

load_indic()

# ==================== TTS ====================

PIPER_MODELS = {}

def load_piper(path, name):
    try:
        voice = PiperVoice.load(path)
        PIPER_MODELS[name] = voice
        logger.info(f"✅ Loaded TTS: {name}")
    except Exception as e:
        logger.warning(f"⚠️ Failed loading {name}: {e}")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "piper"))

load_piper(os.path.join(BASE_DIR, "en_US-lessac-medium.onnx"), "en")
load_piper(os.path.join(BASE_DIR, "hi_IN-priyamvada-medium.onnx"), "hi")

DEFAULT_TTS = PIPER_MODELS.get("en")


class TTSRequest(BaseModel):
    text: str
    language: str = "en"


@app.post("/tts")
async def tts(req: TTSRequest):
    if not req.text.strip():
        raise HTTPException(400, "Empty text")

    voice = PIPER_MODELS.get(req.language, DEFAULT_TTS)

    def stream():
        for chunk in voice.synthesize(req.text):
            yield chunk.audio_int16_bytes

    return StreamingResponse(stream(), media_type="audio/pcm")


# ==================== STT CORE ====================

def detect_language(audio):
    """
    Whisper-based language detection
    """
    segments, info = whisper.transcribe(
        audio,
        language=None,
        beam_size=1,
        best_of=1
    )

    # trigger detection
    for _ in segments:
        break

    lang = info.language or "en"
    prob = float(info.language_probability)

    logger.info(f"🌐 [WHISPER DETECT] → {lang} ({prob:.2%})")

    return lang, prob


def transcribe_whisper(audio, lang=None):
    segments, info = whisper.transcribe(audio, language=lang)

    text = " ".join([s.text.strip() for s in segments]).strip()

    return {
        "text": text,
        "language": info.language or lang or "en",
        "engine": "whisper"
    }


def transcribe_indic(audio, lang):
    if indic_model is None:
        logger.warning("⚠️ IndicConformer unavailable → fallback to Whisper")
        return transcribe_whisper(audio, lang)

    try:
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)

        with torch.no_grad():
            output = indic_model(audio, lang)

        text = " ".join(output) if isinstance(output, list) else str(output)

        return {
            "text": text.strip(),
            "language": lang,
            "engine": "indic-conformer"
        }

    except Exception as e:
        logger.error(f"❌ IndicConformer failed: {e}")
        return transcribe_whisper(audio, lang)


# ==================== STT API ====================

@app.post("/transcribe")
async def transcribe(file: UploadFile):
    try:
        audio_bytes = await file.read()
        audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")

        # Convert to mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        start = time.time()

        # STEP 1: Detect language
        lang, prob = detect_language(audio)

        # STEP 2: Route
        if lang in INDIC_LANGS:
            logger.info(f"🇮🇳 [ROUTE] Indic → {lang}")
            result = transcribe_indic(audio, lang)
        else:
            logger.info(f"🇬🇧 [ROUTE] Whisper → {lang}")
            result = transcribe_whisper(audio, lang)

        result["detected_language"] = lang
        result["confidence"] = round(prob, 3)
        result["time"] = round(time.time() - start, 2)

        logger.info(
            f"📊 Done | Lang={lang} | Engine={result['engine']} | Time={result['time']}s"
        )

        return result

    except Exception as e:
        logger.error(f"❌ Transcription error: {e}")
        raise HTTPException(500, str(e))


# ==================== HEALTH ====================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models": {
            "whisper": "loaded",
            "indic_conformer": "loaded" if indic_model else "not_loaded"
        },
        "tts": list(PIPER_MODELS.keys())
    }


# ==================== RUN ====================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        access_log=False
    )