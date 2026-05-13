from faster_whisper import WhisperModel
from fastapi import FastAPI, UploadFile, HTTPException
import uvicorn
import soundfile as sf
import io
import logging
import os
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from piper.voice import PiperVoice
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Multi-Language STT & TTS (Improved)")

# ========================== CONFIG ==========================
ALLOWED_LANGS = {"en", "hi", "gu", "te", "ta", "kn", "ml", "bn", "pa", "mr", "ur", "or", "as"}

# ========================== WHISPER ==========================
model = WhisperModel(
    "medium",   # change to large-v3 if GPU available
    device="cpu",
    compute_type="int8"
)

# ======================= PIPER TTS ==========================
PIPER_MODELS = {}

def load_piper_model(model_path: str, name: str):
    try:
        voice = PiperVoice.load(model_path)
        PIPER_MODELS[name] = voice
        logger.info(f"✅ Loaded TTS: {name}")
        return voice
    except Exception as e:
        logger.warning(f"⚠️ Failed loading {name}: {e}")
        return None

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "piper"))

EN_MODEL_PATH = os.path.join(BASE_DIR, "en_US-lessac-medium.onnx")
HI_MODEL_PATH = os.path.join(BASE_DIR, "hi_IN-priyamvada-medium.onnx")

piper_en = load_piper_model(EN_MODEL_PATH, "en")
piper_hi = load_piper_model(HI_MODEL_PATH, "hi")

DEFAULT_PIPER = piper_en

# ======================= LANGUAGE HELPERS =======================

def detect_script_language(text: str):
    """Detect Indic language from Unicode ranges"""
    for c in text:
        code = ord(c)

        if 0x0900 <= code <= 0x097F:
            return "hi"
        elif 0x0A80 <= code <= 0x0AFF:
            return "gu"
        elif 0x0B80 <= code <= 0x0BFF:
            return "ta"
        elif 0x0C00 <= code <= 0x0C7F:
            return "te"
        elif 0x0C80 <= code <= 0x0CFF:
            return "kn"
        elif 0x0D00 <= code <= 0x0D7F:
            return "ml"
        elif 0x0980 <= code <= 0x09FF:
            return "bn"
        elif 0x0A00 <= code <= 0x0A7F:
            return "pa"
    return None


def resolve_language(detected_lang, confidence, text):
    """
    Combines Whisper detection + script fallback
    """
    if detected_lang in ALLOWED_LANGS and confidence >= 0.75:
        return detected_lang

    script_lang = detect_script_language(text)
    if script_lang:
        return script_lang

    return "en"


def convert_to_devanagari(text: str, lang: str):
    """Transliterates regional Indic scripts deterministically into Devanagari script"""
    mapping = {
        "te": sanscript.TELUGU,
        "ta": sanscript.TAMIL,
        "kn": sanscript.KANNADA,
        "ml": sanscript.MALAYALAM,
        "bn": sanscript.BENGALI,
        "gu": sanscript.GUJARATI,
        "pa": sanscript.GURMUKHI,
    }

    if lang in mapping:
        return transliterate(text, mapping[lang], sanscript.DEVANAGARI)

    return text

# ======================= TTS =======================

class TTSRequest(BaseModel):
    text: str
    language: str = "en"


@app.post("/tts")
async def generate_tts(req: TTSRequest):
    if not req.text.strip():
        raise HTTPException(400, "Text is empty")

    text = req.text
    lang = req.language or "en"

    # Detect script override if regional characters are present
    script_lang = detect_script_language(text)
    if script_lang and script_lang != "hi":
        lang = script_lang

    # Transliteration Layer: Convert if not Hindi/English natively
    if lang not in ["hi", "en"]:
        try:
            text = convert_to_devanagari(text, lang)
            logger.info(f"✨ Transliterated ({lang} -> Devanagari): {text[:60]}")
        except Exception as e:
            logger.warning(f"⚠️ Transliteration failed: {e}")
        lang = "hi"  # route natively to the Hindi Piper voice model

    # Select voice
    voice = PIPER_MODELS.get(lang)

    # Smart fallback
    if not voice:
        if lang != "en" and "hi" in PIPER_MODELS:
            voice = PIPER_MODELS["hi"]
        else:
            voice = DEFAULT_PIPER

    def stream():
        for chunk in voice.synthesize(text):
            yield chunk.audio_int16_bytes

    return StreamingResponse(stream(), media_type="audio/pcm")

# ======================= STT =======================

@app.post("/transcribe")
async def transcribe(file: UploadFile):
    try:
        audio_bytes = await file.read()
        audio_stream = io.BytesIO(audio_bytes)

        audio_data, sample_rate = sf.read(audio_stream, dtype="float32")

        segments, info = model.transcribe(
            audio_data,
            language=None,
            beam_size=5,
            temperature=[0.0, 0.2, 0.4],
            repetition_penalty=1.2,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=400,
                threshold=0.5
            ),
            no_speech_threshold=0.6,
            initial_prompt=None,  # removed bias
            word_timestamps=False
        )

        text = " ".join([seg.text.strip() for seg in segments]).strip()

        detected_lang = info.language or "unknown"
        confidence = float(info.language_probability)

        final_lang = resolve_language(detected_lang, confidence, text)
        script_lang = detect_script_language(text)

        logger.info(
            f"Detected={detected_lang} | Final={final_lang} | Conf={confidence:.2f} | Text={text[:60]}"
        )

        return {
            "text": text,
            "language": final_lang,
            "confidence": round(confidence, 3),
            "raw_detected": detected_lang,
            "script_detected": script_lang,
            "duration": round(info.duration, 2)
        }

    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(500, str(e))

# ======================= RUN =======================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        access_log=False
    )