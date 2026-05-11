from faster_whisper import WhisperModel
from fastapi import FastAPI, UploadFile, HTTPException
import numpy as np
import soundfile as sf
import io
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Optimized 3-Language STT")

# Load model once — "small" is fast and sufficient for phone audio
model = WhisperModel(
    "small",           
    device="cpu",
    compute_type="int8"
)

# Your allowed languages
ALLOWED_LANGS = {"en", "hi", "gu"}

@app.post("/transcribe")
async def transcribe(file: UploadFile):
    try:
        # Read file directly into memory — no disk I/O
        file_bytes = await file.read()
        audio_stream = io.BytesIO(file_bytes)

        # Read WAV from memory into numpy array
        # Note: 'soundfile' reads the WAV container in memory
        audio_data, sample_rate = sf.read(audio_stream, dtype="float32")

        # faster-whisper accepts numpy arrays directly
        segments, info = model.transcribe(
            audio_data,
            language="en",           
            beam_size=1,
            best_of=1,
            temperature=0.0,         
            vad_filter=True,         
            vad_parameters=dict(
                min_silence_duration_ms=500,
                threshold=0.5
            ),
            word_timestamps=False
        )

        text = " ".join([seg.text.strip() for seg in segments]).strip()

        detected_lang = info.language
        confidence = float(info.language_probability)

        logger.info(f"Raw detected: {detected_lang} | Confidence: {confidence:.3f}")

        # === Language Correction Logic ===
        final_lang = detected_lang
        if detected_lang not in ALLOWED_LANGS:
            if confidence < 0.75:                     
                final_lang = "hi" if "hi" in text or any(ord(c) > 127 for c in text) else "en"
                logger.warning(f"Unknown language {detected_lang} → fallback to {final_lang}")

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