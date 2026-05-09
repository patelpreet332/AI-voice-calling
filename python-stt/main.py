from faster_whisper import WhisperModel
from fastapi import FastAPI, UploadFile, HTTPException
import shutil
import uuid
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Optimized 3-Language STT")

# Load model once
model = WhisperModel(
    "medium",           # or "medium" / "small.en" if mostly English
    device="cpu",
    compute_type="int8"
)

# Your allowed languages
ALLOWED_LANGS = {"en", "hi", "gu"}

@app.post("/transcribe")
async def transcribe(file: UploadFile):
    file_id = str(uuid.uuid4())
    input_path = f"temp_{file_id}.wav"

    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        segments, info = model.transcribe(
            input_path,
            language="hi",           # auto detection
            beam_size=1,
            best_of=1,
            temperature=0.0,         # more deterministic
            vad_filter=True,         # highly recommended
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
            if confidence < 0.75:                     # low confidence
                # Fallback to most probable among your 3 languages
                # You can also run a second pass with each language and pick the best
                final_lang = "hi" if "hi" in text or any(ord(c) > 127 for c in text) else "en"
                logger.warning(f"Unknown language {detected_lang} → fallback to {final_lang}")
            else:
                # High confidence but wrong language? Rare, but keep original
                pass

        return {
            "text": text,
            "language": final_lang,
            "confidence": round(confidence, 3),
            "duration": round(info.duration, 2),
            "raw_detected": detected_lang
        }

    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(500, detail=str(e))
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)