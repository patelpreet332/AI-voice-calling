from faster_whisper import WhisperModel
from fastapi import FastAPI, UploadFile, HTTPException, Form
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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HYBRID-STT")

app = FastAPI(title="Dynamic STT + TTS (Whisper + IndicConformer)")


SAMPLE_RATE = 16000


INDIC_LANGS = {"hi", "gu", "te", "ta"}
VALID_LANGS = {"en", "hi", "te", "ta", "gu"}


FORCE_LANG = (os.getenv("STT_FORCE_LANG") or "").strip() or None



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
load_piper(os.path.join(BASE_DIR, "te_IN-padmavathi-medium.onnx"), "te")
load_piper(os.path.join(BASE_DIR, "gu_epoch229.onnx"), "gu")

DEFAULT_TTS = PIPER_MODELS.get("en")


class TTSRequest(BaseModel):
    text: str
    language: str = "en"


@app.post("/tts")
async def tts(req: TTSRequest):
    if not req.text.strip():
        raise HTTPException(400, "Empty text")

    lang = (req.language or "en").strip().lower()[:2]
    voice = PIPER_MODELS.get(lang) or PIPER_MODELS.get("hi") or DEFAULT_TTS
    if voice is None:
        raise HTTPException(500, "No TTS voice loaded")

    def stream():
        for chunk in voice.synthesize(req.text):
            yield chunk.audio_int16_bytes

    return StreamingResponse(stream(), media_type="audio/pcm")




def _resample_to_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr == SAMPLE_RATE:
        return audio
    try:
        from scipy.signal import resample_poly
        g = np.gcd(sr, SAMPLE_RATE)
        up = SAMPLE_RATE // g
        down = sr // g
        return resample_poly(audio, up, down).astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"Unsupported sample rate {sr}; install scipy for resampling. ({e})")


def transcribe_whisper(audio: np.ndarray, lang: str | None = None, detect: bool = False):
    start = time.time()
    segments, info = whisper.transcribe(
        audio,
        language=lang if not detect else None,
        beam_size=1,
        temperature=0.0
    )
    segs = list(segments)
    text = "".join([s.text for s in segs]).strip()
    took = time.time() - start

    detected_lang = (info.language or lang or "en").strip().lower()[:2]
    prob = float(getattr(info, "language_probability", 0.0) or 0.0)

    avg_logprob = None
    try:
        if segs:
            avg_logprob = float(sum(getattr(s, "avg_logprob", 0.0) for s in segs) / len(segs))
    except Exception:
        avg_logprob = None

    return {
        "text": text,
        "language": detected_lang,
        "confidence": prob,
        "avg_logprob": avg_logprob,
        "time": took,
        "engine": "whisper",
    }


def transcribe_indic(audio, lang):
    if indic_model is None:
        logger.warning("⚠️ IndicConformer unavailable → fallback to Whisper")
        return transcribe_whisper(audio, lang=lang, detect=False)

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
        return transcribe_whisper(audio, lang=lang, detect=False)




@app.post("/transcribe")
async def transcribe(
    file: UploadFile,
    hint_language: str | None = Form(default=None),
):
    try:
        overall_start = time.time()
        audio_bytes = await file.read()
        audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")


        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        audio = _resample_to_16k(audio, sr)
        duration_sec = float(len(audio) / SAMPLE_RATE) if len(audio) else 0.0

        hint = (hint_language or "").strip().lower()[:2] or None


        if FORCE_LANG:
            lang = FORCE_LANG.strip().lower()[:2]

            if lang in INDIC_LANGS:
                logger.info(f"🇮🇳 [FORCE] Indic → {lang}")
                detect_time = 0.0
                result = transcribe_indic(audio, lang)
                confidence = 0.0
            else:
                logger.info(f"🇬🇧 [FORCE] Whisper → {lang}")
                out = transcribe_whisper(audio, lang=lang, detect=False)
                detect_time = 0.0
                result = {"text": out["text"], "language": lang, "engine": "whisper"}
                confidence = out["confidence"]

            pipeline_time = time.time() - overall_start
            return {
                "text": (result.get("text") or "").strip(),
                "language": lang,
                "engine": result.get("engine", "whisper"),
                "detected_language": lang,
                "confidence": round(float(confidence), 3),
                "detect_time": round(float(detect_time), 3),
                "pipeline_time": round(float(pipeline_time), 3),
                "duration": round(float(duration_sec), 3),
            }

        out = transcribe_whisper(audio, lang=None, detect=True)
        detect_time = out["time"]
        detected = (out["language"] or "en").strip().lower()[:2]

        conf = float(out.get("confidence", 0.0) or 0.0)
        if hint and hint in VALID_LANGS and conf < 0.35:
            logger.info(f"🧷 [HINT] Very low detect confidence ({conf:.2f}) → using hint={hint}")
            detected = hint

        if detected == "hi" and 0.25 <= conf <= 0.90:
            try:
                hi_out = transcribe_whisper(audio, lang="hi", detect=False)
                gu_out = transcribe_whisper(audio, lang="gu", detect=False)
                hi_lp = hi_out.get("avg_logprob")
                gu_lp = gu_out.get("avg_logprob")
                if hi_lp is not None and gu_lp is not None:
                    logger.info(f"🔎 [HIvsGU] hi_lp={hi_lp:.3f} gu_lp={gu_lp:.3f}")
                    if gu_lp > hi_lp + 0.05:
                        detected = "gu"
                        out = gu_out
                        logger.info("🔎 [HIvsGU] Switching detected language → gu")
            except Exception as e:
                logger.warning(f"🔎 [HIvsGU] Disambiguation failed: {e}")

        if detected == "te" and conf <= 0.80 and duration_sec <= 6.5:
            try:
                te_out = transcribe_whisper(audio, lang="te", detect=False)
                en_out = transcribe_whisper(audio, lang="en", detect=False)
                te_lp = te_out.get("avg_logprob")
                en_lp = en_out.get("avg_logprob")
                if te_lp is not None and en_lp is not None:
                    logger.info(f"🔎 [TEvsEN] te_lp={te_lp:.3f} en_lp={en_lp:.3f}")
                    if en_lp > te_lp + 0.10:
                        detected = "en"
                        out = en_out
                        logger.info("🔎 [TEvsEN] Switching detected language → en")
            except Exception as e:
                logger.warning(f"🔎 [TEvsEN] Disambiguation failed: {e}")

        if detected not in VALID_LANGS:
            detected = "hi"

        if detected in INDIC_LANGS:
            logger.info(f"🇮🇳 [ROUTE] Indic → {detected}")
            result = transcribe_indic(audio, detected)
            text = (result.get("text") or "").strip()
            engine = result.get("engine", "indic-conformer")
        else:
            logger.info(f"🇬🇧 [ROUTE] Whisper → {detected}")
            text = (out.get("text") or "").strip()
            engine = "whisper"

        pipeline_time = time.time() - overall_start

        logger.info(
            f"📊 Done | Lang={detected} | Engine={engine} | detect={detect_time:.2f}s total={pipeline_time:.2f}s"
        )

        return {
            "text": text,
            "language": detected,
            "engine": engine,
            "detected_language": detected,
            "confidence": round(float(out.get("confidence", 0.0) or 0.0), 3),
            "detect_time": round(float(detect_time), 3),
            "pipeline_time": round(float(pipeline_time), 3),
            "duration": round(float(duration_sec), 3),
        }

    except Exception as e:
        logger.error(f"❌ Transcription error: {e}")
        raise HTTPException(500, str(e))




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




if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        access_log=False
    )
