# AI Voice Agent (Phone + Local)

Two ways to run this project:

1) 📞 **Phone calls (Twilio)**: Twilio → Node (`server.ts`) → Python STT/TTS (`python-stt/main.py`) → Twilio  
2) 🎙️ **Local mic test (no Twilio)**: `python-stt/local_test.py`

## Repo Layout (Quick)

- `server.ts` — Express server + Twilio webhook + WebSocket media stream
- `lib/` — local pipeline (VAD, STT client, LLM, audio utils)
- `python-stt/main.py` — FastAPI service providing `/transcribe` and `/tts`
- `python-stt/local_test.py` — local mic → STT → Groq → Piper → speaker
- `piper/` — Piper ONNX voices (ignored by git; download separately)

## Prerequisites

- Node.js 18+
- Python 3.10+ (3.11 recommended)
- Groq API key (both flows)
- Twilio Voice-capable account (phone-call flow)
- Microphone + speakers (local mic flow)

### System Packages (Local Mic Test)

`sounddevice` needs PortAudio installed:
- Ubuntu/Debian: `sudo apt-get install portaudio19-dev`
- macOS: `brew install portaudio`

## Quick Start: Phone Call Flow (Twilio)

### 1) Install Node deps

```bash
npm install
```

### 2) Create Python venv + install deps

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r python-stt/requirements.txt
```

### 3) Configure env

Copy `.env.example` → `.env` and fill:

```bash
cp .env.example .env
```

Required (`.env`):
- `GROQ_API_KEY`
- `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`
- `TWILIO_PHONE_NUMBER`, `USER_PHONE_NUMBER`
- `NGROK_URL` (public `https://...` for your local server)

Optional:
- `STT_URL` (default `http://localhost:8000`)
- `STT_FORCE_LANG` (example: `hi`, `en`, `te`)
- `DEFAULT_LANG` (hint language for first turns)
- `GREETING_TEXT` (send a greeting after connect)
- `VAD_THRESHOLD`, `VAD_SILENCE_MS`, `VAD_MIN_SPEECH_MS`, `VAD_MAX_SPEECH_MS`

### 4) Piper voices (not committed)

Put your Piper `.onnx` (and `.onnx.json`) files in `piper/`.

From repo root:

```bash
mkdir -p piper
```

English:

```bash
wget -O piper/en_US-lessac-medium.onnx https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
wget -O piper/en_US-lessac-medium.onnx.json https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

Hindi:

```bash
wget -O piper/hi_IN-priyamvada-medium.onnx https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/priyamvada/medium/hi_IN-priyamvada-medium.onnx
wget -O piper/hi_IN-priyamvada-medium.onnx.json https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/priyamvada/medium/hi_IN-priyamvada-medium.onnx.json
```

Telugu:

```bash
wget -O piper/te_IN-padma-medium.onnx https://huggingface.co/rhasspy/piper-voices/resolve/main/te/te_IN/padmavathi/medium/te_IN-padmavathi-medium.onnx
wget -O piper/te_IN-padma-medium.onnx.json https://huggingface.co/rhasspy/piper-voices/resolve/main/te/te_IN/padmavathi/medium/te_IN-padmavathi-medium.onnx.json
```

Gujarati:

- Request access and download manually from `https://huggingface.co/Arjun4707/piper-gujarati-male/tree/main`
- Place these files in `piper/`:
  - `gu_epoch229.onnx`
  - `gu_epoch229.onnx.json`

### 5) Start Python STT+TTS

```bash
cd python-stt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Sanity check:

```bash
curl http://localhost:8000/health
```

### 6) Start ngrok + Node server

Run ngrok in a separate terminal:

```bash
ngrok http 3000
```

Update `NGROK_URL` in `.env` with the HTTPS forwarding URL, then:

```bash
npm run dev
```

### 7) Make a call

```txt
http://localhost:3000/make-call
```

This triggers an outbound call from `TWILIO_PHONE_NUMBER` → `USER_PHONE_NUMBER` and connects Twilio Media Streams to `NGROK_URL/voice`.

### ⚠️ Twilio Free Trial Note

If your Twilio account is in **Free Trial Mode**, Twilio may automatically disconnect the call after pickup.

To prevent this:
1. Answer the incoming call
2. Open the phone dialer keypad
3. Press any number/key once after the call connects

This helps keep the call active during testing.

## Quick Start: Local Mic Flow

No Twilio. No Node server.

1) Ensure `.env` is set (from repo root):

```bash
cp .env.example .env
```

Required for local mic flow:
- `GROQ_API_KEY`

Also required:
- Piper voice models in `piper/` (follow the “Piper voices (not committed)” step above)

2) Activate the venv (same as above), then:

```bash
python3 python-stt/local_test.py
```

Optional: force a language (2-letter code) for testing:

```bash
python3 python-stt/local_test.py hi
```

## Hugging Face / IndicConformer Notes

The Python STT code loads `ai4bharat/indic-conformer-600m-multilingual` via `transformers`.
On some machines this download may prompt for Hugging Face auth.

If you get prompted for a token:
- Login once: `huggingface-cli login`
- Or export a token: `export HF_TOKEN="..."` (or `export HUGGINGFACE_HUB_TOKEN="..."`)

## Troubleshooting

- `No default input device` / microphone not working:
  check OS audio permissions and that PortAudio is installed.
- `IndicConformer failed` in logs:
  the app will auto-fallback to Whisper; set up Hugging Face auth if needed.
- Call connects but no audio / hang-up: verify `NGROK_URL` (HTTPS, current ngrok URL) and Twilio creds/numbers.

## Development Notes

- Node server listens on `PORT` (default `3000`) and uses `/voice` as Twilio webhook.
- Node → Python STT/TTS uses `STT_URL` (default `http://localhost:8000`).
- The Python service exposes:
  - `POST /transcribe` (`multipart/form-data`: `file`, optional `hint_language`)
  - `POST /tts` (JSON: `{ "text": "...", "language": "en" }`)
