# AI Voice Agent

Voice agent server that handles Twilio voice calls, integrates with ElevenLabs API, and uses faster-whisper for speech-to-text conversion.

## Tech Stack

- **Backend**: Node.js + Express + TypeScript
- **Voice Integration**: Twilio Voice API
- **AI**: ElevenLabs (text-to-speech & conversational AI), Groq (LLM)
- **STT**: Python + faster-whisper

## Prerequisites

- Node.js (v18+)
- Python (v3.9+)
- Twilio account with Voice capability
- ElevenLabs account with Agent ID
- Groq API key

## Installation

### Node.js Dependencies

```bash
npm install
```

### Python Dependencies

```bash
cd python-stt
pip install -r requirements.txt
```

For the new local terminal test flow, install the helper packages here:

```bash
pip install -r requirements.txt
```

## Environment Setup

Create a `.env` file in the root directory:

```env
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_PHONE_NUMBER=
USER_PHONE_NUMBER=

ELEVENLABS_AGENT_ID=
ELEVENLABS_API_KEY=

GROQ_API_KEY=

NGROK_URL=https://your-ngrok-url.ngrok-free.app
```

## Running the Project

### Start Node Server

```bash
npm run dev
# or
npm start
```

Server runs on port 3000.

### Start Python STT Service

```bash
python python-stt/main.py
```

### Local Terminal Voice Agent Test

For a live, end-to-end voice agent simulation in the terminal:

1. Start the Python STT service:

   ```bash
   cd python-stt
   python main.py
   ```

2. Start the Node.js local agent WebSocket server:

   ```bash
   npx tsx local_agent.js
   ```

3. In another terminal, run the client:
   ```bash
   cd python-stt
   python local_test.py --duration 5
   ```

This will record your speech, send it to the agent via WebSocket, transcribe with local Whisper, query Groq LLM, generate TTS with Piper, and play the response audio.

The script requires `GROQ_API_KEY` to be set in the project root `.env` or your shell environment.

## How It Works

1. Twilio receives inbound calls and routes audio to the Node server
2. Audio stream is processed by faster-whisper (Python) for speech-to-text
3. Transcribed text is sent to Groq LLM for processing
4. Response is converted to speech by ElevenLabs
5. Audio is streamed back to the caller via Twilio

---

# ▶️ Run the Project

Start the development server:

```bash
npm run dev
```

Project will run on:

```txt
http://localhost:3000
```

---

# 📞 Making Calls

This project currently supports **Outbound Calls Only**.

Open the following route in your browser:

```txt
http://localhost:3000/make-call
```

This will:

1. Trigger a Twilio outbound call
2. Call the number defined in `USER_PHONE_NUMBER`
3. Connect the call to the ElevenLabs AI agent
4. Start the AI conversation

---

# ⚠️ Twilio Free Trial Note

If your Twilio account is in **Free Trial Mode**, Twilio may automatically disconnect the call after pickup.

To prevent this:

1. Answer the incoming call
2. Open the phone dialer keypad
3. Press any number/key once after the call connects

This helps keep the call active during testing.
