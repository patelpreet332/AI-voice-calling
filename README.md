# AI Voice Agent (Next.js + Twilio + ElevenLabs)

AI Voice Calling project using:

- Next.js
- Twilio Voice API
- ElevenLabs Conversational AI
- ngrok

This project currently supports **Outbound Calling Only**.

---

# 🚀 Getting Started

## 1. Clone Repository

```bash
git clone <your-repository-url>
cd <your-project-folder>
```

---

## 2. Install Dependencies

```bash
npm install
```

---

# ⚙️ Environment Setup

Create a `.env` file in the root directory.

Add the following environment variables:

```env
ELEVENLABS_AGENT_ID=

ELEVENLABS_API_KEY=

TWILIO_ACCOUNT_SID=

TWILIO_AUTH_TOKEN=

TWILIO_PHONE_NUMBER=

USER_PHONE_NUMBER=

NGROK_URL=
```

---

# 🔑 Required Accounts

Before running the project, make sure you have:

## Twilio

- Create a Twilio account
- Buy or use a Twilio phone number
- Enable Voice capability on the number

Required values from Twilio:

- `TWILIO_ACCOUNT_SID`
- `TWILIO_AUTH_TOKEN`
- `TWILIO_PHONE_NUMBER`

---

## ElevenLabs

- Create an ElevenLabs account
- Create a Conversational AI Agent

Required values from ElevenLabs:

- `ELEVENLABS_AGENT_ID`
- `ELEVENLABS_API_KEY`

---

## ngrok

Install and start ngrok:

```bash
ngrok http 3000
```

Copy the generated HTTPS URL and add it to:

```env
NGROK_URL=https://your-ngrok-url.ngrok-free.app
```

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
