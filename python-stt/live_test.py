import sounddevice as sd
from scipy.io.wavfile import write
import requests

SAMPLE_RATE = 16000
CHUNK_DURATION = 3  # seconds

print("🎤 Live STT Test Started (Ctrl+C to stop)")

while True:
    print("\n🎙️ Speak now...")

    audio = sd.rec(
        int(CHUNK_DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='int16'
    )
    sd.wait()

    write("chunk.wav", SAMPLE_RATE, audio)

    try:
        files = {"file": open("chunk.wav", "rb")}
        res = requests.post("http://localhost:8000/transcribe", files=files)

        print("🧠 Result:", res.json())

    except Exception as e:
        print("❌ Error:", e)