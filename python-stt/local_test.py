#!/usr/bin/env python3
import argparse
import base64
import json
import time
import webrtcvad
import sounddevice as sd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import websocket
import threading

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent

SAMPLE_RATE = 16000
CHANNELS = 1
VAD_MODE = 3                    # 3 = most aggressive (good for noisy environments)
SILENCE_THRESHOLD = 1.5         # Seconds of silence = end of speech
CHUNK_DURATION = 0.03           # 30ms chunks

vad = webrtcvad.Vad(VAD_MODE)

def load_environment():
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)

def is_speech(frame: bytes) -> bool:
    try:
        return vad.is_speech(frame, SAMPLE_RATE)
    except:
        return False

def record_with_vad():
    print("\n🎤 Listening... Speak now (natural conversation)")

    buffer = []
    silence_frames = 0
    recording = False

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype='int16',
        blocksize=int(SAMPLE_RATE * CHUNK_DURATION)
    )
    stream.start()

    try:
        while True:
            audio_chunk, _ = stream.read(stream.blocksize)
            frame = audio_chunk.tobytes()

            if is_speech(frame):
                if not recording:
                    print("🗣️  Speech started...")
                    recording = True
                silence_frames = 0
                buffer.extend(audio_chunk)
            else:
                if recording:
                    silence_frames += 1
                    buffer.extend(audio_chunk)   # keep some trailing silence

                    if silence_frames > int(SILENCE_THRESHOLD / CHUNK_DURATION):
                        print("⏹️  Speech ended")
                        break
    finally:
        stream.stop()
        stream.close()

    audio_bytes = np.array(buffer, dtype=np.int16).tobytes()
    duration = len(buffer) / SAMPLE_RATE
    print(f"✅ Recorded {duration:.1f} seconds")
    
    return audio_bytes

def play_audio(audio_bytes: bytes):
    print("▶️ Playing AI response...")
    
    try:
        audio = np.frombuffer(audio_bytes, dtype="int16")
        
        # Play with lower latency settings
        sd.play(audio, samplerate=SAMPLE_RATE, blocking=True, device=None)
        
        # Optional: Add a very small delay to prevent glitches
        # time.sleep(0.05)
        
        print("✅ Response finished. Listening again...\n")
        
    except Exception as e:
        print(f"Playback error: {e}")
        # Fallback
        import subprocess
        with open("temp_playback.wav", "wb") as f:
            f.write(audio_bytes)
        subprocess.run(["aplay", "temp_playback.wav"])   # For Linux


def on_message(ws, message):
    try:
        data = json.loads(message)
        if data["type"] == "tts":
            audio_bytes = base64.b64decode(data["audio"])
            play_audio(audio_bytes)
    except Exception as e:
        print(f"Error in on_message: {e}")

def on_message(ws, message):
    try:
        data = json.loads(message)
        if data["type"] == "tts":
            audio_bytes = base64.b64decode(data["audio"])
            play_audio(audio_bytes)
    except Exception as e:
        print(f"Error playing audio: {e}")

def on_error(ws, error):
    print(f"❌ WebSocket Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("🔴 WebSocket connection closed")

def on_open(ws):
    print("✅ Connected to Voice Agent! You can start speaking now.")

def main():
    load_environment()

    ws = websocket.WebSocketApp(
        "ws://localhost:3001",
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
        on_open=on_open,
    )

    # Continuous conversation loop
    def conversation_loop():
        while True:
            try:
                audio_bytes = record_with_vad()
                
                # Ignore very short sounds
                if len(audio_bytes) < SAMPLE_RATE * 0.6:  
                    print("⚠️  Too short, ignoring...")
                    continue

                ws.send(json.dumps({
                    "type": "audio",
                    "audio": base64.b64encode(audio_bytes).decode("utf-8")
                }))
            except Exception as e:
                print(f"Error in conversation loop: {e}")
                time.sleep(1)

    # Start WebSocket in main thread
    threading.Thread(target=conversation_loop, daemon=True).start()
    ws.run_forever()

if __name__ == "__main__":
    main()