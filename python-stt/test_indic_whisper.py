#!/usr/bin/env python3
import argparse
import sys
import time
import numpy as np
from pathlib import Path

def detect_indic_language(text: str) -> str:
    """Detects the Indian language/script based on Unicode character ranges."""
    ranges = {
        "Hindi / Devanagari": (0x0900, 0x097F),
        "Bengali": (0x0980, 0x09FF),
        "Gurmukhi / Punjabi": (0x0A00, 0x0A7F),
        "Gujarati": (0x0A80, 0x0AFF),
        "Odia": (0x0B00, 0x0B7F),
        "Tamil": (0x0B80, 0x0BFF),
        "Telugu": (0x0C00, 0x0C7F),
        "Kannada": (0x0C80, 0x0CFF),
        "Malayalam": (0x0D00, 0x0D7F),
    }
    
    counts = {lang: 0 for lang in ranges}
    for char in text:
        code = ord(char)
        for lang, (start, end) in ranges.items():
            if start <= code <= end:
                counts[lang] += 1
                break
                
    detected = max(counts, key=counts.get)
    return detected if counts[detected] > 0 else "English / Latin or Unknown"

def record_live_audio():
    """Records voice from the local microphone using aggressive VAD."""
    try:
        import sounddevice as sd
        import webrtcvad
    except ImportError:
        print("\n❌ Error: 'sounddevice' or 'webrtcvad' is missing for live voice recording.")
        print("💡 Please install them by running:\n   pip install sounddevice webrtcvad")
        sys.exit(1)

    SAMPLE_RATE = 16000
    CHUNK_DURATION = 0.03  # 30ms chunks
    SILENCE_THRESHOLD = 1.5  # 1.5 seconds of silence stops recording
    vad = webrtcvad.Vad(3)   # Aggressive filtering

    print("\n🎤 Listening... Speak naturally into your microphone")

    buffer = []
    silence_frames = 0
    recording = False

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='int16',
        blocksize=int(SAMPLE_RATE * CHUNK_DURATION)
    )
    stream.start()

    try:
        while True:
            audio_chunk, _ = stream.read(stream.blocksize)
            frame = audio_chunk.tobytes()

            # Safely check if chunk contains speech
            try:
                is_speech = vad.is_speech(frame, SAMPLE_RATE)
            except:
                is_speech = False

            if is_speech:
                if not recording:
                    print("🗣️  Speech started...")
                    recording = True
                silence_frames = 0
                buffer.extend(audio_chunk)
            else:
                if recording:
                    silence_frames += 1
                    buffer.extend(audio_chunk)  # Include short trailing pauses

                    if silence_frames > int(SILENCE_THRESHOLD / CHUNK_DURATION):
                        print("⏹️  Speech ended. Processing audio...")
                        break
    finally:
        stream.stop()
        stream.close()

    # Convert recorded int16 PCM buffer to flat float32 array normalized to [-1.0, 1.0]
    audio_int16 = np.frombuffer(np.array(buffer, dtype=np.int16).tobytes(), dtype=np.int16)
    audio_float32 = audio_int16.astype(np.float32) / 32768.0
    duration = len(audio_float32) / SAMPLE_RATE
    print(f"✅ Captured {duration:.1f} seconds of audio.")
    
    return audio_float32

def main():
    parser = argparse.ArgumentParser(description="Test faster-whisper models via file transcription or live system microphone loops.")
    parser.add_argument("--model", default="medium", help="Model size identifier (default: medium)")
    parser.add_argument("audio", nargs="*", help="Paths to audio files to transcribe. If omitted, launches interactive live voice recording mode.")
    args = parser.parse_args()

    model_name = args.model
    audio_files = args.audio
    is_live_mode = len(audio_files) == 0

    print("\n" + "="*65)
    print(f"🚀 TESTING FASTER-WHISPER '{model_name.upper()}' MODEL 🚀".center(65))
    print("="*65)
    print(f"📦 Model Identifier : {model_name}")
    print(f"⚙️ Execution Mode   : {'Live System Microphone' if is_live_mode else f'File Processing ({len(audio_files)} file/s)'}")
    print("-" * 65)

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("\n❌ Error: 'faster-whisper' is not installed.")
        sys.exit(1)

    print("\n🧠 Loading model into CPU memory...")
    start_load = time.time()
    try:
        model = WhisperModel(
            model_name,
            device="cpu",
            compute_type="int8"
        )
        print(f"✅ Model loaded successfully in {time.time() - start_load:.2f} seconds!")
    except Exception as e:
        print(f"\n❌ Failed to load Whisper model '{model_name}': {e}")
        sys.exit(1)

    # Common optimized transcription configuration parameters
    transcribe_params = dict(
        language=None,                   # Auto detect
        beam_size=5,                     # Higher beam search improves token alignment
        repetition_penalty=1.2,          # Strongly penalizes emitting the same repeating character
        condition_on_previous_text=False,# Prevents past hallucinations from polluting future chunks
        temperature=[0.0, 0.2, 0.4],     # Automatically retries if compression/looping triggers
        vad_filter=True,                 # Strips non-speech background noise/hiss to prevent phantom tokens
        vad_parameters=dict(
            min_silence_duration_ms=400,
            threshold=0.5
        ),
        no_speech_threshold=0.6,         # Ignores pure noise segments
        initial_prompt="નમસ્તે नमस्ते வணக்கம் నమస్కారం Hello" # Multi-script anchor prompt steering
    )

    if is_live_mode:
        print("\n" + "="*65)
        print("🎙️ ENTERING CONTINUOUS LIVE VOICE MODE 🎙️".center(65))
        print("💡 Press Ctrl+C anytime to exit the loop.")
        print("="*65)

        while True:
            try:
                audio_input = record_live_audio()
                
                # Skip if sound was too short
                if len(audio_input) < 16000 * 0.5:
                    print("⚠️ Audio too short, ignoring...\n")
                    continue

                start_transcribe = time.time()
                segments, info = model.transcribe(audio_input, **transcribe_params)
                
                text = " ".join([seg.text.strip() for seg in segments]).strip()
                elapsed = time.time() - start_transcribe

                detected_lang = info.language or "unknown"
                confidence = float(info.language_probability)
                detected_script = detect_indic_language(text)

                print(f"\n🗣️  Detected Language : {detected_lang.upper()} (Confidence: {confidence:.1%})")
                print(f"✍️  Inferred Script   : {detected_script}")
                print(f"⏱️  Processing Time   : {elapsed:.2f} seconds")
                print(f"📝 Transcript:\n{text if text else '[(No recognizable speech detected)]'}\n")
                print("-" * 65)

            except KeyboardInterrupt:
                print("\n\n🛑 Exiting live voice mode. Goodbye!\n")
                break
            except Exception as e:
                print(f"\n❌ Error during live transcription: {e}\n")
                time.sleep(1)
    else:
        # File processing mode
        for audio_arg in audio_files:
            audio_path = Path(audio_arg)
            print(f"\n" + "-"*65)
            print(f"🎙️ Processing file: '{audio_path.name}'...")
            
            if not audio_path.exists():
                print(f"❌ Error: Audio file '{audio_path}' not found. Skipping.")
                continue

            start_transcribe = time.time()
            try:
                segments, info = model.transcribe(str(audio_path), **transcribe_params)

                text = " ".join([seg.text.strip() for seg in segments]).strip()
                elapsed = time.time() - start_transcribe

                detected_lang = info.language or "unknown"
                confidence = float(info.language_probability)
                detected_script = detect_indic_language(text)

                print(f"🗣️  Detected Language : {detected_lang.upper()} (Confidence: {confidence:.1%})")
                print(f"✍️  Inferred Script   : {detected_script}")
                print(f"⏱️  Processing Time   : {elapsed:.2f} seconds")
                print(f"📝 Transcript:\n{text}\n")

            except Exception as e:
                print(f"❌ Error transcribing '{audio_path.name}': {e}\n")

        print("="*65)
        print("✅ ALL FILES PROCESSED SUCCESSFULLY".center(65))
        print("="*65 + "\n")

if __name__ == "__main__":
    main()
