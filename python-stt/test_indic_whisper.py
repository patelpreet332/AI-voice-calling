#!/usr/bin/env python3

import argparse
import sys
import time
import numpy as np
from pathlib import Path

LANGUAGE_NAMES = {
    "hi": "Hindi",
    "te": "Telugu",
    "ta": "Tamil",
    "kn": "Kannada",
    "ml": "Malayalam",
    "gu": "Gujarati",
    "mr": "Marathi",
    "pa": "Punjabi",
    "en": "English",
}


def detect_indic_script(text: str) -> str:
    """Detect Unicode script used in transcript."""
    ranges = {
        "Devanagari": (0x0900, 0x097F),
        "Gurmukhi": (0x0A00, 0x0A7F),
        "Gujarati": (0x0A80, 0x0AFF),
        "Tamil": (0x0B80, 0x0BFF),
        "Telugu": (0x0C00, 0x0C7F),
        "Kannada": (0x0C80, 0x0CFF),
    }

    counts = {lang: 0 for lang in ranges}

    for char in text:
        code = ord(char)

        for lang, (start, end) in ranges.items():
            if start <= code <= end:
                counts[lang] += 1
                break

    detected = max(counts, key=counts.get)

    return detected if counts[detected] > 0 else "Latin / Unknown"


def record_live_audio():
    """Records microphone audio using WebRTC VAD."""

    try:
        import sounddevice as sd
        import webrtcvad
    except ImportError:
        print("\n❌ Missing dependencies.")
        print("Install with:")
        print("pip install sounddevice webrtcvad")
        sys.exit(1)

    SAMPLE_RATE = 16000
    CHUNK_DURATION = 0.03
    SILENCE_THRESHOLD = 1.5

    vad = webrtcvad.Vad(2)

    print("\n🎤 Listening...")

    buffer = []
    silence_frames = 0
    recording = False

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=int(SAMPLE_RATE * CHUNK_DURATION),
    )

    stream.start()

    try:
        while True:
            audio_chunk, _ = stream.read(stream.blocksize)

            frame = audio_chunk.tobytes()

            try:
                is_speech = vad.is_speech(frame, SAMPLE_RATE)
            except Exception:
                is_speech = False

            if is_speech:
                if not recording:
                    print("🗣️ Speech detected...")
                    recording = True

                silence_frames = 0
                buffer.extend(audio_chunk)

            else:
                if recording:
                    silence_frames += 1
                    buffer.extend(audio_chunk)

                    if silence_frames > int(
                        SILENCE_THRESHOLD / CHUNK_DURATION
                    ):
                        print("⏹️ Processing...")
                        break

    finally:
        stream.stop()
        stream.close()

    audio_int16 = np.frombuffer(
        np.array(buffer, dtype=np.int16).tobytes(),
        dtype=np.int16,
    )

    audio_float32 = audio_int16.astype(np.float32) / 32768.0

    duration = len(audio_float32) / SAMPLE_RATE

    print(f"✅ Captured {duration:.1f}s audio")

    return audio_float32


def print_transcription_result(info, text, elapsed):
    detected_lang = info.language or "unknown"

    lang_name = LANGUAGE_NAMES.get(
        detected_lang,
        detected_lang.upper()
    )

    confidence = float(info.language_probability)

    detected_script = detect_indic_script(text)

    print(f"\n🗣️ Language   : {lang_name}")
    print(f"🎯 Confidence : {confidence:.1%}")
    print(f"✍️ Script     : {detected_script}")
    print(f"⏱️ Time       : {elapsed:.2f}s")

    print("\n📝 Transcript:")
    print(text if text else "[No speech detected]")

    print("\n" + "-" * 65)


def main():
    parser = argparse.ArgumentParser(
        description="Fast multilingual Whisper transcription"
    )

    parser.add_argument(
        "--model",
        default="small",
        help="Whisper model name"
    )

    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable live microphone mode"
    )

    args = parser.parse_args()

    model_name = args.model

    print("\n" + "=" * 65)
    print(f"🚀 FASTER-WHISPER ({model_name.upper()}) 🚀".center(65))
    print("=" * 65)

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("\n❌ faster-whisper not installed.")
        print("Install with:")
        print("pip install faster-whisper")
        sys.exit(1)

    print("\n🧠 Loading model...")

    start_load = time.time()

    try:
        model = WhisperModel(
            model_name,
            device="cpu",
            compute_type="int8",
        )

        print(
            f"✅ Loaded in {time.time() - start_load:.2f}s"
        )

    except Exception as e:
        print(f"\n❌ Failed loading model: {e}")
        sys.exit(1)

    # Optimized multilingual settings
    transcribe_params = dict(
        language=None,
        task="transcribe",
        beam_size=1,
        best_of=1,
        temperature=0.0,
        condition_on_previous_text=True,
        repetition_penalty=1.05,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=500,
        ),
        no_speech_threshold=0.5,
        initial_prompt=None,
    )

    # =========================================================
    # LIVE MICROPHONE MODE
    # =========================================================

    if args.live:
        print("\n" + "=" * 65)
        print("🎙️ LIVE MICROPHONE MODE 🎙️".center(65))
        print("Press Ctrl+C to exit")
        print("=" * 65)

        while True:
            try:
                audio_input = record_live_audio()

                if len(audio_input) < 16000 * 0.5:
                    print("⚠️ Audio too short")
                    continue

                start_transcribe = time.time()

                segments, info = model.transcribe(
                    audio_input,
                    **transcribe_params
                )

                text = " ".join(
                    seg.text.strip()
                    for seg in segments
                ).strip()

                elapsed = time.time() - start_transcribe

                print_transcription_result(
                    info,
                    text,
                    elapsed
                )

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break

            except Exception as e:
                print(f"\n❌ Error: {e}")

    # =========================================================
    # INTERACTIVE FILE MODE
    # =========================================================

    else:
        print("\n" + "=" * 65)
        print("📂 INTERACTIVE FILE MODE 📂".center(65))
        print("Type audio file path")
        print("Type 'exit' to quit")
        print("=" * 65)

        while True:
            try:
                audio_input = input(
                    "\n🎵 Audio file: "
                ).strip()

                if audio_input.lower() in [
                    "exit",
                    "quit"
                ]:
                    print("\n👋 Goodbye!")
                    break

                audio_path = Path(audio_input)

                if not audio_path.exists():
                    print("❌ File not found")
                    continue

                print(f"\n🎙️ Processing: {audio_path.name}")

                start_transcribe = time.time()

                segments, info = model.transcribe(
                    str(audio_path),
                    **transcribe_params
                )

                text = " ".join(
                    seg.text.strip()
                    for seg in segments
                ).strip()

                elapsed = time.time() - start_transcribe

                print_transcription_result(
                    info,
                    text,
                    elapsed
                )

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break

            except Exception as e:
                print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()