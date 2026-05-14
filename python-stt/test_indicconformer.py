import torch
import torchaudio
from transformers import AutoModel
import time
import sys
import numpy as np
import sounddevice as sd
from pathlib import Path

# Constants for recording
SAMPLE_RATE = 16000
CHANNELS = 1

def record_audio_live():
    """Records audio from the microphone until the user stops it."""
    print("\n🎤 Recording... Press Ctrl+C to stop recording and transcribe.")
    audio_data = []
    try:
        def callback(indata, frames, time, status):
            if status:
                print(status, file=sys.stderr)
            audio_data.append(indata.copy())

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=callback):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n🛑 Recording stopped.")
    
    if not audio_data:
        return None
        
    # Concatenate all chunks and convert to torch tensor
    audio_np = np.concatenate(audio_data, axis=0)
    audio_tensor = torch.from_numpy(audio_np).float().T # Shape [1, samples]
    return audio_tensor

def run_transcription(model, audio_tensor, language="hi", decoding="rnnt"):
    """Runs inference on a given audio tensor."""
    try:
        print(f"⏳ Transcribing ({language})...")
        start_inf = time.time()
        
        # model(wav, language) 
        with torch.no_grad():
            transcription = model(audio_tensor, language)
        
        elapsed = time.time() - start_inf
        print("\n" + "="*50)
        print(f"✨ TRANSCRIPTION RESULT")
        print("="*50)
        print(f"📝 {transcription}")
        print(f"⏱️  Time: {elapsed:.2f}s")
        print("="*50 + "\n")
        return transcription
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        return None

def main():
    # Parse arguments
    # Usage: python test_indicconformer.py [file_or_lang] [lang_if_file]
    if len(sys.argv) > 1:
        arg1 = sys.argv[1]
    else:
        arg1 = "hi"
    
    is_file = Path(arg1).exists()
    
    if is_file:
        language = sys.argv[2] if len(sys.argv) > 2 else "hi"
    else:
        language = arg1
    
    print(f"🚀 Loading IndicConformer-600M (HF/ONNX path)...")
    start_load = time.time()
    try:
        model = AutoModel.from_pretrained(
            "ai4bharat/indic-conformer-600m-multilingual", 
            trust_remote_code=True
        )
        print(f"✅ Model loaded in {time.time() - start_load:.2f}s")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    if is_file:
        print(f"🎙️ Processing file: {arg1}")
        try:
            wav, sr = torchaudio.load(arg1)
            if sr != SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
                wav = resampler(wav)
            if wav.shape[0] > 1:
                wav = torch.mean(wav, dim=0, keepdim=True)
            run_transcription(model, wav, language)
        except Exception as e:
            print(f"❌ Error loading audio file: {e}")
    else:
        print(f"🌟 Entering Live Mode (Language: {language})")
        print("Instructions: Speak into the mic. Press Ctrl+C to finish speaking and see result.")
        while True:
            try:
                audio_tensor = record_audio_live()
                if audio_tensor is not None:
                    run_transcription(model, audio_tensor, language)
                
                print("Options: [Enter] Speak again | [q] Quit")
                choice = input("> ").lower()
                if choice == 'q':
                    break
            except KeyboardInterrupt:
                print("\nExiting live mode.")
                break
            except EOFError:
                break

if __name__ == "__main__":
    main()