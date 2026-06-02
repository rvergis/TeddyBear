import sounddevice as sd
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
import time

print("🎤 Audio Test Starting...")

# Load whisper (small model)
model = WhisperModel("small", device="cpu", compute_type="float32")

print("Recording for 6 seconds... Speak now!")

# Record audio
audio = sd.rec(int(6 * 16000), samplerate=16000, channels=1, dtype=np.float32)
sd.wait()

print("Recording finished. Processing...")

# Save temp file
sf.write("test_audio.wav", audio, 16000)

# Transcribe
segments, _ = model.transcribe("test_audio.wav", beam_size=5)
text = " ".join(segment.text for segment in segments).strip()

print(f"\n✅ You said: {text}")

# Cleanup
import os
if os.path.exists("test_audio.wav"):
    os.remove("test_audio.wav")
