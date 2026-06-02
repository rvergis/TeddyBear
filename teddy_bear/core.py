import cv2
import time
import asyncio
import base64
import json
import os
from pathlib import Path
import sounddevice as sd
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from langchain_xai import ChatXAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

# ===================== CONFIG =====================
XAI_API_KEY = os.getenv("X_API_KEY_TEDDYBEAR")
if not XAI_API_KEY:
    raise ValueError("❌ X_API_KEY_TEDDYBEAR not set!")

llm = ChatXAI(model="grok-4-1-fast-reasoning", temperature=0.7, api_key=XAI_API_KEY)

whisper = WhisperModel("small", device="cpu", compute_type="float32")

MEMORY_FILE = Path("teddy_memory.json")

def load_memory():
    if MEMORY_FILE.exists():
        try:
            return json.loads(MEMORY_FILE.read_text())
        except:
            return {"known_people": {}}
    return {"known_people": {}}

def save_memory(memory):
    MEMORY_FILE.write_text(json.dumps(memory, indent=2))

memory = load_memory()

SYSTEM_PROMPT = """You are Teddy, a cute, playful, friendly 3-year-old teddy bear.
Speak in very short, warm, excited, simple sentences."""

# ===================== VISION =====================
def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def see():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    cv2.imwrite("teddy_view.jpg", frame)
    base64_img = encode_image("teddy_view.jpg")
    
    msg = HumanMessage(content=[
        {"type": "text", "text": "Is there a person in this image? If yes, describe them briefly."},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
    ])
    response = llm.invoke([msg])
    return response.content.strip()

# ===================== SPEECH =====================
def listen():
    try:
        print("🎤 Listening...")
        audio = sd.rec(int(6 * 16000), samplerate=16000, channels=1, dtype=np.float32)
        sd.wait()
        
        temp_path = "temp_audio.wav"
        sf.write(temp_path, audio, 16000)
        
        segments, _ = whisper.transcribe(temp_path, beam_size=5)
        text = " ".join(segment.text for segment in segments).strip()
        
        if text:
            print(f"You said: {text}")
        return text
    except Exception as e:
        print(f"Audio error: {e}")
        return None
    finally:
        if os.path.exists("temp_audio.wav"):
            os.remove("temp_audio.wav")

def speak(text: str):
    import pyttsx3
    engine = pyttsx3.init()
    engine.setProperty('rate', 155)
    engine.say(text)
    engine.runAndWait()

# ===================== MAIN =====================
async def main():
    print("🧸 Teddy Bear is awake and watching...")
    while True:
        description = see()
        if description and "person" in description.lower():
            print("👤 Person detected!")
            
            known = False
            for name, desc in memory["known_people"].items():
                if desc.lower() in description.lower():
                    speak(f"Hi {name}!")
                    known = True
                    break
            
            if not known:
                speak("Who are you?")
                name_response = listen()
                if name_response:
                    name = name_response.split("is")[-1].strip() if "is" in name_response else name_response.title()
                    memory["known_people"][name] = description
                    save_memory(memory)
                    speak(f"Hi {name}! Nice to meet you!")

        await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(main())
