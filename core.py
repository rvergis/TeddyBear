import cv2
import time
import asyncio
import base64
from langchain_xai import ChatXAI
from langchain_core.messages import HumanMessage
import speech_recognition as sr
import pyttsx3
import os
from dotenv import load_dotenv

load_dotenv()

# ===================== CONFIG =====================
XAI_API_KEY = os.getenv("X_API_KEY_TEDDYBEAR")

if not XAI_API_KEY:
    raise ValueError("❌ X_API_KEY_TEDDYBEAR environment variable not set!")

llm = ChatXAI(
    model="grok-4-1-fast-reasoning",
    temperature=0.75,
    api_key=XAI_API_KEY
)

SYSTEM_PROMPT = """You are Teddy, a cute, playful, friendly 3-year-old teddy bear.
Speak in very short, warm, excited, simple sentences.
Be curious, affectionate, and fun."""

# ===================== VISION =====================
def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def see():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return "I can't see anything right now."
    
    cv2.imwrite("teddy_view.jpg", frame)
    
    # Proper image encoding for Grok
    base64_image = encode_image("teddy_view.jpg")
    
    message = HumanMessage(
        content=[
            {"type": "text", "text": "Describe exactly what you see right now in 1-2 short sentences."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]
    )
    
    response = llm.invoke([message])
    return response.content.strip()

# ===================== SPEECH =====================
recognizer = sr.Recognizer()

def listen():
    try:
        with sr.Microphone() as source:
            audio = recognizer.listen(source, timeout=4, phrase_time_limit=7)
        text = recognizer.recognize_whisper(audio, model="tiny").strip()
        return text
    except:
        return None

def speak(text: str):
    engine = pyttsx3.init()
    engine.setProperty('rate', 155)
    engine.say(text)
    engine.runAndWait()

# ===================== MAIN LOOP =====================
async def main():
    print("🧸 Teddy Bear is awake and watching... (Grok 4.1 Fast Reasoning)")

    last_proactive = time.time()

    while True:
        spoken = listen()
        
        if spoken:
            print(f"You: {spoken}")
            lower = spoken.lower()
            
            if any(word in lower for word in ["see", "look", "what is", "what's this", "this"]):
                what_i_see = see()
                prompt = f"{SYSTEM_PROMPT}\n\nYou see: {what_i_see}\nUser: {spoken}\nTeddy:"
            else:
                prompt = f"{SYSTEM_PROMPT}\n\nUser: {spoken}\nTeddy:"
            
            reply = llm.invoke(prompt).content.strip()
            print(f"Teddy: {reply}")
            speak(reply)
            last_proactive = time.time()
            continue

        # Proactive watching
        if time.time() - last_proactive > 14:
            what_i_see = see()
            prompt = f"{SYSTEM_PROMPT}\n\nYou see: {what_i_see}\nMake a short playful comment."
            comment = llm.invoke(prompt).content.strip()
            
            print(f"Teddy sees: {what_i_see}")
            print(f"Teddy says: {comment}")
            speak(comment)
            last_proactive = time.time()

        await asyncio.sleep(0.8)

if __name__ == "__main__":
    asyncio.run(main())