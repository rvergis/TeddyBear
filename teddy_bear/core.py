import cv2
import asyncio
import base64
import json
from pathlib import Path
from langchain_xai import ChatXAI
from langchain_core.messages import HumanMessage
import pyttsx3
from dotenv import load_dotenv
import os

load_dotenv()

XAI_API_KEY = os.getenv("X_API_KEY_TEDDYBEAR")
if not XAI_API_KEY:
    raise ValueError("❌ X_API_KEY_TEDDYBEAR not set!")

llm = ChatXAI(model="grok-4-1-fast-reasoning", temperature=0.6, api_key=XAI_API_KEY)

MEMORY_FILE = Path("teddy_memory.json")
memory = {"known_people": {}}
if MEMORY_FILE.exists():
    try:
        memory = json.loads(MEMORY_FILE.read_text())
    except:
        pass

def save_memory():
    MEMORY_FILE.write_text(json.dumps(memory, indent=2))

async def main():
    print("🧸 Teddy Bear - Person Identity Mode")

    cap = cv2.VideoCapture(0, 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    for _ in range(6):  # warmup
        cap.read()

    while True:
        ret, frame = cap.read()
        if not ret:
            await asyncio.sleep(5)
            continue

        small = cv2.resize(frame, (480, 360))
        _, buffer = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 70])
        base64_img = base64.b64encode(buffer).decode('utf-8')

        # Better prompt for identity
        msg = HumanMessage(content=[
            {"type": "text", "text": "Look at this person. Describe their face, hair, glasses, clothing, and any distinctive features in 1-2 short sentences. Be specific."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
        ])

        try:
            response = llm.invoke([msg])
            desc = response.content.strip()
            print(f"Vision: {desc}")

            # Try to match with known people
            known = False
            for name, stored_desc in memory["known_people"].items():
                if name.lower() in desc.lower() or any(word in desc.lower() for word in stored_desc.lower().split() if len(word) > 4):
                    speak(f"Hi {name}!")
                    print(f"Teddy: Hi {name}!")
                    known = True
                    break

            if not known:
                speak("Who are you?")
                print("Teddy: Who are you?")
                name_response = input("You: ")
                if name_response.strip():
                    name = name_response.title()
                    memory["known_people"][name] = desc
                    save_memory()
                    speak(f"Hi {name}! Nice to meet you!")
                    print(f"Teddy: Hi {name}! Nice to meet you!")

        except Exception as e:
            print(f"Error: {e}")

        await asyncio.sleep(8)

def speak(text: str):
    engine = pyttsx3.init()
    engine.setProperty('rate', 155)
    engine.say(text)
    engine.runAndWait()

if __name__ == "__main__":
    asyncio.run(main())
