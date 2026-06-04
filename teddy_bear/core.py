import asyncio
import subprocess
import base64
import sounddevice as sd
import numpy as np
import soundfile as sf
from mlx_whisper import transcribe
from langchain_xai import ChatXAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os
import json
from pathlib import Path

load_dotenv()

XAI_API_KEY = os.getenv("X_API_KEY_TEDDYBEAR")
if not XAI_API_KEY:
    raise ValueError("❌ X_API_KEY_TEDDYBEAR not set!")

llm = ChatXAI(model="grok-4-1-fast-reasoning", temperature=0.65, api_key=XAI_API_KEY)

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
    print("🧸 Teddy Bear - Improved Person Memory")

    while True:
        subprocess.run(["imagesnap", "-w", "1", "current.jpg"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        try:
            with open("current.jpg", "rb") as f:
                base64_img = base64.b64encode(f.read()).decode('utf-8')

            # Get description for identification
            msg = HumanMessage(content=[
                {"type": "text", "text": "Describe this person briefly for future recognition: hair, glasses, clothes color, face shape."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
            ])

            vision_resp = llm.invoke([msg])
            description = vision_resp.content.strip()
            print(f"Vision: {description}")

            # Better matching
            known = False
            for name, stored_desc in memory["known_people"].items():
                if name.lower() in description.lower() or description.lower() in stored_desc.lower():
                    print(f"Teddy: Hi {name}!")
                    known = True
                    break

            if not known:
                print("Teddy: Who are you?")
                # Voice input
                audio = sd.rec(int(5 * 16000), samplerate=16000, channels=1, dtype=np.float32)
                sd.wait()
                sf.write("temp.wav", audio, 16000)
                result = transcribe("temp.wav")
                text = result["text"].strip()

                if text:
                    print(f"You said: {text}")
                    name = text.title()
                    memory["known_people"][name] = description
                    save_memory()
                    print(f"Teddy: Hi {name}! Nice to meet you!")
        except Exception as e:
            print(f"Error: {e}")

        await asyncio.sleep(6)

if __name__ == "__main__":
    asyncio.run(main())
