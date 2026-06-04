import cv2
import asyncio
import base64
import sounddevice as sd
import numpy as np
import soundfile as sf
from mlx_whisper import transcribe
from langchain_xai import ChatXAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

XAI_API_KEY = os.getenv("X_API_KEY_TEDDYBEAR")
if not XAI_API_KEY:
    raise ValueError("❌ X_API_KEY_TEDDYBEAR not set!")

llm = ChatXAI(model="grok-4-1-fast-reasoning", temperature=0.7, api_key=XAI_API_KEY)

async def main():
    print("🧸 Teddy Bear with mlx-whisper (Apple Silicon optimized)")

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            await asyncio.sleep(3)
            continue

        small = cv2.resize(frame, (480, 360))
        _, buffer = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 70])
        base64_img = base64.b64encode(buffer).decode('utf-8')

        msg = HumanMessage(content=[
            {"type": "text", "text": "Is there a person? Answer YES or NO first, then describe them briefly."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
        ])

        try:
            vision_response = llm.invoke([msg])
            vision_text = vision_response.content.strip()
            print(f"Grok Vision: {vision_text}")

            if "yes" in vision_text.lower() or "person" in vision_text.lower():
                print("🎤 Teddy: Who are you?")
                
                # Voice Recording + Transcription
                try:
                    print("Listening...")
                    audio = sd.rec(int(5 * 16000), samplerate=16000, channels=1, dtype=np.float32)
                    sd.wait()
                    
                    temp_file = "temp_voice.wav"
                    sf.write(temp_file, audio, 16000)
                    
                    result = transcribe(temp_file, path_or_hf_repo="mlx-community/whisper-tiny")
                    text = result["text"].strip()
                    
                    if text:
                        print(f"You said: {text}")
                        name = text.title()
                        print(f"Teddy: Hi {name}! Nice to meet you!")
                except Exception as e:
                    print(f"Voice error: {e}")
                    name = input("Type your name: ").title()
                    print(f"Teddy: Hi {name}! Nice to meet you!")

        except Exception as e:
            print(f"Error: {e}")

        await asyncio.sleep(7)

if __name__ == "__main__":
    asyncio.run(main())
