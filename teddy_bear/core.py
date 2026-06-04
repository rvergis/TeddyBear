import cv2
import asyncio
import base64
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
    print("🧸 Teddy Bear with Debug Images")

    cap = cv2.VideoCapture(0)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            await asyncio.sleep(3)
            continue

        frame_count += 1
        
        # Save raw debug image
        cv2.imwrite(f"debug_frame_{frame_count}.jpg", frame)
        print(f"✅ Saved debug_frame_{frame_count}.jpg")

        # Low res for Grok
        small = cv2.resize(frame, (480, 360))
        _, buffer = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 70])
        base64_img = base64.b64encode(buffer).decode('utf-8')

        msg = HumanMessage(content=[
            {"type": "text", "text": "Is there a person? Answer YES or NO first, then describe them briefly."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
        ])

        try:
            response = llm.invoke([msg])
            print(f"Grok: {response.content}\n")
        except Exception as e:
            print(f"Error: {e}\n")

        await asyncio.sleep(7)

if __name__ == "__main__":
    asyncio.run(main())
