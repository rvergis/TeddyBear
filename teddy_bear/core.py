import asyncio
import subprocess
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
    print("🧸 Teddy Bear running with imagesnap (stable Mac camera)")

    while True:
        # Capture image using native tool
        subprocess.run(["imagesnap", "-w", "1", "current.jpg"], 
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        try:
            with open("current.jpg", "rb") as f:
                base64_img = base64.b64encode(f.read()).decode('utf-8')

            msg = HumanMessage(content=[
                {"type": "text", "text": "Is there a person in this image? Answer YES or NO first, then describe them briefly."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
            ])

            response = llm.invoke([msg])
            print(f"Grok: {response.content}\n")
        except Exception as e:
            print(f"Error: {e}\n")

        await asyncio.sleep(6)

if __name__ == "__main__":
    asyncio.run(main())
