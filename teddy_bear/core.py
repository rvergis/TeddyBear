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
from datetime import datetime

load_dotenv()

XAI_API_KEY = os.getenv("X_API_KEY_TEDDYBEAR")
if not XAI_API_KEY:
    raise ValueError("❌ X_API_KEY_TEDDYBEAR not set!")

llm = ChatXAI(model="grok-4-1-fast-reasoning", temperature=0.3, api_key=XAI_API_KEY)

MEMORY_FILE = Path("teddy_memory.json")
GREET_COOLDOWN_SECONDS = 180  # avoid repeating greetings while someone lingers


def save_memory(memory: dict):
    MEMORY_FILE.write_text(json.dumps(memory, indent=2))


def load_memory() -> dict:
    memory = {"known_people": {}}
    if MEMORY_FILE.exists():
        try:
            data = json.loads(MEMORY_FILE.read_text())
            if isinstance(data, dict):
                memory = data
        except Exception:
            pass

    # Upgrade legacy flat format {"Ron": "desc string"} -> {"Ron": {"desc": "...", "last_greeted": ...}}
    known = memory.get("known_people", {})
    if known and isinstance(next(iter(known.values()), None), str):
        upgraded = {}
        for name, desc in known.items():
            upgraded[name] = {"desc": desc, "last_greeted": None}
        memory["known_people"] = upgraded
        save_memory(memory)
    return memory


def speak(text: str):
    """Speak using Mac 'say' (high quality, zero extra deps). Falls back silently."""
    try:
        subprocess.run(
            ["say", text],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except Exception:
        print(f"🔊 {text}")


def capture_image(path: str = "current.jpg") -> bool:
    """Capture photo with imagesnap. Returns True if file exists."""
    try:
        result = subprocess.run(
            ["imagesnap", "-w", "0.5", path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return result.returncode == 0 and Path(path).exists()
    except Exception:
        return False


def get_known_summary(memory: dict) -> str:
    """Compact known-people list for vision prompt (filters out junk entries)."""
    lines = []
    for name, info in memory.get("known_people", {}).items():
        if not isinstance(info, dict):
            continue
        desc = (info.get("desc") or "")[:110].replace("\n", " ")
        if len(name) >= 2 and name[0].isupper() and len(name.split()) <= 3:
            lines.append(f"- {name}: {desc}")
    return "\n".join(lines) if lines else "None yet."


def vision_detect_and_identify(image_path: str, memory: dict) -> tuple[bool, str | None, str]:
    """
    One vision call to:
    - Detect person presence
    - Match against known people using the actual photo + stored visual signatures
    - Return fresh 1-sentence description for memory
    """
    try:
        with open(image_path, "rb") as f:
            base64_img = base64.b64encode(f.read()).decode("utf-8")

        known_summary = get_known_summary(memory)

        prompt = f"""You are the vision module for an interactive teddy bear.

Examine the image carefully.

Respond with EXACTLY three lines and nothing else:

LINE 1: YES if you see a recognizable person (face or upper body clearly visible), otherwise NO.

LINE 2: If LINE 1 is YES, output either:
  - the exact name of a person from the Known People list below if the face/clothing/hair/glasses etc. is a confident visual match,
  - or the single word UNKNOWN if it does not match any known person well.
If LINE 1 is NO, output UNKNOWN.

LINE 3: If LINE 1 is YES, output ONE short objective sentence (max 20 words) describing distinctive visual features useful for future re-identification: hair, glasses, age appearance, facial hair, clothing color/style, face shape. No names, no greetings.
If LINE 1 is NO, leave LINE 3 blank.

Known People (name: visual signature):
{known_summary}
"""

        msg = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"},
                },
            ]
        )

        vision_resp = llm.invoke([msg])
        vision_text = vision_resp.content.strip()
        print(f"Grok Vision: {vision_text[:220]}{'...' if len(vision_text) > 220 else ''}")

        lines = [ln.strip() for ln in vision_text.splitlines() if ln.strip()]
        if not lines:
            return False, None, ""

        has_person = lines[0].upper().startswith("Y")

        identified = None
        desc = ""

        if has_person and len(lines) >= 2:
            candidate = lines[1].strip()
            if candidate and candidate.upper() != "UNKNOWN":
                if candidate in memory.get("known_people", {}):
                    identified = candidate
            if len(lines) >= 3:
                desc = lines[2].strip()

        return has_person, identified, desc

    except Exception as e:
        print(f"Vision error: {e}")
        return False, None, ""


def listen_for_response(duration: float = 6.0) -> str:
    """Record from mic and transcribe using mlx-whisper."""
    print("🎤 Listening... (speak your name clearly)")
    try:
        audio = sd.rec(int(duration * 16000), samplerate=16000, channels=1, dtype=np.float32)
        sd.wait()
        sf.write("temp.wav", audio, 16000)

        result = transcribe("temp.wav", path_or_hf_repo="mlx-community/whisper-tiny")
        text = (result.get("text") or "").strip()
        if text:
            print(f'   Heard: "{text}"')
        return text
    except Exception as e:
        print(f"Listen error: {e}")
        return ""


def extract_name(transcript: str) -> str | None:
    """Robust heuristic to pull a name from a spoken reply."""
    if not transcript or len(transcript) < 2:
        return None

    t = transcript.strip()
    lower = t.lower()

    for prefix in ("my name is ", "i am ", "i'm ", "call me ", "it's ", "this is ", "name is "):
        if lower.startswith(prefix):
            rest = t[len(prefix) :].strip()
            token = rest.split()[0].strip(".,!?\"' ") if rest else ""
            if token and token[0].isalpha():
                return token.title()

    # Capitalized words are often names
    caps = [w.strip(".,!?\"' ") for w in t.split() if w and w[0].isupper() and w.strip(".,!?\"' ").isalpha()]
    if caps:
        return caps[-1].title()

    # Last word fallback
    last = t.split()[-1].strip(".,!?\"' ")
    if last and len(last) > 1 and last[0].isalpha():
        return last.title()
    return None


def should_greet(person: dict | None) -> bool:
    if not isinstance(person, dict):
        return True
    lg = person.get("last_greeted")
    if not lg:
        return True
    try:
        last = datetime.fromisoformat(lg)
        return (datetime.now() - last).total_seconds() > GREET_COOLDOWN_SECONDS
    except Exception:
        return True


async def main():
    print("🧸 Teddy Bear — camera + vision + memory")
    memory = load_memory()
    print(f"   Loaded {len(memory.get('known_people', {}))} known people.")

    while True:
        if not capture_image("current.jpg"):
            print("⚠️  Capture failed, retrying soon...")
            await asyncio.sleep(2)
            continue

        try:
            has_person, known_name, fresh_desc = vision_detect_and_identify("current.jpg", memory)

            if has_person:
                if known_name:
                    person = memory["known_people"].get(known_name)
                    if should_greet(person):
                        greeting = f"Hi {known_name}!"
                        print(f"Teddy: {greeting}")
                        speak(greeting)
                        if isinstance(person, dict):
                            person["last_greeted"] = datetime.now().isoformat()
                            if fresh_desc:
                                person["desc"] = fresh_desc
                        save_memory(memory)
                    else:
                        print(f"   (saw {known_name} — skipping repeat greeting)")
                else:
                    # Unknown → ask, learn, remember per spec
                    speak("Who are you?")
                    print("Teddy: Who are you?")

                    transcript = listen_for_response(5.5)
                    name = extract_name(transcript)

                    if name:
                        memory["known_people"][name] = {
                            "desc": fresh_desc or "First seen " + datetime.now().strftime("%Y-%m-%d"),
                            "last_greeted": datetime.now().isoformat(),
                        }
                        save_memory(memory)
                        greeting = f"Hi {name}! Nice to meet you!"
                        print(f"Teddy: {greeting}")
                        speak(greeting)
                    else:
                        print("Teddy: Sorry, I didn't catch that.")
            else:
                print("   No person detected.")

        except Exception as e:
            print(f"Error in main loop: {e}")

        await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(main())
