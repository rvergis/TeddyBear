import asyncio
import subprocess
import base64
import signal
import sounddevice as sd
import numpy as np
import soundfile as sf
import re
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

# Track which male voice was successfully selected (for debugging)
_VOICE = None


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
    """Speak using Mac 'say' (high quality, zero extra deps).
    Uses 'Alex' voice if available.
    """
    # Voice preference order: Alex first, then other voices as fallbacks.
    voices = [
        "Eddy (English (US))",
    ]
    for voice in voices:
        try:
            result = subprocess.run(
                ["say", "-v", voice, text],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if result.returncode == 0:
                global _VOICE
                if _VOICE is None:
                    _VOICE = voice
                    print(f"[Voice: {voice}]")  # one-time info on selected voice
                return  # success
        except Exception:
            continue

    # Final fallback: default system voice (no -v flag)
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
    """Compact known-people list for vision prompt.
    Uses stable identity descriptions (hair, face shape, etc.) per the spec's robustness requirements.
    Filters out junk entries.
    """
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
    - Identify known people robustly (handles glasses on/off, different head orientations/poses, and varying lighting)
    - Return a fresh description focused only on stable, identity-bearing features (per teddy_spec.md)
    """
    try:
        with open(image_path, "rb") as f:
            base64_img = base64.b64encode(f.read()).decode("utf-8")

        known_summary = get_known_summary(memory)

        prompt = f"""You are the vision module for an interactive teddy bear.

Your primary goal is reliable identification of known people despite real-world changes in appearance.

IMPORTANT ROBUSTNESS RULES (follow these strictly):
- Ignore whether the person is wearing glasses or not (people frequently put them on or take them off).
- Ignore head orientation, pose, and camera angle (common variations include looking down at a laptop, turned slightly away, or 3/4 view).
- Ignore lighting conditions, shadows, backlighting, or dim/bright environments.
- Match identity based only on stable features. When in doubt about a match due to these variations, use UNKNOWN instead of guessing.

Respond with EXACTLY three lines and nothing else:

LINE 1: YES if you see a recognizable person (face or upper body clearly visible), otherwise NO.

LINE 2: If LINE 1 is YES, output either:
  - the exact name of a person from the Known People list below if this is a confident match to their identity (even if glasses, pose, or lighting differ from their stored visual signature),
  - or the single word UNKNOWN if it does not match any known person well or confidence is low.
If LINE 1 is NO, output UNKNOWN.

LINE 3: If LINE 1 is YES, output ONE short objective sentence (max 20 words) describing stable, identity-bearing visual features useful for future re-identification. 
Prioritize: hair style/color/length, face shape, facial hair, approximate age appearance, skin tone, distinctive facial features, typical clothing style.
DO NOT mention or rely on: presence/absence of glasses, exact current clothing items or colors, current head pose/orientation, or lighting conditions. No names, no greetings.
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
    lower = t.lower().strip(".,!? ")

    # Reject obvious non-name words that commonly appear in confirmation responses
    non_name_words = {
        "yes", "no", "yeah", "yep", "yup", "nope", "nah", "ok", "okay",
        "correct", "right", "wrong", "sure", "affirmative", "negative", "incorrect"
    }
    if lower in non_name_words:
        return None

    for prefix in ("my name is ", "i am ", "i'm ", "call me ", "it's ", "this is ", "name is "):
        if lower.startswith(prefix):
            rest = t[len(prefix) :].strip()
            token = rest.split()[0].strip(".,!?\"' ") if rest else ""
            if token and token[0].isalpha():
                return token.title()

    # Capitalized words are often names
    caps = [w.strip(".,!?\"' ") for w in t.split() if w and w[0].isupper() and w.strip(".,!?\"' ").isalpha()]
    if caps:
        candidate = caps[-1].title()
        if candidate.lower() not in non_name_words:
            return candidate

    # Last word fallback
    last = t.split()[-1].strip(".,!?\"' ")
    if last and len(last) > 1 and last[0].isalpha():
        candidate = last.title()
        if candidate.lower() not in non_name_words:
            return candidate
    return None


def extract_spelled_name(transcript: str) -> str | None:
    """Extract a name when the user is spelling it out letter by letter.
    Handles things like "R O N", "R as in Romeo, O, N", or "R O N as in Ron".
    Falls back to normal name extraction if no clear spelling is detected.
    """
    if not transcript:
        return None
    t = transcript.lower().strip()

    # Remove common filler phrases like "as in ..."
    t = re.sub(r'\b(as in|like|for example)\s+\w+', '', t)

    # Extract individual single letters (a-z)
    letters = re.findall(r'\b([a-z])\b', t)
    if len(letters) >= 2:
        name = ''.join(letters).title()
        if len(name) >= 2 and name[0].isalpha():
            return name

    # Fallback: first letters of words
    words = re.findall(r'\b([a-z]+)\b', t)
    if words and len(words) >= 2:
        first_letters = ''.join(w[0] for w in words)
        if len(first_letters) >= 2:
            return first_letters.title()

    # Final fallback to the regular name extractor
    return extract_name(transcript)


def is_affirmative(text: str) -> bool:
    """Heuristic check if spoken response sounds like confirmation / yes."""
    if not text:
        return False
    t = text.lower().strip()
    yes_words = ("yes", "yeah", "yep", "yup", "correct", "right", "that's right",
                 "that's correct", "yessir", "sure", "uh huh", "mm-hmm", "affirmative")
    return any(word in t or t.startswith(word) for word in yes_words)


def is_negative(text: str) -> bool:
    """Heuristic check if spoken response sounds like denial / no."""
    if not text:
        return False
    t = text.lower().strip()
    no_words = ("no", "nope", "nah", "wrong", "not", "that's not", "negative", "incorrect")
    return any(word in t or t.startswith(word) for word in no_words)


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


def describe_scene(image_path: str) -> str:
    """Describe the current scene, objects, environment, and any people for room scanning."""
    try:
        with open(image_path, "rb") as f:
            base64_img = base64.b64encode(f.read()).decode("utf-8")

        prompt = "Describe the scene, objects, furniture, environment, and any people visible. Be concise (1-2 sentences). Note any distinct people with brief appearance details if present."

        msg = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"},
                },
            ]
        )

        resp = llm.invoke([msg])
        return resp.content.strip()
    except Exception as e:
        print(f"Scene description error: {e}")
        return ""


async def main():
    print("🧸 Teddy Bear — camera + vision + memory")
    memory = load_memory()
    print(f"   Loaded {len(memory.get('known_people', {}))} known people.")

    main_person = None
    scanning = False
    scan_remaining = 0
    shutdown_event = asyncio.Event()

    def _request_shutdown():
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _request_shutdown)

    try:
        while not shutdown_event.is_set():
            if not capture_image("current.jpg"):
                print("⚠️  Capture failed, retrying soon...")
                try:
                    await asyncio.wait_for(shutdown_event.wait(), timeout=1)
                except asyncio.TimeoutError:
                    continue
                continue

            try:
                has_person, known_name, fresh_desc = vision_detect_and_identify("current.jpg", memory)

                if has_person:
                    if known_name:
                        person = memory["known_people"].get(known_name)
                        if should_greet(person):
                            greeting = f"Hi {known_name}, I'm Teddy"
                            print(f"Teddy: {greeting}")
                            speak(greeting)
                            if isinstance(person, dict):
                                person["last_greeted"] = datetime.now().isoformat()
                                if fresh_desc:
                                    person["desc"] = fresh_desc
                            save_memory(memory)

                            if main_person is None:
                                main_person = known_name
                                speak("Move the laptop to scan.")
                                print("Teddy: Move the laptop to scan.")
                                scanning = True
                                scan_remaining = 30  # scan for ~30 seconds while user moves the laptop
                        else:
                            print(f"   (saw {known_name} — skipping repeat greeting)")
                    else:
                        # Unknown person: ask for name by spelling it out letter by letter (always),
                        # then confirm. See "Name Learning and Confirmation" section in teddy_spec.md.
                        confirmed_name = None
                        current_candidate = None
                        max_loops = 6  # safety limit to avoid infinite listening
                        loops = 0

                        while loops < max_loops and confirmed_name is None and not shutdown_event.is_set():
                            loops += 1

                            if current_candidate is None:
                                if main_person is not None:
                                    desc = fresh_desc or "someone new"
                                    speak(f"I see {desc}. Spell their name.")
                                    print(f"Teddy: I see {desc}. Spell their name.")
                                else:
                                    speak("Spell your name.")
                                    print("Teddy: Spell your name.")
                                transcript = listen_for_response(6.0)
                                current_candidate = extract_spelled_name(transcript)

                            if not current_candidate:
                                current_candidate = None
                                continue

                            # Confirmation step
                            speak(f"{current_candidate}?")
                            print(f"Teddy: {current_candidate}?")
                            response = listen_for_response(4.5)

                            # Allow user to correct the name in the same response (e.g. "No, it's Sarah")
                            corrected = extract_name(response)
                            if corrected and corrected.lower() != current_candidate.lower():
                                current_candidate = corrected
                                continue  # immediately re-confirm the corrected name

                            if is_affirmative(response):
                                confirmed_name = current_candidate
                            elif is_negative(response):
                                current_candidate = None
                            else:
                                # Unclear: just repeat the confirmation question next loop
                                pass

                        if confirmed_name:
                            memory["known_people"][confirmed_name] = {
                                "desc": fresh_desc or "First seen " + datetime.now().strftime("%Y-%m-%d"),
                                "last_greeted": datetime.now().isoformat(),
                            }
                            save_memory(memory)
                            greeting = f"Hi {confirmed_name}, I'm Teddy"
                            print(f"Teddy: {greeting}")
                            speak(greeting)

                            if main_person is None:
                                main_person = confirmed_name
                                speak("Move the laptop to scan.")
                                print("Teddy: Move the laptop to scan.")
                                scanning = True
                                scan_remaining = 30  # scan for ~30 seconds while user moves the laptop
                        else:
                            print("Teddy: Can't get name.")
                            speak("Can't get name.")
                else:
                    print("   No person detected.")

                # Room scanning: describe scene as user pans the camera
                if scanning and scan_remaining > 0:
                    scan_remaining -= 1
                    scene_desc = describe_scene("current.jpg")
                    if scene_desc:
                        print(f"Teddy sees: {scene_desc}")
                        # Speak the scene description as the camera is panned
                        short = scene_desc[:110] + "..." if len(scene_desc) > 110 else scene_desc
                        speak(short)
                    if scan_remaining <= 0:
                        scanning = False

            except Exception as e:
                print(f"Error in main loop: {e}")

            if not shutdown_event.is_set():
                try:
                    await asyncio.wait_for(shutdown_event.wait(), timeout=1)
                except asyncio.TimeoutError:
                    pass

    finally:
        save_memory(memory)
        print("\nShutting down cleanly...")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass  # Handled by signal handler for clean shutdown
