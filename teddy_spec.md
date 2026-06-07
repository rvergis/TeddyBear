# Teddy Bear Spec

## Goal
Build a simple interactive teddy bear using the laptop camera and microphone.
It should recognize people and greet them by name.

## Core Requirements
- Use imagesnap for camera capture (Mac)
- Use Grok Vision (grok-4-1-fast-reasoning) to detect if there is a person
- Use mlx-whisper for voice recognition
- Maintain simple memory of known people

## Behavior Flow
1. Capture image every few seconds
2. If person detected:
   - If known person → Greet by name ("Hi Ron!")
   - If unknown person → Ask "Who are you?"
3. Listen for voice response and learn the name
4. Remember the person for future greetings

## Keep It Simple
- No complex emotions yet
- No Raspberry Pi code yet
- Focus on reliable detection + memory

## Current Tech Stack
- imagesnap (camera)
- Grok API (vision)
- mlx-whisper (voice)
- Python

Please improve the current teddy_bear/core.py based on this spec.
