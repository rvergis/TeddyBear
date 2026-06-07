# Teddy Bear Spec

## Goal
Build a simple interactive teddy bear using the laptop camera and microphone.
It should recognize people and greet them by name.

## Core Requirements
- Use imagesnap for camera capture (Mac)
- Use Grok Vision (grok-4-1-fast-reasoning) for both person detection and identification of known people
- Use mlx-whisper for voice recognition
- Maintain simple memory of known people

## Vision & Recognition Requirements
The vision system is the foundation of reliable "known vs unknown" decisions. It must handle real-world variability:

- **Glasses invariance**: Correctly recognize the same person whether they are wearing glasses or not (glasses are a strong but unreliable cue — people frequently remove them).
- **Orientation / pose robustness**: Work across different head angles and orientations, including:
  - Frontal view
  - 3/4 or slight profile
  - Looking down (common for laptop camera angles)
  - Head turned slightly away
  - Close-up or partial face (e.g. only upper body or lower face visible)
- **Lighting robustness**: Perform under varied lighting conditions, including:
  - Overhead room lighting
  - Side lighting / window light
  - Backlit or silhouetted subjects
  - Dim / low-light indoor environments
  - Mixed or changing lighting

When generating or matching visual signatures for memory:
- Prioritize stable, identity-bearing features: hair style/color/length, face shape, facial hair, approximate age appearance, skin tone, distinctive facial features, typical clothing style.
- De-emphasize or ignore transient cues such as presence/absence of glasses, exact current clothing, current pose, or lighting artifacts.
- The identification prompt should explicitly instruct the model to match identity even when glasses, pose, or lighting differ from previously stored descriptions.

If the model cannot confidently match a person due to these variations, it should treat them as unknown and ask for their name rather than guessing.

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
- No local face embeddings or dedicated recognition models yet — rely on Grok Vision + carefully designed visual signatures in memory
- Focus on reliable detection + memory, with explicit attention to making identification robust to common real-world appearance changes

## Implementation Guidance for Robust Identification
- Memory entries store short, stable visual descriptions focused on invariant traits (not raw vision output or transient details like current glasses state).
- The vision identification prompt should include the list of known people + their signatures and explicitly instruct the model to match identity while discounting differences in glasses, head pose/orientation, and lighting.
- When the model has low confidence due to these variations, default to treating the person as unknown and ask for their name (prefer false negatives over false positive greetings).

## Current Tech Stack
- imagesnap (camera)
- Grok API (vision)
- mlx-whisper (voice)
- Python

Implement and evolve the teddy bear (starting with `teddy_bear/core.py`) to meet the requirements and robustness goals in this spec.
