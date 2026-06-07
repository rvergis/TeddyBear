# Teddy Bear Spec

## Goal
Build a simple interactive teddy bear using the laptop camera and microphone.
It should recognize people and greet them by name, introducing itself as "Teddy" (e.g. "Hi Ron, I'm Teddy").

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
1. Capture image every few seconds.
2. If a person is detected:
   - **Known person** (matches a name in memory, subject to cooldown): Immediately greet by name ("Hi Ron, I'm Teddy").
   - **Unknown person**: Ask "Who are you?", then follow the name confirmation process below.
3. Name confirmation for unknown people (must succeed before remembering):
   - Listen for the spoken name.
   - Confirm the candidate name out loud (e.g. "I heard [Name]. Is that correct?").
   - Loop on confirmation:
     - Affirmative response → accept the name, greet ("Hi <name>, I'm Teddy"), store in memory with a visual description, and remember for future.
     - Negative response or name correction in the reply → ask for the name again and repeat confirmation.
     - Unclear reply → ask for clarification and stay in the confirmation loop.
   - Use a safety limit on attempts. On repeated failure, give a polite message without storing the name.
4. Only store names in memory after explicit user confirmation during the voice interaction.

## Keep It Simple
- No complex emotions yet
- No Raspberry Pi code yet
- No local face embeddings or dedicated recognition models yet — rely on Grok Vision + carefully designed visual signatures in memory
- Focus on reliable detection + memory, with explicit attention to making identification robust to common real-world appearance changes

## Implementation Guidance for Robust Identification
- Memory entries store short, stable visual descriptions focused on invariant traits (not raw vision output or transient details like current glasses state).
- The vision identification prompt should include the list of known people + their signatures and explicitly instruct the model to match identity while discounting differences in glasses, head pose/orientation, and lighting.
- When the model has low confidence due to these variations, default to treating the person as unknown and ask for their name (prefer false negatives over false positive greetings).

## Name Learning and Confirmation
- Teddy must never permanently remember a person's name until the person has explicitly confirmed it during the voice interaction.
- After extracting a candidate name from the user's speech (using mlx-whisper), Teddy should immediately confirm it out loud, for example: "I heard [Name]. Is that correct?"
- Enter a confirmation loop:
  - On affirmative response (yes, yeah, correct, etc.): accept the name, greet with "Hi <name>, I'm Teddy", store it in memory with a visual description, and set last_greeted.
  - On negative response (no, nope, wrong, etc.): if this is the first incorrect attempt, on the next name request ask the person to spell out their name letter by letter (e.g. "Can you please spell out your name letter by letter?"). On subsequent retries, continue using spelling mode. Parse the spelled response (e.g. "R O N", "R as in Romeo O N") by extracting individual letters to form the candidate name.
  - If the confirmation response contains a different name (e.g. "No, it's Sarah"), switch the candidate name and re-confirm the new one.
  - On unclear responses, politely ask for clarification ("Please say yes, no, or tell me the correct name") and stay in the confirmation loop.
- Include a reasonable safety limit on total attempts (e.g. 5-6) to prevent the interaction from looping forever; on failure fall back to a polite message without saving the name.
- The confirmation step only applies to new/unknown people. Known people are greeted directly (subject to the cooldown).

## Speech Output
- Uses macOS `say` command for spoken output.
- Voice preference order:
  1. Alex
  2. Eddy (English (US)), Eddy (English (UK))
  3. Tom
  4. Other available voices as fallbacks (Junior, Fred, Daniel, etc.).
- Falls back to the system default voice only if none of the preferred voices are available.
- On first use, prints the selected voice (e.g. "[Voice: Alex]") for confirmation.
- To install the preferred voice: System Settings → Accessibility → Spoken Content → System Voice → Customize... and download "Alex".

## Current Tech Stack
- imagesnap (camera)
- Grok API (vision)
- mlx-whisper (voice)
- Python

Implement and evolve the teddy bear (starting with `teddy_bear/core.py`) to meet the requirements and robustness goals in this spec.
