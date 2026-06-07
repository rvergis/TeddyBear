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
   - Identify or learn the person (using vision + voice confirmation).
   - Greet them ("Hi <name>, I'm Teddy") if appropriate.
   - If this is the first/main person in the session, ask the main user: "Move the laptop around the room so I can look for other people."
3. While the user moves the laptop (scanning the room):
   - For any newly seen person (not matching known people):
     - Describe them using vision (e.g. "I see a middle-aged man with short dark receding hair, mustache, and olive skin tone.").
     - Ask the main user to identify by spelling: "I see [description]. Can you please spell out their name letter by letter?"
     - Follow the name confirmation process.
     - Once confirmed, store in memory with the visual description (to track other people).
     - Greet the new person: "Hi <other person>, I'm Teddy".
4. Name requests for unknown people (must succeed before remembering):
   - Always request names by spelling out letter by letter initially (for the main person or others): "Can you please spell out your name letter by letter?" or with description for others.
   - Listen for the spelled letters and reconstruct the name.
   - Confirm the candidate name out loud (e.g. "[Name]?").
   - Loop on confirmation:
     - Affirmative response → accept the name, greet ("Hi <name>, I'm Teddy"), store in memory with a visual description.
     - Negative response → loop to ask spelling again.
     - Name correction in the reply → switch candidate and re-confirm.
     - Unclear reply → repeat the confirmation question.
   - Use a safety limit on attempts. On repeated failure, give a short message without storing the name.
5. Only store names in memory after explicit confirmation from the main user. Known people are greeted directly (subject to cooldown). The main user can scan the room at any time by moving the laptop to discover and register additional people. Names are always initially requested via spelling.

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
- Names are always requested initially by spelling out letter by letter: "Can you please spell out your name letter by letter?" (or "I see [description]. Can you please spell out their name letter by letter?" when identifying others during a room scan).
- After extracting the spelled name, immediately confirm it out loud, for example: "I heard [Name]. Is that correct? Please say yes or no."
- Enter a confirmation loop:
  - On affirmative response: accept the name, greet with "Hi <name>, I'm Teddy", store it in memory with a visual description, and set last_greeted.
  - On negative response: loop to ask spelling again.
  - If the confirmation response contains a different name (e.g. "No, S-A-R-A-H"), switch the candidate name and re-confirm.
  - On unclear responses: repeat the confirmation question.
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
