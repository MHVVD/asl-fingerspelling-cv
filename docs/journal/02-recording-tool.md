# 02 — Recording Tool (Stage 4, part 1)

## Goal
Collect your own labeled hand samples from the webcam so the model sees your camera, lighting and hand.

## What was done
- `src/aslcv/dataset.py`: `Recorder` (state machine) and `load_dataset`. This is camera-free and unit-tested (`tests/test_dataset.py`, 5 tests).
- `src/aslcv/record.py`: thin OpenCV UI. Webcam -> flip -> MediaPipe -> draw skeleton -> normalize -> `Recorder.add`.

## How it works
- Each frame with a detected hand becomes one 63-d feature row (the same `normalize_landmarks` used everywhere else),
  using the detector's handedness label so left hands are mirrored to match right hands.
- Select a letter (A–Z key), press SPACE to start. It stores one sample per frame and auto-stops at the target (150 by default).
- `BACKSPACE` undoes the last recording burst (e.g. a bad take). `ENTER` saves. `ESC` saves and quits.
- A reference panel on the right shows the selected letter's hand shape (see "Reference panel" below).
- Output: one `data/recorded/<session>.npz` per run holding `X`, `y`, `session`.

## Why session IDs
Consecutive frames of one recording are near-duplicates. A random train/test split would put almost-identical frames on both
sides and report inflated accuracy. Splitting by session keeps whole recordings together, so test accuracy means something.

## Recording tips (for better data)
- Vary: small rotations, distance, tilt, and lighting between bursts. Slowly move the hand while recording.
- Do 2+ separate sessions (different times or lighting) so there is something to hold out for testing.
- J and Z involve motion; record a static approximation for now (the limitation is documented).

## Reference panel (`src/aslcv/reference.py`)
**What:** a stick-figure hand for each letter, drawn next to the webcam view. Full sheet: `artifacts/reference_alphabet.png`
(regenerate with `python -m aslcv.reference`).

**How:** no images are used. A small *kinematic hand model* generates 21 landmarks in MediaPipe's layout:
- Each finger is a chain of 3 bones from its knuckle (MCP). Each joint has a bend angle; angles add up along the chain,
  and bending rotates the bone from "straight up" toward the viewer: `direction = up*cos(total) + toward_viewer*sin(total)`.
  Presets: straight, fist curl, bent (E), hook (X), C-curve, O-curve.
- The thumb is placed by its tip position, with joints interpolated and bowed outward. It can also "touch" a fingertip (D, F, O, W).
- Sideways/downward letters (G, H, P, Q) rotate the whole hand in the image plane; J and Z get a motion arrow.
- Bones are drawn far-to-near and brighter when nearer, a cheap depth cue.
- One fixed scale for all letters, so a fist really looks smaller than an open hand.

**Why a model instead of photos:** no licensing or download, and the poses are in the same 21-point format as live detections,
so the tutor stage can compare your hand against them.

**Limits:** these are approximations. Fist-family letters (A, E, M, N, S, T) differ mostly in thumb depth, which a 2D stick figure
shows poorly, so each letter also has a written description. Verify with a real ASL chart if a shape looks off.

## Pitfalls / notes
- Bug fixed before first use: the original controls were Q/S/U, which made the letters Q, S and U impossible to select.
  Controls moved to ESC / ENTER / BACKSPACE.
- Camera frame is mirrored before detection (natural to watch). MediaPipe's handedness label assumes mirrored input, so it is correct as-is.
- The UI itself needs a real webcam and window, so it was import-checked but only the `Recorder` logic is covered by automated tests.
