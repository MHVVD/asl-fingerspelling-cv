# 01 — Setup, Capture, Landmarks, Features

## Goal
Get from "a webcam" to "a clean numeric description of the hand shape" that a classifier can learn from.

## What was done
| Step | Result |
|------|--------|
| Environment | `.venv` (system site-packages for OpenCV/PyTorch) + `mediapipe 1.1.0`, `scikit-learn`, `pytest`, `matplotlib` |
| Model file | Downloaded `hand_landmarker.task` (7.8 MB) from Google's MediaPipe model storage into `models/` |
| Stage 1 `capture.py` | `FrameSource`: opens webcam index or video file, returns BGR frames, smoothed FPS |
| Stage 2 `landmarks.py` | `HandDetector`: MediaPipe HandLandmarker -> 21 (x,y,z) points + handedness; `draw_hand` overlay |
| Stage 3 `features.py` | `normalize_landmarks`: mirror left hand, wrist to origin, scale by wrist->middle-MCP distance -> 63-d vector |

## How it works
**Capture.** OpenCV `VideoCapture` returns frames as BGR `uint8` arrays (H, W, 3). FPS is an exponential moving
average of 1/frame-time, so the number is stable instead of jittery.

**Landmarks.** MediaPipe runs two small neural nets: a *palm detector* that finds the hand in the whole frame, then a
*landmark model* on the cropped hand that regresses 21 keypoints (wrist, 4 joints per finger). In VIDEO mode it
tracks the hand between frames instead of re-detecting, which is why it is fast on CPU. x,y are normalized to
[0,1] of image width/height; z is depth relative to the wrist. Frames are converted BGR -> RGB because
MediaPipe expects RGB.

**Features.** Raw coordinates change if the hand moves, gets closer to the camera, or is the other hand. We remove those:
1. Left hand -> flip x, so one model serves both hands.
2. Subtract the wrist so the hand is position independent.
3. Divide by the wrist-to-middle-finger-base distance so the hand is size/distance independent.
What remains is the hand *shape* only. Unit tests (`tests/test_features.py`) prove translation invariance, scale invariance,
wrist-at-origin, and mirror equivalence.

## Results
- `pytest`: 5 passed.
- Blank frame -> no hands detected (no false positives).
- Webcam `/dev/video0` delivers 640x480 BGR frames.

## Pitfalls hit
- MediaPipe 1.1.0 no longer has the old `mp.solutions.hands` API; only the Tasks API (`HandLandmarker`) exists, and it
  needs a separate model file. Code was written against the Tasks API.
- VIDEO running mode requires strictly increasing timestamps; we increment a counter by 33 ms per frame.
- MediaPipe's z is only relative depth and noisy; if accuracy suffers we can test dropping z.

## Next
Stage 4: a recording tool to collect labeled landmark samples per letter.
