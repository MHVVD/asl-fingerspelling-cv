# 03 — Public Dataset Extraction (Stage 4, part 2)

## Goal
Turn the Kaggle "ASL Alphabet" images into the same 63-number landmark features the recorder produces,
so public and personal data can be mixed and compared.

## What was done
- `src/aslcv/extract_public.py`: finds letter folders, runs MediaPipe on each image (6 worker processes),
  normalizes, and writes `data/public/asl_alphabet.npz` plus a report and a sample sheet in `artifacts/`.
- `src/aslcv/landmarks.py`: added `mode="image"` (independent photos) and `landmarks_px` (aspect-correct coordinates).
- `src/aslcv/dataset.py`: `load_dataset` now accepts per-row session IDs.
- Tests: `tests/test_extract_public.py` (discovery, sessions, padding fallback with a fake detector),
  `tests/test_landmarks.py` (aspect correction), plus a regression test in `tests/test_dataset.py`.

## The dataset
`data/raw/asl_alphabet_train/asl_alphabet_train/<CLASS>/<CLASS><n>.jpg`: 29 classes × 3,000 images, 200×200 px.
We use the 26 letters (78,000 images) and skip `del`, `nothing`, `space` and the 28-image test folder.
One signer, similar backgrounds, frames numbered 1..3000 per class.

## How it works, step by step
1. **Discover**: walk `data/raw`, keep images whose parent folder is a single letter A–Z; read the frame number from the file name.
2. **Detect (IMAGE mode)**: each photo is independent, so MediaPipe detects from scratch instead of tracking as in video.
3. **Padding fallback**: if no hand is found, add a black border of 25% and retry. The palm detector struggles when the hand
   fills the frame; the border gives it context. A probe of 132 random images: plain 77%, plain-or-padded 84%.
4. **Pixel space**: convert MediaPipe's [0,1] coordinates to pixels (see bug below).
5. **Normalize**: the same `normalize_landmarks` as live frames, so both sources share one feature space.
6. **Pseudo-sessions**: frames are consecutive, so neighbours are near-duplicates. Frames are grouped into blocks of 500
   (`asl_alphabet:b0..b5`) so training can split by block and avoid leakage, the same idea as recorder session IDs.

## Decisions and why
- **Force handedness = Right.** In the probe MediaPipe labeled 105/111 hands "Right" and 6 "Left". One signer with one hand means
  the "Left" labels are errors, and trusting them would mirror those samples into the wrong shape. The raw label counts are still
  recorded in the report (`mediapipe_handedness_labels`). Override with `--handedness auto`.
- **Keep `files` and `padded` per row** in the npz so any odd sample can be traced back to its image.

## Bugs found and fixed
- **Aspect ratio (pipeline-wide):** MediaPipe x is a fraction of width and y of height. On a 640×480 webcam this squashes the hand
  relative to 200×200 dataset images, so identical hands would give different features. Fixed by `to_pixel_space` (x·w, y·h, z·w)
  before normalizing; tested by `test_same_hand_gives_same_features_on_4x3_and_square_images`.
- `np.savez(..., file=...)` crashed: `file` is `savez`'s own first parameter name. Renamed the array to `files`.
- `load_dataset` assumed one session ID per file and turned the per-row array into a single giant string. Fixed and regression-tested.

## Results
Trial (20 images per letter): 399/520 detected (76.7%), 119 rescued by padding. The first 20 frames of D, E and M were
mostly undetected, so frame order matters and small trials are not representative.
Sample overlays: `artifacts/extraction_samples.png`, where landmarks fit the hands. Some dataset noise is visible
(the T example looks closer to a thumb-out shape; W has stray thumb points).

### Full run
| Metric | Value |
|---|---|
| Letter images | 78,000 |
| Hands detected | **65,707 (84.2%)** |
| Rescued by the padding retry | 5,425 (7% of all images) |
| MediaPipe handedness labels | Right 62,473 / Left 3,234 (4.9% "Left", overridden to Right) |
| Time | 1,134 s (~19 min, 6 workers, ~69 img/s) |
| Output | `data/public/asl_alphabet.npz`, 65,707 × 63, 26 letters, 6 pseudo-sessions (b0–b5) |

Per-letter detection rate:
`A 77 · B 76 · C 75 · D 89 · E 79 · F 99 · G 88 · H 84 · I 84 · J 91 · K 93 · L 91 · M 69 · N 61 · O 83 · P 74 · Q 80 · R 90 · S 92 · T 86 · U 88 · V 91 · W 90 · X 84 · Y 89 · Z 87` (%)

**Reading the numbers.** The weakest letters are N (61%) and M (69%): the thumb is hidden under the fingers, so the hand
is a compact blob with few visible keypoints. Every letter still has 1,800+ samples, so no class is starved, but the
detector's bias means *the hardest letters for MediaPipe are also the most under-represented*. Watch M/N in the confusion matrix.
The 4.9% "Left" labels confirm the decision to force handedness: those samples would otherwise have been mirrored.
