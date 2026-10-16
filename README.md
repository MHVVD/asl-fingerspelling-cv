# ASL Fingerspelling: Tutor + Video Translator

A computer-vision project that recognises American Sign Language fingerspelling (A–Z) from a camera or a video.
One recognition model powers two modes:

| Mode | What it does | Input | Output |
|---|---|---|---|
| **1. Tutor** | Teaches the alphabet: shows a target letter, watches your hand live, tells you if it's right and what to fix, tracks progress | Webcam | Live feedback, progress stats |
| **2. Video translator** | A real-world test: runs any fingerspelling video through the model and subtitles it | A video file (own recording or real-world footage) | Captioned MP4 + `.srt` subtitles |

## Pipeline
```
frame ─▶ MediaPipe hand landmarks (21 pts) ─▶ pixel space ─▶ normalize (position/size/hand) ─▶ classifier ─▶ letter
                                                                                                   │
                     Tutor:  compare to target letter, give feedback ◀──────────────────────────────┤
                     Video:  smooth over frames ─▶ steady letters ─▶ words ─▶ subtitles ◀──────────┘
```

## Status
| Stage | Status |
|---|---|
| 1 Capture · 2 Landmarks · 3 Features | done |
| 4 Data: recorder with reference panel + live hand check, public ASL Alphabet extraction (65,707 samples), 4 personal sessions | done |
| 5 Training + evaluation: NumPy-only MLP, 73 KB, 0.08 ms/frame | done |
| 6 Smoothing (letter stream, double letters, spaces) + shared frame pipeline; 0.22 char error rate on simulated words | done |
| 7b Video translator: captioned MP4 + SRT, two-hand aware, 33–55 frames/s on CPU | built; first real-world test 12/26 letters |
| 7a Tutor: spaced repetition, hold-to-confirm, finger-level tips (2.4% false alarms), teach mode, debug view | built; awaiting a live session |
| 8 Stretch: CNN comparison, motion letters (J, Z) | planned |

## Results so far
| Test | Accuracy |
|---|---|
| Public signer, held-out frames | 95.0% |
| **A new session of my own hand** (model trained on public data + my 3 earlier sessions) | **72.6%** |
| Same, public data only | 42.4% (a model of one person does not transfer to another) |
| Spelling words from real held-out frames, after smoothing | 0.19 character error rate |
| **Real-world video: a stranger fingerspelling A–Z** ([CC BY-SA clip](https://commons.wikimedia.org/wiki/File:American_Sign_Language_demo_-_Travis_Dougherty.ogv)) | **12 / 26 letters** (up from 1/26 after fixing which hand is read; journal 11) |

Weakest letters: thumb-position fists (M, N, S, T), crossed/spread pairs (R/U/V), and letters signed inconsistently between
sessions. Details and the debugging stories (including two mirroring bugs) are in the journal.

## Built for a laptop CPU
Runs on an Intel i5-8265U with no GPU required. The runtime needs only OpenCV, MediaPipe and NumPy (no PyTorch).
The classifier is 18k parameters (73 KB) and costs 0.08 ms per frame; MediaPipe (~15–25 ms) is the only real cost.
Training takes under 2 minutes and about 300 MB RAM.

## How everything was built
Step-by-step journal with what was done, why, how, results and pitfalls: [`docs/journal/`](docs/journal/).

## Setup
```bash
git clone https://github.com/MHVVD/asl-fingerspelling-cv.git && cd asl-fingerspelling-cv
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# MediaPipe's hand landmark model (~7.8 MB, from Google):
mkdir -p models && curl -fsSL -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```
The trained letter model (`models/letter_mlp_final.npz`) and tip table (`models/finger_states.json`) are included,
so the tutor and translator run straight away. To retrain from scratch you also need the public dataset (below).

## Data
- `data/recorded/`: my own recordings: 4 sessions of hand **landmarks** (21 points × 3 numbers per frame, no images),
  made with `aslcv.record`. Sessions 1 and 3 were repaired for mirroring bugs; see journals 05 and 07.
- Public data (not included): the [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) dataset on Kaggle.
  Download it into `data/raw/`, then run `aslcv.extract_public` to create `data/public/asl_alphabet.npz`.

## Usage
```bash
source .venv/bin/activate
PYTHONPATH=src python -m aslcv.record          # record your own samples (reference hand shown on the right)
PYTHONPATH=src python -m aslcv.extract_public  # landmarks from the Kaggle ASL Alphabet set in data/raw/
PYTHONPATH=src python -m aslcv.train           # train + evaluate (writes models/ and artifacts/)
PYTHONPATH=src python -m aslcv.tutor_app      # the tutor (SPACE skip, ENTER teach, D debug, ESC quit)
PYTHONPATH=src python -m aslcv.record_clip     # record a test clip into clips/
PYTHONPATH=src python -m aslcv.translate clips/<clip>.mp4   # captioned video + .srt into outputs/
python -m pytest -q                            # tests
```
