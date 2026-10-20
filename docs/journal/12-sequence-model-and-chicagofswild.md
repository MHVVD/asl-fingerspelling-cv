# 12 — A Sequence Model, and ChicagoFSWild (in progress)

## Why
The real-world test (journal 11) scored 12/26 letters on a stranger. The model had learned two people's hands from still
poses, and it reads one frame at a time with hand-made smoothing rules. Two changes target that:
1. **Many signers:** ChicagoFSWild, 7,304 real-world fingerspelling clips from 160 signers.
2. **A sequence model:** read the whole clip and output the word, instead of a letter per frame plus rules.

## The dataset (checked from its README and CSV)
B. Shi, A. Martinez Del Rio, J. Keane, J. Michaux, D. Brentari, G. Shakhnarovich and K. Livescu,
"American Sign Language fingerspelling recognition in the wild", IEEE SLT 2018 (the dataset asks to be cited).
- 5,455 train / 981 dev / 868 test sequences; 87 / 37 / 36 signers; **no signer appears in two splits**.
- Fast, natural signing: median 22 frames for a median 4-letter label, about 5 frames per letter
  (my recordings hold each letter for 150 frames).
- Labels: a–z, space, and 12 rare characters (`. & @ '`). Vocabulary used: a–z, space, apostrophe.
- Signing-hand boxes for a subset of clips (`BBox/`), used to pick the right hand.
- Terms: no formal license; the page says the data is "released publicly in the interest of improving digital interfaces
  for signers". The clips are real people from online videos, so **frames and extracted landmarks stay local**
  (`data/chicagofswild/` is git-ignored); only code and scores are published.

## What was built (tested before the data arrived)
- `seqmodel.py`: NumPy runtime. Per frame: 63 shape numbers + wrist velocity in hand sizes + a hand-present flag (66).
  Network: Linear+ReLU → bidirectional GRU ×2 → Linear → letters + CTC blank. Greedy CTC decoding.
- `seqtrain.py`: PyTorch training (CPU) with CTC loss, packed sequences, best epoch chosen on validation.
- Test: the NumPy forward pass matches PyTorch to 1e-4. (PyTorch's 20–40 s import stays out of the apps.)
- `extract_fsw.py`: per frame, MediaPipe with two hands; signing hand = annotated box > continuity with the previous
  frame > raised hand. 6 tests.
- `fsw_train.py`: scores today's system on the test signers, trains the sequence model (selected on dev), scores on test.
- Note: `torch` reports the MX130 GPU (sm_50) as unsupported, confirming CPU-only training.

## Sanity check on simulated words (`seqsim.py`)
Words spelled with my recorded frames: train on sessions 1–3 (4,000 words), test on session 4 (200 words).

| System | Error rate | Exact words |
|---|---|---|
| Today's per-frame model + smoothing, **trained without session 4** | **0.26** | 14% |
| Sequence model, trained on my 3 sessions only (290k parameters, 23 min CPU) | 0.42 | 4% |

**The sequence model lost.** It learns (error 1.00 → ~0.40 on the unseen session) but overfits: training loss fell to 0.08
while test error stalled. Three sessions of one person are too little variety, and the simulation (letters held 6–18
frames, smooth made-up transitions) suits the per-frame approach.

**A mistake caught:** my first comparison used `letter_mlp_final`, which was trained on all four of my sessions,
*including the test session*. That made the baseline look better (0.23). The fair figure above uses a model trained without
session 4. Lesson: check what every model in a comparison has seen.

## Getting the data
- **Download failed once.** After 6.2 GB the server closed the connection and the network dropped for minutes. I had used
  `curl --retry` with `-C -` (resume); on each retry curl rewinds the file to where *that command* started (0 bytes), so the
  partial download was wiped. Re-downloaded with `wget -c --tries=0`, which re-measures the file before every retry:
  14,270,632,063 bytes, 2 h 46 min. Lesson: test the resume path before trusting a long download to it.
- Unpacked: 205,139 frames in 7,304 clip folders; none missing; frame counts match the CSV.
- The footage really is "in the wild": webcam vlogs, news studios, an interpreter in a small corner window, two people in
  one shot, and a background graphic of hands.

## Picking the signing hand, take two
A 60-clip trial reported a hand in 100% of frames, which was too good. Drawing the chosen hand on the frames showed why:
when MediaPipe saw only one hand, my per-frame rule accepted it, even when it was the *other* hand, and "follow the previous
hand" then kept following the wrong one (e.g. the skeleton jumping from the spelling hand to a pointing hand mid-clip).

Fix (`select_signing_track`): look at the whole clip first. Link detections into tracks by wrist position (within
2.5 hand sizes), then choose one track for the clip: the one overlapping the dataset's signing-hand boxes most, else the one
held highest on average (among tracks present for at least 30% of the longest one). Frames where only the other hand was
seen become "no hand". Four tests. Re-drawn on the same frames: both wrong cases now follow the spelling hand; hand found in
99% of frames.

## Results on ChicagoFSWild (test = 36 signers never seen in training)
Extraction: 58 min on CPU, hand found in 97–98% of frames in every split.

| System | Test letter accuracy |
|---|---|
| Today's per-frame model + smoothing, as built (0.2 s hold) | 9.4% (mostly empty outputs: real signers hold a letter for ~5 frames) |
| Same, hold and threshold tuned on the **dev** signers (best: hold 0.10 s, threshold 0.4) | 25.3% |
| **Sequence model (CTC), trained on 87 train signers, epoch chosen on dev (66.8%)** | **67.3%** |

Sequence model: 510k parameters (2 MB), 40 epochs, 31 minutes on the laptop CPU. Dev accuracy levelled off at 66–67% from
epoch ~17 while training loss kept falling, so more epochs would not help.

**Published results on the same test set** (read from the papers' tables):
- Shi et al., SLT 2018 (the dataset's paper; hand crops + CTC + language model): 41.9%.
- Shi et al., ICCV 2019 (iterative visual attention), trained on ChicagoFSWild: up to 45.1%;
  with 50k extra crowdsourced clips (ChicagoFSWild+): up to 61.2%.

So on the same training data this scores higher than the dataset's original published results. Fair framing: those are
2018–2019 image-based models; much of the gain here comes from MediaPipe, a modern pre-trained hand tracker they did not
have. This is not a claim to be state of the art, and newer work was not checked.

**Main remaining error: double letters.** `jasll → jasl`, `issue → isue`, `taboo → tabo`, `all → a`. Greedy CTC needs a
"blank" between two identical letters; fast signers often give none.

## The Travis clip (journal 11), with the new model
| Run | Letters right | Error rate |
|---|---|---|
| Per-frame model, after the wrong-hand fix (run 2) | 12 / 26 | 0.73 |
| **Sequence model, whole clip at native speed (run 3)** | **9 / 26** | 0.65 (fewer insertions, many letters missed) |

Diagnosis, not tuning: Travis spells deliberately (~30 frames per letter); ChicagoFSWild signers use ~5. Keeping every
k-th frame to match the training speed gave 18–19 / 26 for every k from 2 to 6 (output close to the full alphabet:
`ABCDEFGHJKLMNOPRSTUXY…`). The setting was chosen after seeing the clip, so the headline stays **9 / 26**; the proper fix
is training-time: augment with slow signing (the current speed augmentation is only 0.8–1.25×). Slow, deliberate signing
is also how learners sign in the tutor, so this matters beyond one clip.

## Next
1. Retrain with a wide speed augmentation (slow signing), re-check ChicagoFSWild test (must not drop) and Travis.
2. Double letters: beam search, possibly with a small language model.
3. Use the sequence model in the video translator.
