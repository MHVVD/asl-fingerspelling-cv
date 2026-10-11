# 05 — Personal Data, the Three-Way Comparison, and a Mirroring Bug

## Goal
Measure how well the model works on a real person who is not in the public dataset, and compare three training choices,
all tested on the same held-out session (`20261011-011708`, "session 2"), which no model learns from:
- **public**: the Kaggle signer only
- **recorded**: my session 1 (`20261011-003953`) only
- **both**: public + session 1

## Code change
`train.py` now separates *what the model learns from* (`--data`) from *what it is tested on* (`--test-sessions`, looked up in all
data). Before, a public-only model could not be tested on my session. New helper `make_split` with a test.

## First results looked wrong
Several letters were confused **100%** of the time (S→M, Q→P, R→V, Z→X, D→O). Random noise gives scattered errors,
not all-or-nothing swaps, so this was investigated before trusting any numbers.

## Debugging, step by step
1. **Wrong labels?** No. Both sessions recorded A→Z in order; no burst was saved under the wrong key.
2. **Per-letter centroids** (average shape of each letter per session): most letters moved <1.0 between sessions, but
   **P moved 4.22 and Q 5.60**. Session 2's P was much closer to session 1's P *mirrored* (1.73).
3. **Chirality test**: the sign of the cross product (index knuckle × pinky knuckle) flips when a hand is mirrored.
   Every letter was consistent across sessions except **P and Q: 100% one way in session 1, 0% in session 2.**
4. **Mirror or physical turn?** Showing the back of the hand would flip x *and* depth; a software mirror flips only x.
   x-only fit far better (P: 1.73 vs 4.24; depth error 0.60 vs 3.91). **It was a software mirror.**

> **Correction (journal 07):** the mechanism below is described the wrong way round. The recorder detected on a mirrored
> frame, so MediaPipe normally said "Left" and wobbled to "Right" on P/Q. The repair here is still correct, but the
> "declared hand" fix was not; see journal 07 for the real root cause.

**Root cause:** the recorder trusted MediaPipe's left/right label on every frame and mirrored "Left" hands. For downward-pointing
hands (P, Q) that label is unreliable: in session 1 it called my right hand "Left" on every P/Q frame, so all 300 samples were
mirrored into a different shape. (The same weakness had already been handled for the public data in journal 03, but not in the recorder.)

## Fix
- `features.resolve_handedness(model_label, setting)`: a declared hand (`--hand right|left`, default right) overrides MediaPipe's guess.
  `features.majority_handedness(labels)`: for the video translator, vote over the whole video instead of per frame.
- The recorder now also saves **raw pixel-space landmarks and MediaPipe's label** per sample (`raw`, `model_hand`), so a future
  pipeline bug can be fixed by recomputing features instead of re-recording.
- **Data repair**: session 1's P/Q rows were un-mirrored (mirroring is an exact x-flip of the stored features, so this is lossless).
  Original kept in `data/recorded_backup/`; the file carries a `repair_note`. Verified: P/Q chirality now matches session 2,
  centroid distances dropped (P 4.22→1.73, Q 5.60→2.73), every other row byte-identical.
- Tests: 63 pass (3 new).

## Results (test = session 2)
| Trained on | Logistic regression | MLP (before fix) | **MLP (after fix)** |
|---|---|---|---|
| public | 41.8% | 44.1% | 44.1% (unaffected: never saw session 1) |
| recorded (session 1) | 68.3% → **71.9%** | 58.6% | 60.1% |
| both | 61.1% → 65.6% | 60.5% | **63.4%** |

## What is left, and why (`artifacts/session_consistency.png`)
Drawing my average hand per session shows the remaining 100% swaps are **not** a pipeline problem:
- **R → V**: in session 1 the two fingers are crossed and close; in session 2 they are spread apart, which is a V.
- **C → T**: session 1's C is an almost flat hand with the thumb out; session 2's is properly bent. A C curves *toward the camera*,
  which the 2D view shows poorly, so it is easy to sign it too flat without noticing.
- **Q → P**: the two sessions' Q shapes differ substantially.
- **Z → X**: Z is a motion letter. Each session froze a different moment (hooked vs straight index), so a static model cannot agree with itself.
- **S → M**: both are fists that differ only by thumb position; the averages are nearly identical.

Two more lessons:
- **The simple model beats the MLP on personal data** (72% vs 60%). With 3,900 rows that are near-copies of 26 frozen poses,
  the MLP memorizes those poses; the linear model cannot, so it generalizes better. More varied data should reverse this.
- **Adding the public data helps the MLP (60→63%) but not the simple model.** One extra signer with a different style adds
  as much confusion as information.

## Next
Re-record the inconsistent letters (C, Q, R, Z, and the fist letters) carefully against a real chart, with more movement,
as a third session. Then re-run this comparison.
