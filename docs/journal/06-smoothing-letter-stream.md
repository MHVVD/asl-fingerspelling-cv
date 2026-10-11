# 06 — Smoothing: From Noisy Frames to Letters (Stage 6)

## Goal
The model guesses a letter on every frame, and the guesses flicker, especially while the hand moves between letters.
Both modes need clean output: the tutor needs a steady "what are you signing now", and the video translator needs
a sequence like `H E L L O` with timestamps.

## What was done
- `src/aslcv/infer.py`: `LetterStream`, a pure-logic state machine (probabilities in, events out) and `transcript()`.
- `src/aslcv/pipeline.py`: `FramePipeline`, shared by both modes: frame → MediaPipe → handedness decision → features → model
  → probabilities, plus wrist motion. `HandednessVote` (running majority) for videos of unknown signers.
- `src/aslcv/simulate.py`: spells words using *real* recorded frames from a held-out session to test the whole chain without a video.
- Tests: `tests/test_infer.py` (12), `tests/test_pipeline.py` (4). Total 79 pass.

## How the letter stream works
1. **Smooth.** Exponential moving average of the probability vectors: `s = 0.4·new + 0.6·s`. A one-frame glitch moves `s` only a little.
2. **Commit.** A letter is emitted once it has been the top smoothed letter with confidence ≥ 0.7 for 6 frames in a row (0.2 s at 30 fps).
3. **Double letters.** After emitting L, another L needs a *release* first: L's smoothed confidence dips below 0.45 (the hand changed),
   the wrist jumps by more than 0.15 hand-sizes in one frame (the small bounce signers use for double letters), or the hand
   briefly disappears. Without this, HELLO would come out HELO, or HELLLLLO if repeats were allowed freely.
4. **Spaces.** No hand for 10 frames (~0.33 s) ends a word: exactly one space, never leading or doubled. A 1–2 frame detector
   miss is ignored. After a word break the smoothing memory is cleared so the old letter does not leak into the new word.

**Motion** is measured in hand sizes (wrist movement ÷ wrist-to-middle-knuckle distance), so it means the same near or far from the camera.

**Handedness in videos:** a stranger's video has no `--hand` setting, so `FramePipeline(hand="auto")` takes a running majority of
MediaPipe's labels. One wrong per-frame label (the P/Q bug from journal 05) can no longer mirror a frame.

## Results: simulated spelling with real frames
Model `letter_mlp_both_test_s2` (63.4% single-frame accuracy on session 2), frames from session 2, 20 frames per letter,
6 blended transition frames between letters, a bounce between double letters, 20 no-hand frames between words.

| Word | Output | Character error rate |
|---|---|---|
| HELLO | HELLO | 0.00 |
| BOOK | BOOK | 0.00 |
| LOVE | LOVE | 0.00 |
| COFFEE | TOFFEE | 0.17 |
| THANK YOU | HAK YOU | 0.22 |
| YES | YEM | 0.33 |
| SIGN LANGUAGE | MIG LAGAGE | 0.31 |
| WORLD | WOUL | 0.40 |
| NO | O | 0.50 |

**Mean character error rate: 0.22** (5 seeds × 9 words). Double letters (LL, OO, FF, EE) are handled correctly.

**Threshold search** (enter confidence 0.5–0.8 × hold 4–10 frames): the error rate stays at 0.22–0.23 almost everywhere
(0.28 only for the loosest setting). **The bottleneck is the classifier, not the smoothing.** Every remaining error is a known weak
letter from journal 05: C→T, R→U, S→M, and N/T/D *dropped* because the model is never confident enough on them to commit.
Defaults kept: enter 0.7, hold 6.

## Caveats
- Simulated transitions are linear blends; real hand transitions pass through other shapes that might briefly look like a third letter.
  The video translator will show how much that matters.
- **Speed:** a 6-frame hold is 0.2 s at 30 fps. Fluent signers can fingerspell faster than that, so real-world video will need a
  shorter hold (or frame-rate-aware settings), at the cost of more false letters. To be measured on real clips.

## Next
Video translator (subtitles) and tutor, both on top of `FramePipeline` + `LetterStream`. Accuracy gains now have to come from data
(the third recording session) and the confusable letters.
