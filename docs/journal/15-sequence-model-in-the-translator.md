# 15 — The Sequence Model in the Video Translator

## Goal
Make the app use the best model: the CTC sequence model with spelling-aware decoding (70% on 36 unseen ChicagoFSWild signers)
instead of the per-frame model (25% on the same signers).

## Design (`seq_translate.py`, `translate.py --engine`)
The per-frame engine commits letters as frames arrive; the sequence model needs a whole stretch of signing, so the sequence
engine works in two passes:
1. **Detect** up to two hands in every frame and keep them.
2. **Segment** (`find_segments`): a gap of more than 0.33 s without a hand ends a segment; segments shorter than 0.15 s are
   dropped; segments longer than 10 s are cut at the calmest frame (lowest wrist motion) in their last 2 s, because the model
   was trained on clips of a few seconds.
3. **Decode** each segment: follow the signing hand across the segment (`select_signing_track`), build features, run the model,
   beam search + character model.
4. **Render** the captioned video and `.srt`. Letters are spread evenly inside their segment (`text_to_events`); word timing
   comes from the segments.

`--engine auto` (default) uses the sequence engine when its model exists locally, otherwise falls back to the per-frame engine
with a message. The sequence model and the character model are built from ChicagoFSWild, so they are not published; a fresh
clone of the repository runs the per-frame engine. Tests: `tests/test_seq_translate.py` (segmenting, events).

## A decoder bug found on real video: "campyl obacter"
Translating the whole USDA video produced "campyl obacter" every time. Diagnosis:
- The model puts a little probability on "space" around the l → o transition. Greedy decoding ignores it; beam search, which
  sums over many alignments, outputs the space even **without** the character model.
- The length bonus β (1.5 per symbol) also rewarded spaces. The character model rated a space after "campyl" at 1%
  (α·log 0.01 ≈ −1.4), and β = +1.5 more than cancelled it: inserting a space was effectively free.
- My fresh-clip scoring stripped spaces before comparing, so it never showed.

Fix: spaces get their own bonus `beta_space` (default 0). Test added. Re-tuned on dev:

| Dev signers | Letter acc. | Spaces output (196 in the labels) |
|---|---|---|
| Spaces shared the letter bonus | 70.0% | 53 |
| **Separate space bonus 0 (α 0.3, β 1.5)** | 69.9% | 17 |

Same accuracy, far fewer stray spaces; chosen on dev. ChicagoFSWild test with the final setting: **70.3%** (70.6% before; a
second look at test for this decoder, taken after the choice). Doubles found on test with the final setting: 17 / 111 (18 / 111 with the earlier shared bonus).

## On the full USDA video (3.6 min; not a score)
`python -m aslcv.translate clips/usda_campylobacter.webm` — 6,501 frames, hand seen in 92%, 271 s on the laptop CPU
(24 frames/s, a little slower than real time because of the two passes).

It finds the fingerspelled technical vocabulary throughout: campylobacter (6 of 7 times as one word), jejuni,
campylobacteriosis, meningit(is), urinary tract, reactive, guillain, syndro(me), FSIS, website, Ask Karen, USDA.

Limits it makes obvious:
- **It transcribes everything as letters.** The interpreter mostly uses lexical signs, which come out as junk letter strings.
  A real-world translator needs a step that detects *when* someone is fingerspelling.
- **Words run together inside a segment** ("campylobacterbacteria…"): the model rarely outputs spaces, so word breaks come
  only from pauses.
- Letter timing within a segment is approximate.

## Next
Fingerspelling detection (separating fingerspelled segments from other signing), or the sign-vocabulary phase (journal 14's
question about words).
