# 13 — A Fresh Real-World Test, Scored Once

## Why
The Travis clip (journals 11–12) had informed several decisions (which hand to read, the speed diagnosis, whether a retraining
helped), so it was no longer an honest test. A fresh test needed clips that no model had seen, with answer keys written
**before** any model was run, and a single scoring run.

## The clips (free licenses)
| Clip | Signer | License | Word(s) scored |
|---|---|---|---|
| [ASL BOOK](https://commons.wikimedia.org/wiki/File:ASL_BOOK.ogv) | Richard Goodrow | CC BY 3.0 | BOOK |
| [(ASL) – Campylobacter 2011](https://commons.wikimedia.org/wiki/File:(ASL)_-_Campylobacter_2011.webm) | USDA Food Safety | Public domain | CAMPYLOBACTER, BACTERIA |

## Building the answer keys (by eye, before scoring)
- BOOK: the file description says "B-O-O-K"; frames at 0.1 s steps show B 2.1–2.6 s, O-O 2.7–2.9 s (one brief hold), K 3.0–3.5 s.
- USDA: the burned-in English captions show which word is being signed; each fingerspelled run was then read at full frame rate.
  CAMPYLOBACTER at 25.9–28.1 s (13 letters in ~2.2 s, about 6 letters per second; A and M too fast to separate by eye, the rest
  clear), and BACTERIA at 29.1–30.8 s.
- Keys, windows, how each was made, and a SHA-256 of each clip are in `artifacts/fresh_test_keys.json`; `score_fresh.py` refuses to
  score a clip whose file changed since its key was written.

## Results (`python -m aslcv.score_fresh`, run once; `artifacts/fresh_test_results.json`)
| Window | Truth | Per-frame (dev-tuned) | Sequence | Sequence + slow signing |
|---|---|---|---|---|
| USDA 25.9–28.1 s | CAMPYLOBACTER | CAOHYLMBACZRX (7/13) | **CAMPYLOBACTER** (13/13) | **CAMPYLOBACTER** (13/13) |
| USDA 29.1–30.8 s | BACTERIA | BACZYAX (3/8) | BACTERIAT (8/8, 1 extra) | BACTERIAT (8/8, 1 extra) |
| BOOK 1.9–3.8 s | BOOK | ACBOZ (1/4) | K (1/4) | BEK (2/4) |
| **Total (25 letters)** | | **40.0%** | **84.0%** | **88.0%** |

Letter accuracy = 1 − total edits / total letters. Per-frame settings: hold 0.10 s, threshold 0.4 (its best, chosen on ChicagoFSWild dev).

## Reading the result
- A fluent signer spelling a 13-letter technical word at ~6 letters per second was transcribed perfectly by both sequence models.
- BOOK is the weak case: a 320×240 video, the hand partly out of frame, and a double O shown as one brief hold. No model got
  the double O; double letters remain the clearest open problem.
- 25 letters from 2 signers is a spot check, not a benchmark. It agrees in direction with ChicagoFSWild (per-frame 25%,
  sequence ~67%), and the ranking of the three models is the same.

## Next
1. Double letters (beam search, possibly a small language model), judged on ChicagoFSWild dev, not on these clips.
2. Put the sequence model into the video translator.
