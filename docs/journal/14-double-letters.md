# 14 — Double Letters

## The problem, measured (dev signers, current best model `seq_fsw_slowaug2`)
- 13% of dev clips (130 / 981) contain a double letter; there are 136 doubles in total.
- **Greedy CTC reproduced 6 of 136 (4%).** "mall" → "mal", "hell" → "hel", "goole" → "gole".
- Accuracy on clips with doubles 58.0% vs 68.5% without. Getting every double right would add only about +1.7 points
  overall, but a missing double makes a word visibly wrong.

Cause: CTC merges identical neighbouring outputs unless a blank separates them. Fast signers show a double as a single hold
with a slight bounce or slide, so the model rarely emits that blank.

## Two fixes, chosen on dev
**Fix 1 — a "repeat" symbol** (`seqmodel.encode_repeats`): train on "mall" → `mal+`, so a double can't be merged away.
594 of 5,455 training labels contain one. Retrained (30 epochs, same settings).

**Fix 2 — spelling-aware decoding** (`ctc_decode.py`): CTC prefix beam search (beam 16) fused with a 5-gram character
model (Witten-Bell), trained on training labels (×3) plus the system English word list; never on dev/test labels.
`score = log P_ctc + α·log P_lm + β·length`.

| Dev signers | Letter acc. | Doubles found | Doubles invented |
|---|---|---|---|
| Current model, greedy | 66.6% | 6 / 136 | 13 |
| Fix 1, greedy | 66.6% | 9 / 136 | 41 |
| Fix 1 + fix 2 (best) | 69.8% | 33 / 136 | 57 |
| **Fix 2 on the current model, α 0.3, β 1.5** | **70.0%** | 31 / 136 | 37 |

**Fix 1 did not work:** a few more doubles, many more invented ones. The visual cue for a double seems too subtle to learn
from 594 examples. Its code and tests stay; the model is not used.
Raising β finds more doubles but invents even more; β 1.5 is the accuracy optimum.

## Final check (run once)
| | Greedy | Beam + character model |
|---|---|---|
| **ChicagoFSWild test, 36 unseen signers** | 66.6% | **70.6%** |
| Doubles found (test) | 2 / 111 | 18 / 111 |
| Fresh clips (journal 13, 25 letters) | 88% | 84% |

On the large test the decoder adds +4.0 points, matching dev. On the fresh clips it lost one letter: BOOK went from "BEK" to
**"BED"**, because the character model knows "bed" is a common word and pulled the last letter towards it. On 25 letters this is
within noise, but it is a real side effect: a spelling model can push towards *wrong* common words, which matters most for names.

## What is still wrong
About 84% of double letters are still missed, including the O-O in BOOK. Better cues would need either more examples of
doubles or features that capture the small bounce/slide signers use (e.g. finger-level motion, not only wrist motion).

## Decision
Use beam search + character model (α 0.3, β 1.5) as the default decoder for the sequence model in the video translator; keep
greedy available. Records: `artifacts/double_letters_dev.json`, `artifacts/double_letters_final.json`.
