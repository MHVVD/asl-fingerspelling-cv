# 09 — A Clean Test Session (Session 4)

## What was done
Recorded session 4 (`20261011-020828`, all 26 letters × 150) with the fixed recorder (detection on the real image, journal 07)
and used it as the held-out test set. All models train on sessions 1–3 (plus public data where stated).

## Checks on the new session
- MediaPipe labeled all 3,900 frames "Right", **including P and Q**, which used to wobble. The journal-07 fix works.
- Variety (average within-letter spread) 0.026, up from 0.015 and 0.020 but still far below the public set (0.147).
- H and X have the opposite 2D knuckle order to earlier sessions. This is **not** mirroring (labels all Right, no software flip):
  for a sideways hand the index and pinky knuckles line up with the wrist, so turning the hand a little further flips the 2D order.
  The hand was turned further this time.
- Letters whose session-4 shape is nearest a *different* letter of sessions 1–3: G→T, L→D, P/Q→T, I↔J, X↔Z, M→N, U→R, V→U.

## Results (test = session 4)
| Trained on | Logistic regression | MLP |
|---|---|---|
| public | 41.4% | 42.4% |
| recorded (sessions 1–3) | **64.1%** | 59.0% |
| **both** | 63.7% | **72.6%** (macro-F1 67.2%) |

Same pattern as journal 07 on session 2: public data alone does not transfer to me, my data alone is too uniform for the MLP,
and the combination is best.

Letters at 0% for the best model: D, G, H, P, X. Swaps: H→R, P→Z, D→X, G→T (all 100%), S→E 87%.
Each is a letter signed differently in session 4 than in sessions 1–3 (turned further, or a different shape). With 3–4 examples
of my own style per letter, the model cannot cover that much session-to-session variation.

## What would help most
1. **Consistency**: the same shape for a letter every time. The tutor (next stage) shows the reference and checks the sign live,
   which also makes it a natural way to collect consistent practice data.
2. **Variety within the shape**: turning, tilting and distance changes while holding the same handshape.
3. **More signers**: the only real cure for generalizing to strangers' videos.
