# 04 — Training a Lightweight Letter Classifier (Stage 5)

## Goal
Train a letter classifier that fits the hardware: a laptop with an Intel i5-8265U (4 cores), 15 GB RAM and a 2 GB GeForce MX130.
It must run in real time on CPU for both the tutor and the video translator.

## Compute budget (measured)
| Resource | Finding | Decision |
|---|---|---|
| GPU | MX130, 2 GB; untested, very likely too old and small to help | CPU only |
| PyTorch | `import torch` takes **21 s** on this machine | Not used at runtime |
| MediaPipe | ~15–25 ms per frame | The real bottleneck; everything else must be negligible |

## What was done
- `src/aslcv/model.py`: `LetterModel`, a pure-NumPy MLP forward pass. It copies weights out of scikit-learn and saves them to `.npz`.
  A test checks it gives *identical* probabilities to scikit-learn, and another checks that importing it does not load torch.
- `src/aslcv/train.py`: session-based split, augmentation, logistic-regression baseline and MLP; writes the model, metrics and confusion matrix.
- `src/aslcv/evaluate.py`: accuracy, macro-F1, per-letter F1, top confusions, confusion-matrix plot, per-frame latency.
- Tests: `tests/test_model.py`, `tests/test_train.py` (59 tests total pass).

## How it works
**The model.** A multilayer perceptron, 63 inputs → 128 → 64 → 26 letters, with ReLU between layers and softmax at the end.
That is 18,138 learned numbers (73 KB). Predicting is three matrix multiplications:
`p = softmax(relu(relu(x·W1+b1)·W2+b2)·W3+b3)`.

**Why train with scikit-learn but predict with NumPy?** Training needs an optimizer (Adam), loss and early stopping, which
scikit-learn provides. Prediction only needs the weights. Copying them out keeps the runtime to NumPy only: instant start-up, about 300 MB RAM.

**The split.** Test = pseudo-session `asl_alphabet:b5` (frames 2501–3000 of every letter). Training never sees those frames.
A random split would put near-identical neighbouring frames on both sides and inflate the score.

**Augmentation.** Each training sample gets one copy rotated in the image plane by up to ±15° around the wrist, plus small
jitter (σ = 0.02). This mimics tilted hands and cameras, which the single-signer dataset barely covers, and helps on real-world video.

**Baseline.** Logistic regression is a linear model. If it does almost as well as the MLP, the features are doing most of the work.

## Results (public data, test on b5)
| Model | Accuracy | Macro-F1 | Train time | Size | Latency |
|---|---|---|---|---|---|
| Logistic regression | 94.1% | – | 34 s | – | – |
| **MLP 128-64** | **95.2%** | **93.6%** | 63 s (34 epochs) | 73 KB, 18k params | **0.08 ms/frame** |

Whole run: 1 min 40 s, 300 MB peak RAM. Model latency is ~200× cheaper than MediaPipe, so it costs essentially nothing live.

Confusion matrix: `artifacts/confusion_public.png`. Weakest letters (F1): A 0.72, M 0.74, R 0.81, T 0.82, U 0.84.
| Mistake | Rate | Why it makes sense |
|---|---|---|
| A → M | 30% | Both are fists; the difference is where the thumb sits (beside vs under the fingers) |
| R → U | 25% | Both are two raised fingers; R crosses them, which looks almost the same in 2D landmarks |
| T → A / W | 9% / 9% | T is a fist with the thumb between fingers, again a thumb-position letter |
| P → Y | 9% | Unexpected; possibly dataset noise (noted earlier that some samples look mislabeled) |

The logistic regression is only ~1 point behind, so the normalized landmark features carry most of the information.
The remaining errors are mostly the thumb-position letters, which is also where MediaPipe detection was weakest (journal 03).

## Honest caveats
- **This is an optimistic number.** Train and test are the *same signer, same background, same camera*. A stranger or
  real-world video will score lower. The real test needs another signer (your recordings) held out entirely.
- **J and Z score 100% here, which is misleading.** In this dataset they are still frames of one moment of the motion.
  Real J/Z are movements and are not handled yet.
- Early stopping holds out a random 10% of the training rows. Those rows are near-duplicates of other training rows, so
  stopping may be slightly late. It does not touch the test set.

## First real-world check: public model on my own hand
Session `20261011-003953` (all 26 letters × 150 samples, recorded with `aslcv.record`), scored by the public-only model:
**50.8% correct**, against 95.2% on the dataset signer. This is the generalization gap.
- Fine (88–100%): B E F I K L O S U V W X Y.
- Failing (0–26%): A C D G H J M N P Q R T Z. Typical swaps: G→Y, H→Z (sideways letters), P→N, Q→C (downward letters), D→X, M→E, N→S, T→S.
- Variety check: average per-feature spread within a letter is **0.015** for this session against **0.147** in the public set. The hand
  was held almost still, so 150 samples are close to 150 copies of one pose. Session 2 will add movement, tilt and distance changes.

Takeaway: a model trained on one person does not transfer to another person, especially for sideways/downward letters and fist letters.
This is why personal (and ideally several people's) recordings are part of the plan.

## Next
Record your sessions, then rerun with `--data recorded` and `--data both` holding out one of your sessions,
for the A/B/C comparison. Then Stage 6: smoothing over frames, shared by the tutor and the video translator.
