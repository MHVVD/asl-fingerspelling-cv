# 10 — The Tutor (Mode 1)

## Goal
Teach the alphabet live: show a target letter, watch the hand, say whether it's right and *what to change*, and focus
practice on weak letters. Also turn practice into consistent training data, which is the project's main remaining problem (journal 09).

## What was done
- `src/aslcv/hints.py`: measures the hand's shape and turns differences into instructions. `python -m aslcv.hints` builds
  `models/finger_states.json` from real data.
- `src/aslcv/tutor.py`: `Planner` (spaced repetition, progress in `data/progress.json`) and `Attempt` (hold-to-confirm, timeout).
- `src/aslcv/tutor_app.py`: the app. `TutorApp.step(frame, t)` renders one screen, so it can be driven by a video file for testing;
  `main()` only adds the webcam and keys.
- `render_panel(footer=...)`: each app shows its own key hints.
- Tests: `tests/test_tutor.py` (12). 100 tests pass.

## How it works
**One attempt.** A target letter is shown with its reference hand. The *smoothed* prediction (Stage 6) must be the target with
confidence ≥ 0.6, continuously for 0.6 s (the green bar fills). 12 s without success = missed; the answer stays on screen for 2.5 s.

**Spaced repetition.** Mastery per letter = (correct + 1) / (attempts + 2), so a new letter starts at 0.5. The next letter is drawn
with weight (1 − mastery)² + 0.02: weak letters return often, mastered ones occasionally; never twice in a row. `--letters` restricts the set.

**Tips (after 2 s without progress, refreshed every 0.5 s, max 2 + one classifier tip):**
| Measure | How | Tip |
|---|---|---|
| Finger bend | total angle along each finger (knuckle + 2 joints); <70° straight, >130° curled | "Straighten your index finger" / "Curl your ring finger down" |
| Spread | angle between index and middle; <12° together, >22° apart | "Spread your index and middle fingers apart" / "…together" |
| Thumb | thumb tip to index knuckle in hand sizes; <0.45 in, >0.65 out | "Stick your thumb out" / "Tuck your thumb in" |
| Palm direction | wrist → middle knuckle | only "sideways" / "down" (see below) |
| Classifier | top prediction ≠ target with confidence ≥ 0.6 | "That looks like V" |

**Expected shapes come from real hands, not the drawn references.** Measured on the reference poses, my hand and the public signer:
straight fingers read <60° and curled >160° everywhere, but the references over-curl (240°) and get X wrong (a real hooked index
measures ~60°, the reference 163°). So `build_table` learns, per letter, the states that ≥80% of real samples (public + mine) agree
on; a tip is only given for those, and only for clear mistakes (straight vs curled).

## Calibration on real data
| Check | Result |
|---|---|
| Correct signs that still get a tip (false alarms), mean over letters | **2.4%** (worst: P 14%, J 7%) |
| Target L, showing B | tip 99% ("Curl your middle finger down", "…ring…") |
| Target W, showing V | tip 100% ("Straighten your ring finger") |
| Target U, showing V | tip 41% ("Hold your index and middle fingers together") |
| Target V, showing U | tip 2%: my V spreads vary too much for 80% agreement; the classifier tip covers it |

Two problems found and fixed during calibration:
- G got a misleading tip on 24% of *correct* signs: the direction measure follows the palm, and in G the palm points up while the
  index points sideways. "Point up" tips were removed (G: 24% → 5%).
- U vs V got no tip at all, because they differ by spread rather than bend. Added the spread measure.

## Teach mode (ENTER)
If you sign a letter correctly but the model disagrees, ENTER saves the next 60 hand frames as examples of the target, into
`data/recorded/practice-<time>.npz` (with raw landmarks), which the next training run picks up automatically.
Caution: it saves whatever you show, so only use it when you're sure of the sign. Delete a practice file to undo.

## Verification
Driven by a video file instead of the webcam (`TutorApp.step`), screens inspected: target + hold bar + timer; stacked tips
("Straighten your middle finger" / "That looks like O" while showing O for target H); debug view (top-3 predictions, finger bends,
thumb, spread, palm direction, hand side, motion); teach banner and 60 saved frames; a missed attempt advancing to the next letter.
Bugs fixed from the screenshots: tips ran off-screen (now stacked), the panel showed the recorder's keys, and the result message
appeared one frame late.

Not yet verified: a live session with the webcam (needs the user).
