# 11 — First Real-World Test: A Stranger Fingerspelling the Alphabet

## The clip
- **"American Sign Language demo – Travis Dougherty"**, Wikimedia Commons, by Fuzheado, 22 March 2014,
  **CC BY-SA 3.0** ([file page](https://commons.wikimedia.org/wiki/File:American_Sign_Language_demo_-_Travis_Dougherty.ogv)).
  Saved as `clips/asl_alphabet_travis.ogv` (sha256 `9130e2a5…3bcc3ef`). 45.5 s, 960×540, 29.97 fps.
- **Content (checked frame by frame):** 0–15 s an introduction in full ASL (both hands, not fingerspelling);
  ~16.5–44 s the alphabet fingerspelled with one hand at shoulder height, about one letter per second (spot-checked: A, B, C at
  18–20 s; F, G, H at 23–25 s; L at 28 s; O, P, Q at 31–33 s; V at 37 s; Y at 39 s; Z traced at 42–43 s); hands down at 45 s.
- **Answer key** for the scored window 16–45.5 s: `ABCDEFGHIJKLMNOPQRSTUVWXYZ`. The introduction has no correct fingerspelled
  output, so it is reported but not scored.
- Why it is a real test: a signer the model has never seen, a different camera and room, and his hand is small in the frame
  (~100 px tall at 540p, far smaller than in any training data).

## Results
| Run | Change | Output for the alphabet | Letters right | Error rate |
|---|---|---|---|---|
| 1 | translator as built (journal 08) | `ZZSZSZSZSSJIO` | **1 / 26** | 0.96 |
| 2 | detect both hands, read the raised one | `OIRTHZIARLONONNRSZTUVWZYZACLC` | **12 / 26** | 0.73 |

Run 2, letter by letter (`truth>output`, `-` = nothing):
```
A>-  B>-  C>O  D>I  E>R  F>T  G>H  H>Z  I  J>A  K>R  L  M>O  N  O  P>N  Q>N  R  S  ->Z  T  U  V  W  X>Z  Y  Z  ->A ->C ->L ->C
```
Right: I, L, N, O, R, S, T, U, V, W, Y, Z. Records: `artifacts/realworld_travis_run1.json`, `artifacts/realworld_travis_run2.json`;
captioned videos and subtitles in `outputs/` (not in git).

## Run 1: what went wrong
A hand was found in 98.5% of frames, yet only 1 letter was right. The captioned video showed why: **the skeleton was on Travis's
resting hand at his waist, not the hand doing the spelling.** The detector was set to find one hand; MediaPipe locked onto the
resting one and kept tracking it, so for 30 seconds the model read a relaxed fist (mostly "S" and "Z").
This never showed up in my own recordings, where only one hand was ever raised.

## The fix
- `pipeline.signing_hand(hands)`: with two hands in view, use the one whose landmarks sit **highest** in the image.
  Fingerspelling is done around shoulder height; the other hand usually rests lower.
- The translator now detects up to two hands (`HandDetector(max_hands=2)`).
- Test: `test_signing_hand_is_the_raised_one_not_the_resting_one` (resting hand listed first, raised hand must win). 101 tests pass.
- Verified on the output video: the skeleton follows the spelling hand in every checked frame (19–43 s).

## Other observations
- **Handedness vote said "Left".** The spelling hand looks like a left hand in this recording (he spells left-handed, or the
  video was recorded mirrored). The automatic vote (journal 06) handled it: the hand is mirrored into the same orientation as
  the training data either way.
- **Speed:** 33 frames/s with two-hand detection (55 with one), still faster than real time on the laptop CPU.
- **Remaining errors:** the first eight letters (A–H) are almost all wrong; P/Q (pointing down) read as N; J and X wrong;
  four extra letters after Z as the hand drops; the introduction produces junk letters (expected: it is not fingerspelling).

## Demo video
`python -m aslcv.make_demo` builds `outputs/demo_realworld_before_after.mp4` (1080×1080, 39 s): a title card, then the
fingerspelled part with run 1 (BEFORE, wrong hand) and run 2 (AFTER, signing hand) side by side, each output letter coloured
green if it matches the alphabet and red if not, then a results card. The CC BY-SA credit is on screen throughout; as an
adaptation, the demo video is itself CC BY-SA 3.0.

## What I did not do
I stopped tuning after the root-cause fix. Adjusting thresholds until this one clip scores well would make it a training clip,
not a test. Further improvements will be judged on new clips.

## Takeaways
1. A tracker that follows *a* hand is not the same as following *the signing* hand. Real video has two hands.
2. Generalization to a stranger is the hard part: 95% on the dataset signer, 73% on my own unseen session, 46% of letters on
   a stranger's real video.
3. The full chain (video → hand → letter → smoothing → subtitles) works end to end on real footage, faster than real time.

## Next
- More training variety: other signers, smaller/farther hands (augment by scale is impossible since features are scale-normalized,
  so it must come from detection on small hands, i.e. real data), downward letters (P, Q).
- Treat letter transitions and the hand dropping out of signing position as "no letter" (a "rest" class), to stop extra letters.
- Evaluate on more real clips, ideally a slice of ChicagoFSWild (160 signers, labelled).
