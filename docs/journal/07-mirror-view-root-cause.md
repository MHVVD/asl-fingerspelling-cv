# 07 — The Real Root Cause: Detecting on a Mirrored Frame

## What happened
Session 3 (`20261011-015015`, re-recordings of C, M, N, Q, R, S, T, Z) came out as an exact **mirror image** of sessions 1–2,
and MediaPipe labeled all 1,200 frames "Left", although the signer used the right hand, palm to camera.
A new live hand check in the recorder showed "camera sees a LEFT hand" for a right palm.

## Investigation
1. **Hypothesis: the camera image got flipped** (driver setting changed by another app). Disproved: reading the V4L2 controls
   of the HP Wide Vision HD camera directly shows it has no flip control at all, and no other camera app was running.
   I had told the user the camera was flipped before checking this; that was wrong.
2. **Re-reading all the evidence with one consistent rule** explained everything:
   - The recorder mirrored each webcam frame **before** detection, so the window felt like a mirror.
   - **MediaPipe's label describes the hand as it appears in the image it receives.** In a mirrored frame a physical right hand
     looks like a left hand, so it was labeled "Left", in every session.
   - The *original* recorder mirrored "Left" hands, which accidentally flipped them back to the correct orientation.
     Sessions 1–2 were right by luck; only P/Q broke, where the label wobbled to "Right".
   - My journal-05 "fix" (declared hand, never mirror) removed that accidental correction while the frame was still mirrored,
     so session 3 was stored backwards. The new hand check compared a mirrored image's label with the real hand, hence the false alarm.
   - The public dataset agrees: its unmirrored photos of a right hand were labeled "Right" (95%).
3. **Test of the rule** (`test_mediapipe_label_follows_the_unmirrored_image`): on real dataset photos, flipping the image flips
   MediaPipe's label in >90% of cases. So the label follows the image, not the person.

**Correction to journal 05:** the mechanism there was described the wrong way round. In the mirrored frames MediaPipe normally said
"Left" and said "Right" on P/Q. The effect and the repair (un-mirroring session 1's P/Q) are unchanged and still correct.

## Fix
- **Detection now always runs on the camera's real, unmirrored image**, exactly like the public photos and any video file.
  Only the on-screen view is mirrored (`landmarks.mirror_hand` for drawing). Now `--hand right` means the signer's right hand everywhere,
  and there is no double mirroring left to cancel out.
- The live hand check (`record.HandMismatch`) now compares like with like, so a red bar means a genuinely wrong hand or the back of the hand.
- **Data repair, session 3**: X flipped back to the correct orientation; raw pixel x converted from mirrored-frame to real-image coordinates
  (`640 − x`). Backup in `data/recorded_backup/`, note in `repair_note`. Verified: chirality now matches all other data,
  and features recomputed from the corrected raw points with the new pipeline equal the repaired X exactly.
- Sessions 1–2 need no change: their two mirrorings cancelled out, giving exactly what the new pipeline produces.

| Letter | Session 3 vs sessions 1–2 average, before repair | after |
|---|---|---|
| M / N / R / S / T / Z | 3.51 / 3.27 / 2.49 / 3.95 / 3.31 / 2.51 | **0.77 / 0.81 / 0.68 / 0.47 / 1.10 / 1.31** |
| C / Q | 4.59 / 5.73 | 3.03 / 8.15 (deliberately re-signed differently: C now curves toward the camera) |

## Results with three sessions (test = session 2)
| Trained on | Logistic regression | MLP | Before session 3 (best) |
|---|---|---|---|
| recorded (sessions 1 + 3) | 73.3% | 70.3% | 71.9% |
| **both** (public + sessions 1 + 3) | 69.0% | **76.9%** | 63.4% |

Simulated spelling (journal 06 method): mean character error rate **0.22 → 0.19**; THANK YOU now comes out THAK YOU.

**The test set is part of the problem now.** Session 2 contains the inconsistent versions: its R is signed with spread fingers (a V)
and its C is flat. The model trained on the corrected session 3 is *penalized* for getting them right (R→V 75–100%, C→T/D).
A clean, full fourth session is needed as an honest test set.

## Lessons
- A "fix" that makes one symptom go away can be wrong if the explanation is wrong. The journal-05 diagnosis found the right
  symptom (P/Q mirrored) with the wrong mechanism, and the fix built on that mechanism created a new bug.
- Mirroring must happen in exactly one place, for display only. Never let two mirrorings cancel by accident.
- Saving raw landmarks paid off immediately: session 3 could be verified and repaired without re-recording.
