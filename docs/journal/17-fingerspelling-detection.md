# 17 — Fingerspelling Detection: Which Moments Are Spelling?

## Goal
The letter models read **all** signing as letters (journal 15). On real video most signing is ordinary signs, which
come out as junk ("reveidisemittar") in the captions and, since journal 16, in the narration. A detector should mark
*when* someone is fingerspelling, so that only those moments are read as letters.

## Data: the videos behind ChicagoFSWild
ChicagoFSWild ships only the fingerspelled clips, but its CSV gives every clip's source URL and start time. Between
the marked clips is ordinary signing by the same people, with the same cameras: ideal "not fingerspelling" examples,
with no risk of the detector learning "which dataset is this" instead of "is this fingerspelling". Shi et al.
(CVPR 2021) built their detection benchmark the same way.

`fsw_source.py`:
1. **`list`:** 194 source videos, 106 on YouTube and 87 on deafvideo.tv.
   - deafvideo.tv refuses downloads (HTTP 403), so it isn't used.
   - 89 YouTube videos are still public; the rest are now private.
   - Videos whose clips span more than one split are skipped (7), so test signers stay unseen.
2. **`download`:** max 480p with `yt-dlp`, local only (`data/fsw_source/` is git-ignored).
   - About 1–2 minutes per video.
   - 10 downloads failed with HTTP 503 on the first try; retries recovered all but one.
3. **`check`:** does the download match the annotation? For 3 clips per video, ChicagoFSWild's first frame is
   matched against the video around the clip's start time.
   - Matching videos agree at 0.999 similarity within one frame (offset ≈ 0.002 s).
   - 7 videos don't match at all, probably re-cut since the dataset was built, and are excluded.
4. **`extract`:** both hands at 15 fps over each video's annotated region (±5 s), with a 0/1 label per frame
   from the clip timestamps. 32 frames/s per worker; about 2 hours for everything with 3–4 workers.

| Split | Videos | Signers | Frames (15 fps) | Fingerspelling |
|---|---|---|---|---|
| train | 49 | 36 | 284,754 (5.3 h) | 12.2% |
| dev | 14 | 13 | 39,927 (0.7 h) | 12.3% |
| test | 7 | 8 | 20,778 (0.4 h) | 11.2% |

**Is every fingerspelled word in these videos marked?** If annotators missed some, unmarked stretches would teach
"fingerspelling is not fingerspelling". Check: read 2-second unmarked windows with the letter model and look for
real words of 6+ letters.
- In 20 videos, 36 of 755 marked clips (4.8%) produce one, while 0 of 1,359 unmarked windows do.
- At that rate, even 5% of missed fingerspelling would show up as about 3 words.
- This is supporting evidence, not proof.

## Model (`fsdetect.py`, `fsdetect_train.py`)
- **Per frame**, for the upper (more raised) and the lower hand:
  - whether it's present
  - its 63-number shape (wrist-centred, scaled, left hands mirrored)
  - wrist position as a share of the frame
  - hand size relative to the frame height
  - wrist velocity in hand sizes per frame
  - plus the lower hand's position relative to the upper one: 140 numbers in all.
- **Network:** Linear+ReLU → BiGRU (2 layers, 64 units) → one output per frame, 134k parameters.
  - The NumPy BiGRU forward pass is now shared with the letter model.
- **Training:** random 12.8 s windows, weighted loss (fingerspelling is 12% of frames) and mirror augmentation.
  3,000 steps, 19 minutes on the laptop CPU; the best dev checkpoint is kept. Dev AP peaks around step 1,000–1,500,
  after which the model starts to overfit.
- **Segments:** probabilities smoothed over 5 frames and thresholded (0.7, chosen on dev). Gaps ≤ 0.3 s are bridged
  and runs < 0.3 s dropped.

## Results
**AP** (average precision) ranks frames by probability; 1.0 is perfect, and guessing scores the fingerspelling share
(~0.12). A clip is **found** if a detected segment overlaps it with IoU ≥ 0.5; a detected segment is **correct** if it
overlaps some clip with IoU ≥ 0.5.

| Test: 8 unseen signers, 162 clips (scored once) | AP | Precision | Recall | F1 | Clips found | Segments correct |
|---|---|---|---|---|---|---|
| "A hand is visible" (what the translator did before) | 0.12 | 12% | 100% | 0.21 | 0% | 0% |
| Per-frame classifier, same features, no time context | 0.68 | 58% | 70% | 0.63 | 47% | 51% |
| **BiGRU detector** | **0.82** | **73%** | **78%** | **0.76** | **59%** | **72%** |

- Dev was close: 0.86 AP, F1 0.77.
- Context over time is worth +0.14 AP on test.
- The test set is small (7 videos, 8 signers), so treat these as rough numbers.
- The IoU-based counts are strict: back-to-back fingerspelled words are often detected as one stretch, which then
  overlaps each single clip by less than half.

## In the translator
`translate.py` (sequence engine) now runs the detector at 15 fps before decoding and reads only the marked stretches,
padded by 0.2 s. This is on by default when the local model exists; `--no-detect` turns it off. The detector adds
little time, since hand detection is still the slow part.

On the USDA video (a different signer, camera and setting, used neither for training nor tuning; spoken words checked
against the captions burned into the video):

| USDA video | Frames read as letters | Segments that are one known word | Spoken (dictionary only) | Correct | Wrong |
|---|---|---|---|---|---|
| Detector off | 92% | 5 (mostly junk: "scull", "ps", "pm") | 10 | 7 | 2 + 1 unclear |
| **Detector on** | **48%** | **12** | 18 | **12** + 1 close | 4 + 1 unclear |

- **Clean single-word segments now:** bacterial, strain, intestines, feces, antibiotics, and "campylobacteriosis"
  twice. Before, "campylobacteriosis" was only ever read run together with other letters.
- **New finds:** meningitis and syndrome, which were missed before.
- **Wrong:** "coon", "bus" (U.S.), "purina" (urinary) and "sisters" (FSIS).
  - The detector cuts signing into shorter pieces, so a short piece that happens to spell a dictionary word gets
    spoken.
  - I haven't tuned the spotting rules on this video, because its numbers would then flatter the result.
- **Not fixed:** the opening and closing stretches are still junk ("ustacoosify", "asl vidomcat").
- Record: `artifacts/fsdetect_usda.json`.

## Pitfalls
- **Fragment files.** A download that fails during merging leaves `yt-dlp` fragments (`<id>.f140.m4a`, audio only).
  These were first taken for videos and "failed" the alignment check (2 of the 9 mismatches). Fragments are now
  ignored.
- **Buffered progress lines.** Progress printed through a pipe was buffered until the end; it's now flushed.
- **Data licence and terms.** The source videos are other people's public uploads, used locally for research, as
  ChicagoFSWild itself is. Videos, landmarks and the trained detector stay local, like the ChicagoFSWild models;
  only code and summary numbers are committed.

## Next
- **Spotting rule:** require a minimum length for short whole-segment words, chosen on a held-out video rather than
  on USDA.
- **Opening and closing junk:** a stricter threshold, or a minimum confidence from the letter model inside detected
  stretches.
- **The detector as a building block:** the next phase, recognising whole signs, needs exactly the stretches it
  marks as *not* fingerspelling.
