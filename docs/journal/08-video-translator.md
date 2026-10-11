# 08 — The Video Translator (Mode 2)

## Goal
Give it any fingerspelling video; get back a captioned video and an `.srt` subtitle file. This is the project's real-world test.

## What was done
- `src/aslcv/subtitles.py`: events → one cue per word (`events_to_cues`), SRT formatting, and the running caption for the video (`caption_at`). 5 tests.
- `src/aslcv/translate.py`: the translator CLI. `python -m aslcv.translate VIDEO`.
- `src/aslcv/record_clip.py`: records webcam test clips into `clips/` (mirrored window, unmirrored file).
- `LetterStream.for_fps`: thresholds in **seconds** (hold 0.2 s, gap 0.33 s, dropout 0.1 s), converted to frames per video. 1 test.
- `HandDetector.detect(t_ms=...)`: real video timestamps for MediaPipe's VIDEO mode.
- **Final model** `models/letter_mlp_final.npz`: trained on public + all three personal sessions (only public block b5 held out:
  95.0% there, the same as before adding personal data, so the personal data did not hurt the public signer).
- 88 tests pass.

## How it works
```
video ─▶ every k-th frame (≤ 30 fps) ─▶ MediaPipe VIDEO mode (real timestamps, tracks between frames)
      ─▶ handedness: majority vote over the video ─▶ normalize ─▶ NumPy MLP ─▶ LetterStream (seconds-based)
      ─▶ events ─▶ cues ─▶ .srt
                 └─▶ overlay per frame: running caption (bottom), current letter + confidence bar (top right), skeleton ─▶ .mp4
```
- **Lightweight:** videos above 30 fps are subsampled; a 30 fps clip runs at ~50 frames/s on the i5-8265U, faster than real time.
- **Subtitles:** a word cue starts when its first letter started forming and ends at the space that closed it (the last word lingers 1 s).
  Cues never overlap.
- **Output video** is written by OpenCV, then re-encoded to H.264 with ffmpeg so it plays in browsers and on phones.
- **Any signer:** `--hand auto` (default) votes over all frames; `--hand right/left` if known. `--hold` can be lowered for fast signers.

## Verification
**Mechanics (verified):** H.264 MP4 and SRT written, overlay correct (`ffprobe`: h264, 30 fps; frames inspected).

**Recognition on a synthetic clip (not a valid test):** I built "HELLO WORLD" from held-out public photos (block b5, 20 frames per letter).
Output: `LLO WO D`. The cause was that MediaPipe found the hand in only 46% of frames, so H, E, R and part of WORLD were never seen and a
mid-word detection gap became a space. Two hypotheses were tested:
1. *The hand fills the frame* (as in journal 03), so frame it smaller: **rejected**. Pasting the photo into a larger frame made it worse (25%):
   the hard-edged box with a cut-off arm confuses the palm detector.
2. *VIDEO-mode tracking loses the hand*: **rejected**. VIDEO mode found more hands than IMAGE mode on the same frames (45% vs 29%).

Conclusion: upscaled, re-compressed dataset stills are simply poorly detectable; the translator design is fine (journal 06's simulation
on real recorded features already validates the recognition chain). **The real test is a real webcam clip**, hence `record_clip.py`.

## Known risks for real-world video
- A detection dropout longer than 0.33 s inside a word becomes a space (seen in the synthetic clip).
- Fluent fingerspelling can be faster than the 0.2 s hold.
- Weak letters (C, R, S/M/N/T, Z) from journal 07 still apply.

## Next
Record real clips, translate them, and measure: the share of frames with a hand detected, and the character error rate against what was signed.
