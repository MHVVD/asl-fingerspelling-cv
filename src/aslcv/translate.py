"""Video translator: subtitle a fingerspelling video.

    python -m aslcv.translate clips/hello.mp4
    python -m aslcv.translate any_video.mp4 --out outputs/demo.mp4 --hand auto --no-skeleton

Outputs (default: outputs/<name>_subtitled.mp4 and outputs/<name>.srt):
  * a captioned video: running text at the bottom, current letter + confidence top-right, hand skeleton
  * an .srt subtitle file, one cue per word, playable alongside the original video in any player

Pipeline per frame: MediaPipe (VIDEO mode, real timestamps) -> handedness by majority vote ->
normalized features -> NumPy MLP -> LetterStream (thresholds in seconds) -> events -> subtitles.
Videos above --max-fps are subsampled to keep it light on a laptop CPU.
"""
import argparse
import shutil
import subprocess
import time
from pathlib import Path
import cv2
import numpy as np

from .infer import LetterStream, transcript
from .landmarks import HandDetector, draw_hand
from .model import LetterModel
from .pipeline import FramePipeline
from .subtitles import caption_at, events_to_cues, format_srt

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "models" / "letter_mlp_final.npz"


def translate(video_path, out_video=None, out_srt=None, model_path=DEFAULT_MODEL, hand="auto",
              skeleton=True, max_fps=30.0, hold_s=0.2, verbose=True):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, round(src_fps / max_fps))          # e.g. 60 fps video -> every 2nd frame
    fps = src_fps / step

    model = LetterModel.load(model_path)
    detector = HandDetector(mode="video", max_hands=2)   # both hands, then read the signing one (journal 11)
    pipe = FramePipeline(detector, model, hand=hand)
    stream = LetterStream.for_fps(model.classes, fps, hold_s=hold_s)

    writer, tmp_video = None, None
    events, frames_with_hand, processed = [], 0, 0
    t0 = time.time()
    idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % step:
            continue
        t = idx / src_fps
        r = pipe.process(frame, t_ms=t * 1000)
        events += stream.update(r.probs, t=t, motion=r.motion)
        processed += 1
        frames_with_hand += r.probs is not None

        if out_video:
            if writer is None:
                tmp_video = Path(out_video).with_suffix(".tmp.mp4")
                tmp_video.parent.mkdir(parents=True, exist_ok=True)
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(tmp_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            if skeleton and r.hand is not None:
                draw_hand(frame, r.hand)
            letter, conf = stream.current
            _overlay(frame, caption_at(events, t, max_chars=_chars_for(frame)), letter, conf)
            writer.write(frame)
        if verbose and processed % 150 == 0:
            done = f"{idx + 1}/{n_frames}" if n_frames else f"{idx + 1}"
            print(f"  frame {done}  '{transcript(events)}'", flush=True)

    cap.release()
    detector.close()
    elapsed = time.time() - t0

    cues = events_to_cues(events)
    if out_srt:
        Path(out_srt).parent.mkdir(parents=True, exist_ok=True)
        Path(out_srt).write_text(format_srt(cues))
    if writer is not None:
        writer.release()
        _finalize_video(tmp_video, Path(out_video))

    return {
        "text": transcript(events), "cues": cues, "events": events,
        "video_fps": round(src_fps, 2), "processed_fps": round(fps, 2), "frames_processed": processed,
        "hand_detected_share": round(frames_with_hand / max(processed, 1), 3),
        "handedness_used": pipe.vote.counts.most_common(1)[0][0] if pipe.vote.counts else hand,
        "seconds": round(elapsed, 1), "speed_fps": round(processed / max(elapsed, 1e-6), 1),
    }


def _chars_for(frame):
    return max(12, frame.shape[1] // 22)


def _overlay(frame, caption, letter, conf):
    h, w = frame.shape[:2]
    s = h / 480                                                   # scale fonts with resolution
    # caption bar
    bar = frame[int(h - 70 * s):h]
    bar[:] = (bar * 0.35).astype(np.uint8)
    cv2.putText(frame, caption or "...", (int(14 * s), int(h - 24 * s)), cv2.FONT_HERSHEY_SIMPLEX,
                1.1 * s, (255, 255, 255), max(1, int(2 * s)), cv2.LINE_AA)
    # current letter + confidence
    x0, y0, bw, bh = int(w - 110 * s), int(10 * s), int(100 * s), int(110 * s)
    box = frame[y0:y0 + bh, x0:x0 + bw]
    box[:] = (box * 0.35).astype(np.uint8)
    if letter:
        cv2.putText(frame, letter, (x0 + int(22 * s), y0 + int(70 * s)), cv2.FONT_HERSHEY_SIMPLEX,
                    2.2 * s, (80, 200, 255), max(1, int(4 * s)), cv2.LINE_AA)
        cv2.rectangle(frame, (x0 + int(8 * s), y0 + int(88 * s)),
                      (x0 + int(8 * s) + int((bw - 16 * s) * conf), y0 + int(98 * s)), (80, 200, 255), -1)
    else:
        cv2.putText(frame, "no hand", (x0 + int(10 * s), y0 + int(60 * s)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55 * s, (180, 180, 180), 1, cv2.LINE_AA)


def _finalize_video(tmp, out):
    """Re-encode to H.264 if ffmpeg is available (plays in browsers/phones); otherwise keep OpenCV's mp4v."""
    if shutil.which("ffmpeg"):
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)])
        if r.returncode == 0:
            tmp.unlink()
            return
    tmp.replace(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--out", help="captioned video path (default outputs/<name>_subtitled.mp4)")
    ap.add_argument("--srt", help="subtitle path (default outputs/<name>.srt)")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--hand", choices=["auto", "right", "left"], default="auto",
                    help="signing hand; auto = majority vote over the video")
    ap.add_argument("--no-skeleton", action="store_true")
    ap.add_argument("--max-fps", type=float, default=30.0)
    ap.add_argument("--hold", type=float, default=0.2, help="seconds a letter must be held (lower for fast signers)")
    a = ap.parse_args()

    stem = Path(a.video).stem
    out = a.out or ROOT / "outputs" / f"{stem}_subtitled.mp4"
    srt = a.srt or ROOT / "outputs" / f"{stem}.srt"
    hand = a.hand if a.hand == "auto" else a.hand.capitalize()
    r = translate(a.video, out, srt, a.model, hand, not a.no_skeleton, a.max_fps, a.hold)
    print(f"\ntranscript: {r['text'] or '(nothing recognised)'}")
    print(f"hand seen in {r['hand_detected_share']:.0%} of frames | hand: {r['handedness_used']} | "
          f"{r['frames_processed']} frames at {r['processed_fps']} fps | {r['seconds']} s ({r['speed_fps']} frames/s)")
    print(f"saved {out}\nsaved {srt}")


if __name__ == "__main__":
    main()
