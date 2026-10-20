"""Video translator, sequence engine: the CTC sequence model + spelling-aware decoding (journals 12-14).

The per-frame engine (translate.py) commits letters as frames arrive. The sequence model reads a whole stretch
of signing at once, so this engine works in two passes:

  1. detect up to two hands in every frame and keep them;
  2. split the video into signing segments (a gap of more than max_gap_s without a hand ends one; very long
     segments are cut at their calmest moment, because the model was trained on clips of a few seconds);
  3. per segment: follow the signing hand across the segment, build features, run the model, decode with
     beam search + character model;
  4. render the captioned video and the .srt from the decoded words.

Letter timing inside a segment is approximate (letters are spread evenly); word timing comes from the segments.
The model and the character model are trained on ChicagoFSWild and stay local (git-ignored); without them the
translator falls back to the per-frame engine.
"""
from functools import lru_cache
from pathlib import Path
import time
import cv2
import numpy as np

from .ctc_decode import CharLM, beam_decode
from .extract_fsw import FSW, select_signing_track
from .features import majority_handedness
from .infer import Event, transcript
from .landmarks import HandDetector, draw_hand
from .pipeline import signing_hand, wrist_motion
from .seqmodel import SeqModel, sequence_features
from .subtitles import caption_at, events_to_cues, format_srt

ROOT = Path(__file__).resolve().parents[2]
SEQ_MODEL = ROOT / "models" / "seq_fsw_slowaug2.npz"
ALPHA, BETA = 0.3, 1.5                      # chosen on ChicagoFSWild dev (journal 14)


def find_segments(present, fps, max_gap_s=0.33, min_len_s=0.15, max_len_s=10.0, motion=None):
    """-> [(start, end)] frame ranges (end exclusive) of continuous signing."""
    max_gap, min_len, max_len = round(max_gap_s * fps), round(min_len_s * fps), round(max_len_s * fps)
    segs, start, last = [], None, None
    for i, p in enumerate(present):
        if not p:
            continue
        if start is None:
            start = i
        elif i - last - 1 > max_gap:
            segs.append((start, last + 1))
            start = i
        last = i
    if start is not None:
        segs.append((start, last + 1))
    out = []
    for a, b in segs:
        while b - a > max_len:                       # cut long stretches at the calmest frame near the limit
            lo = max(a + 1, a + max_len - round(2 * fps))
            window = range(lo, a + max_len)
            cut = min(window, key=lambda i: motion[i] if motion is not None else 0.0) + 1
            out.append((a, cut))
            a = cut
        out.append((a, b))
    return [(a, b) for a, b in out if b - a >= max(1, min_len)]


def text_to_events(text, t0, t1):
    """Decoded text for one segment -> letter/space events spread evenly over [t0, t1]; the segment ends a word."""
    words = [w for w in text.split(" ") if w]
    n = sum(len(w) for w in words)
    events, k = [], 0
    for wi, w in enumerate(words):
        for c in w:
            t = t0 + (k + 0.5) / n * (t1 - t0)
            events.append(Event("letter", c.upper(), t, t, 1.0))
            k += 1
        if wi < len(words) - 1:
            t = t0 + k / n * (t1 - t0)
            events.append(Event("space", " ", t, t))
    if events:
        events.append(Event("space", " ", t1, t1))
    return events


@lru_cache(maxsize=1)
def load_char_lm():
    """Character model: ChicagoFSWild training labels (if present locally) + the system English word list."""
    texts = []
    train = FSW / "landmarks_train.npz"
    if train.exists():
        texts += [str(l) for l in np.load(train)["labels"] if str(l)] * 3
    words = Path("/usr/share/dict/words")
    if words.exists():
        texts += [w.strip() for w in words.read_text(errors="ignore").split("\n")
                  if w.strip() and set(w.strip()) <= set("abcdefghijklmnopqrstuvwxyz")]
    return CharLM(texts, order=5) if texts else None


def translate_sequence(video_path, out_video=None, out_srt=None, model_path=SEQ_MODEL, hand="auto", skeleton=True,
                       max_fps=30.0, use_lm=True, verbose=True):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(src_fps / max_fps))
    fps = src_fps / step
    t0 = time.time()

    # pass 1: hands per frame
    det = HandDetector(max_hands=2, mode="video")
    hands, times, idx = [], [], -1
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx % step:
                continue
            t = idx / src_fps
            hands.append(det.detect(frame, t_ms=t * 1000))
            times.append(t)
    finally:
        det.close()
        cap.release()

    # motion of the raised hand, used only to choose where to cut very long segments
    motion, prev = [], None
    for hs in hands:
        h = signing_hand(hs) if hs else None
        motion.append(wrist_motion(prev, h.landmarks_px) if h is not None and prev is not None else 0.0)
        prev = h.landmarks_px if h is not None else None

    segments = find_segments([bool(h) for h in hands], fps, motion=motion)
    chosen = [None] * len(hands)
    for a, b in segments:
        chosen[a:b] = select_signing_track(hands[a:b])
    if hand == "auto":
        side = majority_handedness([h.handedness for h in chosen if h is not None])
    else:
        side = hand

    model = SeqModel.load(model_path)
    lm = load_char_lm() if use_lm else None
    events, decoded = [], []
    for a, b in segments:
        F = sequence_features([None if h is None else h.landmarks_px for h in chosen[a:b]], side)
        text = beam_decode(model.logits(F), model.vocab, lm=lm, alpha=ALPHA, beta=BETA) if lm else model.decode(F)
        decoded.append({"start_s": round(times[a], 2), "end_s": round(times[b - 1], 2), "text": text})
        events += text_to_events(text, times[a], times[b - 1])
        if verbose:
            print(f"  {times[a]:6.1f}-{times[b - 1]:6.1f}s  {text}", flush=True)

    cues = events_to_cues(events)
    if out_srt:
        Path(out_srt).parent.mkdir(parents=True, exist_ok=True)
        Path(out_srt).write_text(format_srt(cues))

    # pass 2: render
    if out_video:
        from .translate import _chars_for, _finalize_video, _overlay
        cap = cv2.VideoCapture(str(video_path))
        tmp = Path(out_video).with_suffix(".tmp.mp4")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        writer, idx, k = None, -1, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx % step:
                continue
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            if k < len(chosen) and skeleton and chosen[k] is not None:
                draw_hand(frame, chosen[k])
            t = times[k] if k < len(times) else idx / src_fps
            _overlay(frame, caption_at(events, t, max_chars=_chars_for(frame)), None, 0.0,
                     status="signing" if k < len(chosen) and chosen[k] is not None else "no hand")
            writer.write(frame)
            k += 1
        cap.release()
        if writer is not None:
            writer.release()
            _finalize_video(tmp, Path(out_video))

    elapsed = time.time() - t0
    return {"engine": "sequence", "text": transcript(events), "cues": cues, "events": events, "segments": decoded,
            "video_fps": round(src_fps, 2), "processed_fps": round(fps, 2), "frames_processed": len(hands),
            "hand_detected_share": round(float(np.mean([bool(h) for h in hands])) if hands else 0.0, 3),
            "handedness_used": side, "seconds": round(elapsed, 1),
            "speed_fps": round(len(hands) / max(elapsed, 1e-6), 1), "decoder": "beam+LM" if lm else "greedy"}
