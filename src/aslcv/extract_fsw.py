"""ChicagoFSWild: turn the dataset's frame sequences into hand-landmark sequences (local use only).

    python -m aslcv.extract_fsw --frames data/chicagofswild/frames --workers 6

Dataset: B. Shi et al., "American Sign Language fingerspelling recognition in the wild", IEEE SLT 2018
(7,304 sequences, 160 signers, signer-disjoint train/dev/test). The frames come from online videos of real
people, so neither frames nor extracted landmarks are ever committed to the repository (data/chicagofswild/
is git-ignored); only code and scores are published.

Per frame: MediaPipe (VIDEO mode, up to 2 hands) -> pick the signing hand -> store its 21 pixel-space points.
Signing hand: detections are linked over time into tracks (hand A, hand B), and one track is chosen per clip:
the one matching the dataset's signing-hand boxes (available for a subset), else the one held highest on
average (journal 11). Frames where only the other hand was seen become "no hand" (see select_signing_track).
"""
import argparse
import csv
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

from .features import WRIST

ROOT = Path(__file__).resolve().parents[2]
FSW = ROOT / "data" / "chicagofswild"
VOCAB = "abcdefghijklmnopqrstuvwxyz '"


def clean_label(label):
    """Keep letters, space and apostrophe (drops the 9 rare . & @ characters); tidy spaces."""
    return " ".join("".join(c for c in label.lower() if c in VOCAB).split())


def _box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _hand_box(h):
    px = h.landmarks_px
    return (px[:, 0].min(), px[:, 1].min(), px[:, 0].max(), px[:, 1].max())


def _hand_size(h):
    return max(float(np.linalg.norm(h.landmarks_px[9, :2] - h.landmarks_px[WRIST, :2])), 1.0)


def select_signing_track(frames_hands, frames_boxes=None, max_jump=2.5, min_share=0.3):
    """Pick the signing hand for a whole clip, then return it per frame (None where it was not seen).

    1. Link detections over time into tracks: each hand joins the track whose last wrist is nearest
       (within max_jump hand sizes), else it starts a new track.
    2. Choose one track for the clip: the one overlapping the dataset's signing-hand boxes most often;
       without boxes, the one held highest on average, among tracks present in at least min_share of
       the frames of the longest track (so a brief false detection cannot win).
    A frame where only the other hand was detected becomes None instead of silently switching hands.
    """
    tracks = []                                   # each: {"wrist": xy, "size": s, "hands": {frame: Hand}}
    for t, hands in enumerate(frames_hands):
        free = list(hands or [])
        for tr in sorted(tracks, key=lambda tr: -max(tr["hands"])):      # most recently seen first
            if not free:
                break
            d = [float(np.linalg.norm(h.landmarks_px[WRIST, :2] - tr["wrist"])) / tr["size"] for h in free]
            j = int(np.argmin(d))
            if d[j] <= max_jump and t not in tr["hands"]:
                h = free.pop(j)
                tr["hands"][t], tr["wrist"], tr["size"] = h, h.landmarks_px[WRIST, :2], _hand_size(h)
        for h in free:
            tracks.append({"wrist": h.landmarks_px[WRIST, :2], "size": _hand_size(h), "hands": {t: h}})
    if not tracks:
        return [None] * len(frames_hands)

    best = None
    if frames_boxes and any(frames_boxes):
        def box_votes(tr):
            return sum(1 for t, h in tr["hands"].items()
                       if frames_boxes[t] and max(_box_iou(_hand_box(h), b) for b in frames_boxes[t]) > 0)
        top = max(tracks, key=box_votes)
        if box_votes(top) > 0:
            best = top
    if best is None:
        longest = max(len(tr["hands"]) for tr in tracks)
        candidates = [tr for tr in tracks if len(tr["hands"]) >= max(1, min_share * longest)]
        best = min(candidates, key=lambda tr: np.mean([np.mean(h.landmarks[:, 1]) for h in tr["hands"].values()]))
    return [best["hands"].get(t) for t in range(len(frames_hands))]


def read_boxes(bbox_dir, filename, frame_no):
    """Signing-hand boxes (L=1) for one frame, or None if this frame is not annotated."""
    f = Path(bbox_dir) / filename / f"{frame_no:04d}.txt"
    if not f.exists():
        return None
    boxes = []
    for line in f.read_text().split("\n"):
        vals = [float(v) for v in line.replace(",", " ").split()]
        if len(vals) == 5 and int(vals[4]) == 1:
            boxes.append(tuple(vals[:4]))
    return boxes or None


def pack_sequences(seqs, labels, **meta):
    """Ragged sequences -> flat arrays: frames (N,21,3) with NaN for 'no hand', offsets, labels, metadata."""
    lengths = np.array([len(s) for s in seqs])
    frames = np.full((int(lengths.sum()), 21, 3), np.nan, np.float32)
    i = 0
    for s in seqs:
        for f in s:
            if f is not None:
                frames[i] = f
            i += 1
    out = {"frames": frames, "offsets": np.concatenate([[0], np.cumsum(lengths)]), "labels": np.array(labels)}
    out.update({k: np.array(v) for k, v in meta.items()})
    return out


def unpack_sequence(packed, i):
    a, b = int(packed["offsets"][i]), int(packed["offsets"][i + 1])
    return [None if np.isnan(f).any() else f for f in packed["frames"][a:b]]


def letter_accuracy(refs, hyps):
    """The dataset's standard measure: 1 - (total edits) / (total reference letters)."""
    from .simulate import char_error_rate
    edits = sum(char_error_rate(r, h) * len(r) for r, h in zip(refs, hyps))
    return 1.0 - edits / sum(len(r) for r in refs)


# ---- worker side ----
def _extract_one(args):
    seq_dir, filename, bbox_dir = args
    from .landmarks import HandDetector
    import cv2
    frames = sorted(Path(seq_dir).glob("*.jpg"))
    det = HandDetector(max_hands=2, mode="video")
    all_hands, all_boxes = [], []
    try:
        for k, f in enumerate(frames):
            img = cv2.imread(str(f))
            all_hands.append(det.detect(img, t_ms=k * 33.3) if img is not None else [])
            all_boxes.append(read_boxes(bbox_dir, filename, int(f.stem)) if bbox_dir and f.stem.isdigit() else None)
    finally:
        det.close()
    chosen = select_signing_track(all_hands, all_boxes)
    out = [None if h is None else h.landmarks_px.astype(np.float32) for h in chosen]
    labels = ["" if h is None else h.handedness for h in chosen]
    return out, labels


def extract(frames_root, csv_path, bbox_dir, partitions, workers, limit=None):
    rows = [r for r in csv.DictReader(open(csv_path)) if r["partition"] in partitions]
    if limit:
        rows = rows[:limit]
    jobs = [(Path(frames_root) / r["filename"], r["filename"], bbox_dir) for r in rows]
    missing = [j[0] for j in jobs if not j[0].is_dir()]
    if missing:
        print(f"warning: {len(missing)} sequence folders not found, e.g. {missing[0]}")
    t0, results = time.time(), []
    with ProcessPoolExecutor(workers) as pool:
        for i, res in enumerate(pool.map(_extract_one, jobs, chunksize=4)):
            results.append(res)
            if (i + 1) % 200 == 0 or i + 1 == len(jobs):
                print(f"  {i + 1}/{len(jobs)} sequences, {time.time() - t0:.0f}s", flush=True)
    for part in partitions:
        idx = [i for i, r in enumerate(rows) if r["partition"] == part]
        packed = pack_sequences([results[i][0] for i in idx], [clean_label(rows[i]["label_proc"]) for i in idx],
                                label_raw=[rows[i]["label_proc"] for i in idx], filename=[rows[i]["filename"] for i in idx],
                                signer=[rows[i]["signer"] for i in idx],
                                mp_hand=["".join((l or "-")[0] for l in results[i][1]) for i in idx])
        out = FSW / f"landmarks_{part}.npz"
        np.savez_compressed(out, **packed)
        seen = np.mean(~np.isnan(packed["frames"][:, 0, 0])) if len(packed["frames"]) else 0
        print(f"saved {out.name}: {len(idx)} sequences, {len(packed['frames'])} frames, hand found in {seen:.0%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", default=str(FSW / "frames"), help="folder containing <filename>/<xxxx>.jpg")
    ap.add_argument("--csv", default=str(FSW / "ChicagoFSWild" / "ChicagoFSWild.csv"))
    ap.add_argument("--bbox", default=str(FSW / "ChicagoFSWild" / "BBox"))
    ap.add_argument("--partitions", default="train,dev,test")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    extract(a.frames, a.csv, a.bbox if Path(a.bbox).exists() else None, a.partitions.split(","), a.workers, a.limit)


if __name__ == "__main__":
    main()
