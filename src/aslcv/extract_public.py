"""Stage 4b: turn the public "ASL Alphabet" image dataset into landmark features.

    python -m aslcv.extract_public                      # full run, ~87k images
    python -m aslcv.extract_public --limit 50           # quick trial: 50 images per letter

Input:  data/raw/ (the Kaggle download; letter folders A..Z anywhere below it).
Output: data/public/asl_alphabet.npz       X (n,63), y, session, files, padded   (same format as recordings)
        artifacts/extraction_report.json   per-letter detection rates, handedness counts, timing
        artifacts/extraction_samples.png   one detected example per letter, with landmarks drawn

Every image goes through the SAME path as live webcam frames (MediaPipe -> pixel space ->
normalize_landmarks), so public and recorded samples live in one feature space.
"""
import argparse
import json
import re
import string
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import cv2
import numpy as np

from .features import normalize_landmarks

ROOT = Path(__file__).resolve().parents[2]
LETTERS = set(string.ascii_uppercase)
_NUM = re.compile(r"(\d+)")


def discover_images(root, limit_per_class=None):
    """Find (path, letter, frame_number) for images inside folders named A..Z.
    Skips del/nothing/space folders and the flat test folder (A_test.jpg etc.)."""
    by_label = defaultdict(list)
    for p in Path(root).rglob("*"):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png") or p.parent.name not in LETTERS:
            continue
        m = _NUM.search(p.stem)
        by_label[p.parent.name].append((p, p.parent.name, int(m.group(1)) if m else 0))
    items = []
    for label in sorted(by_label):
        files = sorted(by_label[label], key=lambda t: t[2])
        items.extend(files[:limit_per_class] if limit_per_class else files)
    return items


def session_for(frame_number, block):
    """The images in each class are numbered frames, presumably cut from a continuous recording,
    so neighbours are near-duplicates. Grouping consecutive frames into blocks gives
    'pseudo-sessions' that can be split train/test without leakage."""
    return f"asl_alphabet:b{(frame_number - 1) // block}"


def detect_with_fallback(detector, img, pad):
    """Try the image as-is; if no hand, add a black border and retry.
    Hands that fill the whole frame are hard for the palm detector; a border helps.
    Returns (Hand | None, used_padding)."""
    hands = detector.detect(img)
    if hands:
        return hands[0], False
    if pad > 0:
        p = int(round(img.shape[0] * pad))
        hands = detector.detect(cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_CONSTANT, value=(0, 0, 0)))
        if hands:
            return hands[0], True
    return None, False


# ---- worker process side: each process owns one MediaPipe detector ----
_detector = None


def _init_worker(min_conf):
    global _detector
    from .landmarks import HandDetector
    _detector = HandDetector(max_hands=1, min_conf=min_conf, mode="image")


def _process(args):
    path, pad, handedness = args
    img = cv2.imread(str(path))
    if img is None:
        return None
    hand, padded = detect_with_fallback(_detector, img, pad)
    if hand is None:
        return None
    side = hand.handedness if handedness == "auto" else handedness
    feats = normalize_landmarks(hand.landmarks_px, side).astype(np.float32)
    return feats, hand.handedness, padded, hand.landmarks


def extract(source, out_path, limit=None, workers=6, pad=0.25, block=500, min_conf=0.5, handedness="Right"):
    items = discover_images(source, limit)
    if not items:
        raise SystemExit(f"No letter folders (A..Z) with images found under {source}")
    print(f"{len(items)} images in {len({l for _, l, _ in items})} letters, {workers} workers")

    X, y, sessions, files, padded_flags = [], [], [], [], []
    stats = defaultdict(Counter)
    model_hand = Counter()
    samples = {}
    t0 = time.time()
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(min_conf,)) as pool:
        jobs = ((p, pad, handedness) for p, _, _ in items)
        for i, ((path, label, num), res) in enumerate(zip(items, pool.map(_process, jobs, chunksize=64))):
            stats[label]["total"] += 1
            if res is not None:
                feats, mp_hand, padded, lm = res
                stats[label]["detected"] += 1
                stats[label]["padded"] += int(padded)
                model_hand[mp_hand] += 1
                X.append(feats); y.append(label); sessions.append(session_for(num, block))
                files.append(str(path.relative_to(source))); padded_flags.append(padded)
                samples.setdefault(label, (path, lm, padded))
            if (i + 1) % 2000 == 0 or i + 1 == len(items):
                rate = (i + 1) / (time.time() - t0)
                print(f"  {i + 1}/{len(items)}  {rate:.0f} img/s  eta {(len(items) - i - 1) / rate:.0f}s", flush=True)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, X=np.stack(X), y=np.array(y), session=np.array(sessions),
             files=np.array(files), padded=np.array(padded_flags))

    total = sum(s["total"] for s in stats.values())
    found = sum(s["detected"] for s in stats.values())
    report = {
        "source": _rel(source), "output": _rel(out_path),
        "settings": dict(limit=limit, pad=pad, block=block, min_conf=min_conf, handedness=handedness),
        "images": total, "detected": found, "detection_rate": round(found / total, 4),
        "rescued_by_padding": sum(s["padded"] for s in stats.values()),
        "mediapipe_handedness_labels": dict(model_hand),
        "seconds": round(time.time() - t0, 1),
        "per_letter": {k: dict(total=v["total"], detected=v["detected"], padded=v["padded"],
                               rate=round(v["detected"] / v["total"], 3)) for k, v in sorted(stats.items())},
    }
    art = ROOT / "artifacts"
    art.mkdir(exist_ok=True)
    (art / "extraction_report.json").write_text(json.dumps(report, indent=2))
    cv2.imwrite(str(art / "extraction_samples.png"), _sample_sheet(samples, pad))
    return report


def _rel(path):
    """Paths relative to the project, so reports don't contain machine-specific locations."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _sample_sheet(samples, pad, tile=160, cols=9):
    """One example per letter: the image (padded if that's how it was detected) with landmarks drawn."""
    from .landmarks import CONNECTIONS
    letters = sorted(samples)
    rows = -(-len(letters) // cols)
    sheet = np.full((rows * tile, cols * tile, 3), 30, np.uint8)
    for i, letter in enumerate(letters):
        path, lm, padded = samples[letter]
        img = cv2.imread(str(path))
        if padded:
            p = int(round(img.shape[0] * pad))
            img = cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        h, w = img.shape[:2]
        xy = (lm[:, :2] * [w, h]).astype(int)
        for a, b in CONNECTIONS:
            cv2.line(img, tuple(xy[a]), tuple(xy[b]), (0, 255, 0), 2)
        for pt in xy:
            cv2.circle(img, tuple(pt), 3, (0, 0, 255), -1)
        img = cv2.resize(img, (tile, tile))
        cv2.putText(img, letter + (" (pad)" if padded else ""), (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 200, 255), 2)
        r, c = divmod(i, cols)
        sheet[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = img
    return sheet


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=ROOT / "data" / "raw", type=Path)
    ap.add_argument("--out", default=ROOT / "data" / "public" / "asl_alphabet.npz", type=Path)
    ap.add_argument("--limit", type=int, help="max images per letter (for a quick trial)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--pad", type=float, default=0.25, help="border fraction for the retry (0 disables)")
    ap.add_argument("--block", type=int, default=500, help="frames per pseudo-session")
    ap.add_argument("--min-conf", type=float, default=0.5)
    ap.add_argument("--handedness", default="Right", choices=["Right", "Left", "auto"],
                    help="single-signer dataset: force one hand (default) instead of trusting per-image labels")
    a = ap.parse_args()
    r = extract(a.source, a.out, a.limit, a.workers, a.pad, a.block, a.min_conf, a.handedness)
    print(f"\ndetected {r['detected']}/{r['images']} ({r['detection_rate']:.1%}), "
          f"{r['rescued_by_padding']} rescued by padding, {r['seconds']}s")
    worst = sorted(r["per_letter"].items(), key=lambda kv: kv[1]["rate"])[:5]
    print("lowest detection:", ", ".join(f"{k} {v['rate']:.0%}" for k, v in worst))
    print("saved", r["output"])


if __name__ == "__main__":
    main()
