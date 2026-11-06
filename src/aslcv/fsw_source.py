"""Fingerspelling detection data (journal 17): the full source videos behind ChicagoFSWild.

ChicagoFSWild only ships the fingerspelled clips, but its CSV gives each clip's source URL and start time. Between
the marked clips is ordinary signing by the same people, with the same cameras, which is what a detector needs as
"not fingerspelling". This follows Shi et al. (CVPR 2021), who built their detection benchmark the same way.

    python -m aslcv.fsw_source list        # which source videos are still public (metadata only)
    python -m aslcv.fsw_source download    # those videos, max 480p, into data/fsw_source/videos/
    python -m aslcv.fsw_source check       # do the downloads line up with the annotations? (frame matching)
    python -m aslcv.fsw_source extract     # hand landmarks at 15 fps + a fingerspelling label per frame

Only YouTube sources still reachable are used (deafvideo.tv refuses downloads with 403; some videos are private
now). Videos whose clips span more than one split are skipped, so test signers never leak into training. Videos and
everything derived from them stay local (data/ is git-ignored).
"""
import argparse
import collections
import csv
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import cv2
import numpy as np

from .extract_fsw import FSW

CSV = FSW / "ChicagoFSWild" / "ChicagoFSWild.csv"
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "fsw_source"
VIDEOS = SRC / "videos"
AVAILABLE = SRC / "available.txt"           # url|duration_s, written by `list`
LANDMARKS = SRC / "landmarks"
YTDLP = [sys.executable, "-m", "yt_dlp"]
from .fsdetect import FPS                    # detection runs at 15 fps (plenty for spotting fingerspelling)
MARGIN_S = 5.0                              # signing kept before the first / after the last marked clip


def parse_time(s):
    h, m, x = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(x)


def video_id(url):
    return url.split("v=")[-1].split("&")[0]


def load_sources(csv_path=CSV):
    """url -> {"partition", "clips": [{"filename", "start_s", "frames", "label", "signer"}]} for single-split videos."""
    by = collections.defaultdict(list)
    for r in csv.DictReader(open(csv_path)):
        by[r["url"]].append(r)
    out = {}
    for url, rows in by.items():
        parts = {r["partition"] for r in rows}
        if len(parts) != 1 or "youtube.com" not in url:
            continue
        out[url] = {"partition": parts.pop(),
                    "clips": sorted(({"filename": r["filename"], "start_s": parse_time(r["start_time"]),
                                      "frames": int(r["number_of_frames"]), "label": r["label_proc"],
                                      "signer": r["signer"]} for r in rows), key=lambda c: c["start_s"])}
    return out


def available():
    if not AVAILABLE.exists():
        return {}
    return {l.split("|")[0]: float(l.split("|")[1] or 0) for l in AVAILABLE.read_text().splitlines() if l.strip()}


def cmd_list():
    sources = load_sources()
    SRC.mkdir(parents=True, exist_ok=True)
    urls = SRC / "urls.txt"
    urls.write_text("\n".join(sources))
    r = subprocess.run(YTDLP + ["--no-warnings", "--ignore-errors", "--skip-download", "--print",
                                "%(webpage_url)s|%(duration)s", "-a", str(urls)], capture_output=True, text=True)
    AVAILABLE.write_text(r.stdout)
    print(f"{len(available())} of {len(sources)} single-split YouTube sources are public -> {AVAILABLE}")


def cmd_download():
    sources, av = load_sources(), available()
    VIDEOS.mkdir(parents=True, exist_ok=True)
    todo = [u for u in sources if u in av and video_path(u) is None]
    print(f"{len(todo)} videos to download")
    for i, url in enumerate(todo, 1):
        r = subprocess.run(YTDLP + ["-q", "--no-warnings", "--no-playlist", "-f", "bv*[height<=480]+ba/b[height<=480]/b",
                                    "--merge-output-format", "mp4", "-o", str(VIDEOS / "%(id)s.%(ext)s"), url],
                           capture_output=True, text=True)
        print(f"  [{i}/{len(todo)}] {video_id(url)} {'ok' if r.returncode == 0 else 'FAILED ' + r.stderr.strip()[-120:]}",
              flush=True)


def video_path(url):
    # finished downloads only: skip .part files and yt-dlp's per-stream fragments ("<id>.f140.m4a") left by a failed merge
    hits = [h for h in VIDEOS.glob(f"{video_id(url)}.*")
            if not h.name.endswith(".part") and not re.fullmatch(r"\.f\d+", h.suffixes[0] if h.suffixes else "")]
    return hits[0] if hits else None


def _small_gray(img):
    g = cv2.cvtColor(cv2.resize(img, (64, 48), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
    return (g - g.mean()) / (g.std() + 1e-6)


def clip_offset(cap, fps, clip, search_s=2.0):
    """Best time shift (s) between a clip's first ChicagoFSWild frame and the downloaded video, and its match score
    (normalised correlation, 1 = identical)."""
    ref = cv2.imread(str(FSW / "frames" / clip["filename"] / "0001.jpg"))
    if ref is None:
        return None, 0.0
    ref = _small_gray(ref)
    t0 = max(0.0, clip["start_s"] - search_s)
    cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000)
    best = (None, -1.0)
    for i in range(int(2 * search_s * fps)):
        ok, f = cap.read()
        if not ok:
            break
        score = float((_small_gray(f) * ref).mean())
        if score > best[1]:
            best = (t0 + i / fps - clip["start_s"], score)
    return best


def cmd_check(per_video=3):
    """Frame matching on a few clips per video: the downloaded video must be the annotated one, unshifted.
    Incremental: results are kept in check.tsv; aligned.txt lists the videos that pass."""
    log = SRC / "check.tsv"
    done = {l.split("\t")[0]: l for l in log.read_text().splitlines()} if log.exists() else {}
    for url, src in load_sources().items():
        vid, path = video_id(url), video_path(url)
        if path is None or vid in done:
            continue
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        picks = src["clips"][:: max(1, len(src["clips"]) // per_video)][:per_video]
        res = [clip_offset(cap, fps, c) for c in picks]
        cap.release()
        offs = [o for o, sc in res if o is not None and sc > 0.6]
        ok = len(offs) == len(res) and abs(float(np.median(offs))) < 0.25
        done[vid] = f"{vid}\t{src['partition']}\t{len(offs)}/{len(res)}\t" \
                    f"{float(np.median(offs)) if offs else float('nan'):+.3f}\t{'ok' if ok else 'MISMATCH'}"
        print("  " + done[vid], flush=True)
    log.write_text("\n".join(done.values()) + "\n")
    good = [l.split("\t")[0] for l in done.values() if l.endswith("ok")]
    (SRC / "aligned.txt").write_text("\n".join(good))
    print(f"{len(good)} of {len(done)} checked videos line up (all checked clips match within 0.25 s)")


def label_frames(times, clips, fps):
    """1 where t falls inside a marked fingerspelling clip [start, start + frames / fps)."""
    y = np.zeros(len(times), np.int8)
    for c in clips:
        y[(times >= c["start_s"]) & (times < c["start_s"] + c["frames"] / fps)] = 1
    return y


def extract_video(url, src):
    """Two hands per frame at FPS over the annotated region (+ margin) -> data/fsw_source/landmarks/<id>.npz."""
    from .landmarks import HandDetector
    out = LANDMARKS / f"{video_id(url)}.npz"
    if out.exists():
        return out.name, "cached"
    cap = cv2.VideoCapture(str(video_path(url)))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap_w, cap_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, round(fps / FPS))
    t_lo = max(0.0, src["clips"][0]["start_s"] - MARGIN_S)
    t_hi = max(c["start_s"] + c["frames"] / fps for c in src["clips"]) + MARGIN_S
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_lo * fps))
    idx = int(t_lo * fps) - 1
    det = HandDetector(max_hands=2, mode="video")
    times, pts, sides = [], [], []
    try:
        while True:
            ok, frame = cap.read()
            idx += 1
            if not ok or idx / fps > t_hi:
                break
            if idx % step:
                continue
            t = idx / fps
            hands = det.detect(frame, t_ms=t * 1000)
            p = np.full((2, 21, 3), np.nan, np.float32)
            sd = ["", ""]
            for k, h in enumerate(hands[:2]):
                p[k], sd[k] = h.landmarks_px, h.handedness
            times.append(t), pts.append(p), sides.append(sd)
    finally:
        det.close()
        cap.release()
    times = np.array(times)
    LANDMARKS.mkdir(parents=True, exist_ok=True)
    part = out.with_suffix(".part.npz")
    np.savez_compressed(part, times=times, hands=np.array(pts), handedness=np.array(sides),
                        label=label_frames(times, src["clips"], fps), partition=src["partition"], fps=fps,
                        size=np.array([cap_w, cap_h]),
                        signers=np.array(sorted({c["signer"] for c in src["clips"]})))
    part.replace(out)
    return out.name, f"{len(times)} frames, {label_frames(times, src['clips'], fps).mean():.0%} fingerspelling"


def cmd_extract(workers=4):
    aligned = set((SRC / "aligned.txt").read_text().split()) if (SRC / "aligned.txt").exists() else set()
    todo = [(u, s) for u, s in load_sources().items() if video_id(u) in aligned]
    print(f"{len(todo)} aligned videos")
    with ProcessPoolExecutor(workers) as ex:
        for i, (name, msg) in enumerate(ex.map(extract_video, *zip(*todo)), 1):
            print(f"  [{i}/{len(todo)}] {name}: {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["list", "download", "check", "extract"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.command == "extract":
        cmd_extract(a.workers)
    else:
        {"list": cmd_list, "download": cmd_download, "check": cmd_check}[a.command]()


if __name__ == "__main__":
    main()
