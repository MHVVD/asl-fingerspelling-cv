"""Real-world benchmark (journal 18): the whole translator on seven public-domain ASL videos (benchmark/README.md).

    python -m aslcv.benchmark hands                    # pass 1 (hand detection) once per video, cached
    python -m aslcv.benchmark score --split tune       # decode + score; options change only the decode settings
    python -m aslcv.benchmark score --split test --name final     # the test videos: run once

Scores, per video and in total, against benchmark/key.json and the English text (benchmark/<video>.tsv):
  * terms read: of the terms the key marks "spelled", how many appear in the translator's letters within
    [t-2, t+8] s of the caption (best-matching substring, at most a quarter of the letters wrong);
  * terms spoken: how many are among the words --speak would say (exact word, same window);
  * spoken words correct: a spoken word is correct if it is in a caption that starts within [t-10, t+2] s;
  * letters share: how much of the video (with a hand visible) is read as letters at all.
"""
import argparse
import json
import re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "benchmark"
CLIPS = ROOT / "clips" / "benchmark"
CACHE = ROOT / "outputs" / "benchmark"
WINDOW = (-2.0, 8.0)          # signing follows the caption by a few seconds
CAPTION_WINDOW = (-10.0, 2.0)  # a spoken word should match a caption shown shortly before it
MAX_ERR = 0.25


def load_key():
    return json.loads((BENCH / "key.json").read_text())


def load_captions(video):
    rows = []
    for line in (BENCH / f"{video}.tsv").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        t, text = line.split("\t", 1)
        rows.append((float(t), re.findall(r"[a-z]+", text.lower())))
    return rows


def substring_distance(term, text):
    """Fewest edits turning `term` into any substring of `text` (Sellers' algorithm)."""
    prev = [0] * (len(text) + 1)
    for i, c in enumerate(term, 1):
        cur = [i] + [0] * len(text)
        for j, d in enumerate(text, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (c != d))
        prev = cur
    return min(prev)


def same_word(a, b):
    return a == b or a.rstrip("s") == b.rstrip("s")


def score_video(video, terms, captions, segments, spoken, frames_letters):
    spelled = [x for x in terms if x["verdict"] == "spelled"]
    read = said = 0
    found = []
    for x in spelled:
        lo, hi = x["t"] + WINDOW[0], x["t"] + WINDOW[1]
        text = "".join(s["text"].replace(" ", "") for s in segments if s["end_s"] >= lo and s["start_s"] <= hi)
        ok_read = bool(text) and substring_distance(x["word"], text) <= MAX_ERR * len(x["word"])
        ok_said = any(same_word(w["text"], x["word"]) and lo <= w["start_s"] <= hi for w in spoken)
        read += ok_read
        said += ok_said
        found.append({**x, "read": ok_read, "spoken": ok_said})
    correct = []
    for w in spoken:
        words = {c for t, ws in captions if w["start_s"] + CAPTION_WINDOW[0] <= t <= w["start_s"] + CAPTION_WINDOW[1]
                 for c in ws}
        correct.append(any(same_word(w["text"], c) for c in words))
    return {"terms_spelled": len(spelled), "terms_read": read, "terms_spoken": said,
            "spoken": len(spoken), "spoken_correct": int(sum(correct)),
            "spoken_words": [{"t": w["start_s"], "word": w["text"], "correct": c} for w, c in zip(spoken, correct)],
            "terms": found, "letters_share": round(frames_letters, 3)}


def decoded_segments(video, hp, detect, use_lm):
    """Decode once per video and decode setting, saved as soon as it is done, so an interrupted run resumes."""
    from .seq_translate import decode_pass
    path = CACHE / "decoded" / f"{video}__det{int(detect)}_lm{int(use_lm)}.json"
    if path.exists():
        return json.loads(path.read_text())
    segments = decode_pass(hp, detect=detect, use_lm=use_lm)["segments"]
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(".part")
    part.write_text(json.dumps(segments))
    part.replace(path)
    print(f"  decoded {video}", flush=True)
    return segments


def run(split, name, detect=True, min_confidence=0.6, vocab_file=None, use_lm=True):
    from .seq_translate import load_hands
    from .spotting import load_vocab
    from .translate import speech_segments
    key = load_key()
    vocab = load_vocab(vocab_file)
    out = {"split": split, "settings": {"detect": detect, "min_confidence": min_confidence, "vocab": vocab_file,
                                        "lm": use_lm}, "videos": {}}
    for video in key["split"][split]:
        hp = load_hands(CACHE / "hands" / f"{video}.npz")
        segments = decoded_segments(video, hp, detect, use_lm)
        spoken, _ = speech_segments({"segments": segments}, "sequence", min_confidence, vocab)
        in_seg = np.zeros(len(hp["times"]), bool)
        times = np.array(hp["times"])
        for s in segments:
            in_seg[(times >= s["start_s"]) & (times <= s["end_s"])] = True
        with_hand = np.array([bool(h) for h in hp["hands"]])
        share = float(in_seg[with_hand].mean()) if with_hand.any() else 0.0
        out["videos"][video] = score_video(video, key["videos"][video], load_captions(video), segments, spoken, share)
        out["videos"][video]["segments"] = segments
    tot = {k: sum(v[k] for v in out["videos"].values())
           for k in ("terms_spelled", "terms_read", "terms_spoken", "spoken", "spoken_correct")}
    out["total"] = tot
    path = ROOT / "artifacts" / f"benchmark_{split}_{name}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"\n{split} / {name}  {out['settings']}")
    print(f"{'video':16s} {'read':>9s} {'spoken':>9s} {'correct':>10s} {'letters':>8s}")
    for v, r in out["videos"].items():
        print(f"{v:16s} {r['terms_read']:>3d}/{r['terms_spelled']:<5d} {r['terms_spoken']:>3d}/{r['terms_spelled']:<5d}"
              f" {r['spoken_correct']:>4d}/{r['spoken']:<5d} {r['letters_share']:>7.0%}")
    print(f"{'total':16s} {tot['terms_read']:>3d}/{tot['terms_spelled']:<5d} {tot['terms_spoken']:>3d}/{tot['terms_spelled']:<5d}"
          f" {tot['spoken_correct']:>4d}/{tot['spoken']:<5d}")
    print(f"saved {path}")
    return out


def cmd_hands():
    from .seq_translate import hand_pass, save_hands
    (CACHE / "hands").mkdir(parents=True, exist_ok=True)
    for video in load_key()["videos"]:
        path = CACHE / "hands" / f"{video}.npz"
        if path.exists():
            continue
        print(f"  {video} ...", flush=True)
        save_hands(hand_pass(CLIPS / f"{video}.webm"), path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["hands", "score"])
    ap.add_argument("--split", choices=["tune", "test"], default="tune")
    ap.add_argument("--name", default="current")
    ap.add_argument("--no-detect", action="store_true")
    ap.add_argument("--min-confidence", type=float, default=0.6)
    ap.add_argument("--vocab")
    a = ap.parse_args()
    if a.command == "hands":
        cmd_hands()
    else:
        run(a.split, a.name, not a.no_detect, a.min_confidence, a.vocab)


if __name__ == "__main__":
    main()
