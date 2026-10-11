"""Stage 6 check: spell words with REAL recorded frames and see what the smoothing produces.

    python -m aslcv.simulate --model models/letter_mlp_both_test_s2.npz --session 20261011-011708

For each word, frames are taken from a held-out recording session: `hold` frames per letter,
`blend` in-between frames (a linear mix of the two hand shapes) to imitate the hand moving between
letters, a small wrist bounce between double letters, and a no-hand gap between words.
This tests the smoothing on the model's real (imperfect) outputs without needing a video.
"""
import argparse
from pathlib import Path
import numpy as np

from .infer import LetterStream, transcript
from .model import LetterModel

ROOT = Path(__file__).resolve().parents[2]
WORDS = ["HELLO", "WORLD", "BOOK", "LOVE", "COFFEE", "YES", "NO", "THANK YOU", "SIGN LANGUAGE"]


def frames_for(text, X, y, rng, hold=20, blend=6, gap=20):
    """Returns a list of (features | None, motion)."""
    pools = {k: X[y == k] for k in set(y)}
    out, prev = [], None
    for i, ch in enumerate(text):
        if ch == " ":
            out += [(None, 0.0)] * gap
            prev = None
            continue
        pool = pools[ch]
        start = rng.integers(0, max(1, len(pool) - hold))
        seq = pool[start:start + hold]                        # consecutive real frames of this letter
        if prev is not None:
            for a in np.linspace(0, 1, blend + 2)[1:-1]:      # in-between hand shapes
                out.append(((1 - a) * prev + a * seq[0], 0.03))
            if i > 0 and text[i - 1] == ch:
                out.append((seq[0], 0.3))                     # double letter: small bounce
        out += [(f, 0.0) for f in seq]
        prev = seq[-1]
    return out + [(None, 0.0)] * gap


def char_error_rate(ref, hyp):
    """Edit distance / length: 0 is perfect."""
    d = np.arange(len(hyp) + 1)
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[len(hyp)] / max(len(ref), 1)


def run(model, X, y, words=WORDS, seed=0, **stream_kw):
    rng = np.random.default_rng(seed)
    rows = []
    for w in words:
        s = LetterStream(model.classes, **stream_kw)
        events = []
        for i, (f, motion) in enumerate(frames_for(w, X, y, rng)):
            events += s.update(None if f is None else model.predict_proba(f), t=i / 30, motion=motion)
        out = transcript(events)
        rows.append((w, out, char_error_rate(w, out)))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(ROOT / "models" / "letter_mlp_both_test_s2.npz"))
    ap.add_argument("--session", default="20261011-011708")
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    model = LetterModel.load(a.model)
    d = np.load(ROOT / "data" / "recorded" / f"{a.session}.npz")
    X, y = d["X"], d["y"]

    print(f"model {Path(a.model).name}, frames from session {a.session}")
    print(f"single-frame accuracy on this session: {np.mean(model.predict(X) == y):.1%}\n")
    all_cer = []
    for seed in range(a.seeds):
        rows = run(model, X, y, seed=seed)
        all_cer += [c for _, _, c in rows]
        if seed == 0:
            for w, out, c in rows:
                print(f"  {w:<14} -> {out:<16} CER {c:.2f}")
    print(f"\nmean character error rate over {a.seeds} seeds x {len(WORDS)} words: {np.mean(all_cer):.2f}")


if __name__ == "__main__":
    main()
