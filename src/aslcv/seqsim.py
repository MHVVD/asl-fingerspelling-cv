"""Sanity check for the sequence model before real data: words spelled with recorded frames.

    python -m aslcv.seqsim

Train on words built from my sessions 1-3, test on words built from session 4 (never seen), and compare
the CTC sequence model with the per-frame model + LetterStream smoothing on the *same* test sequences.
Transitions are simulated (linear blends between hand shapes), so this checks the pipeline, not real video.
"""
import json
import time
from pathlib import Path
import numpy as np

from .infer import LetterStream, transcript
from .model import LetterModel
from .seqmodel import FEAT_DIM, SeqModel, Vocab
from .simulate import char_error_rate

ROOT = Path(__file__).resolve().parents[2]
LETTERS = "abcdefghijklmnopqrstuvwxyz"
SESSIONS_TRAIN = ["20261011-003953", "20261011-011708", "20261011-015015"]
SESSION_TEST = "20261011-020828"


def load_words(n, rng, min_len=3, max_len=8):
    words = [w.strip() for w in open("/usr/share/dict/words")
             if set(w.strip()) <= set(LETTERS) and min_len <= len(w.strip()) <= max_len]   # plain a-z only
    return list(rng.choice(words, n, replace=False))


def spell(word, pools, rng):
    """-> (T, FEAT_DIM) features: held letters, blended transitions, a bounce between double letters."""
    rows, prev = [], None
    for i, ch in enumerate(word):
        pool = pools[ch.upper()]
        hold = int(rng.integers(6, 18))
        start = int(rng.integers(0, max(1, len(pool) - hold)))
        seq = pool[start:start + hold]
        if prev is not None:
            for a in np.linspace(0, 1, int(rng.integers(3, 9)) + 2)[1:-1]:
                rows.append(((1 - a) * prev + a * seq[0], rng.normal(0, 0.03, 2)))
            if word[i - 1] == ch:
                rows += [(seq[0], np.array([0.35, 0.0])), (seq[0], np.array([-0.35, 0.0]))]
        rows += [(f, rng.normal(0, 0.01, 2)) for f in seq]
        prev = seq[-1]
    F = np.zeros((len(rows), FEAT_DIM), np.float32)
    for t, (f, v) in enumerate(rows):
        F[t, :63], F[t, 63:65], F[t, 65] = f, v, 1.0
    return F


def pools_for(sessions):
    X, y = [], []
    for s in sessions:
        d = np.load(ROOT / "data" / "recorded" / f"{s}.npz")
        X.append(d["X"]); y.append(d["y"])
    X, y = np.concatenate(X), np.concatenate(y)
    return {k: X[y == k] for k in set(y)}


def augment(F, rng, max_rot_deg=15, noise=0.02):
    """Same in-plane rotation for the whole clip (one signer, one camera) + per-frame jitter."""
    F = F.copy()
    a = np.radians(rng.uniform(-max_rot_deg, max_rot_deg))
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]], np.float32)
    pts = F[:, :63].reshape(-1, 21, 3) @ R.T
    pts += rng.normal(0, noise, pts.shape).astype(np.float32)
    pts -= pts[:, :1]
    F[:, :63] = (pts * F[:, 65:66, None]).reshape(-1, 63)   # keep no-hand frames at zero
    return F


def baseline_decode(model, F):
    """Today's system: per-frame MLP + LetterStream, on the same features."""
    s = LetterStream.for_fps(model.classes, 30)
    ev = []
    for t, row in enumerate(F):
        p = model.predict_proba(row[:63]) if row[65] else None
        ev += s.update(p, t=t / 30, motion=float(np.linalg.norm(row[63:65])))
    return transcript(ev).lower().replace(" ", "")


def main():
    import torch
    from .seqtrain import SeqNet, train
    rng = np.random.default_rng(0)
    vocab = Vocab(LETTERS)
    train_pools, test_pools = pools_for(SESSIONS_TRAIN), pools_for([SESSION_TEST])
    train_words, test_words = load_words(4000, rng), load_words(300, np.random.default_rng(99))
    train_items = [(spell(w, train_pools, rng), w) for w in train_words]
    test_items = [(spell(w, test_pools, np.random.default_rng(i)), w) for i, w in enumerate(test_words)]
    val_items = test_items[:100]

    net = SeqNet(FEAT_DIM, vocab.size, hidden=96, layers=2)

    def val_fn(n):
        np_model = SeqModel.from_torch(n, vocab)
        cer = np.mean([char_error_rate(w, np_model.decode(f)) for f, w in val_items])
        return {"val_cer": float(cer), "score": float(-cer)}

    t0 = time.time()
    train(net, train_items, vocab, epochs=15, batch=32, lr=3e-3, augment=augment, val_fn=val_fn)
    seq = SeqModel.from_torch(net.eval(), vocab)
    base = LetterModel.load(ROOT / "models" / "letter_mlp_final.npz")

    held = test_items[100:]                      # never used for model selection
    seq_out = [seq.decode(f) for f, _ in held]
    base_out = [baseline_decode(base, f) for f, _ in held]
    res = {
        "train_words": len(train_items), "test_words": len(held), "train_minutes": round((time.time() - t0) / 60, 1),
        "seq_params": seq.n_params,
        "cer_sequence_model": round(float(np.mean([char_error_rate(w, o) for (_, w), o in zip(held, seq_out)])), 3),
        "cer_per_frame_baseline": round(float(np.mean([char_error_rate(w, o) for (_, w), o in zip(held, base_out)])), 3),
        "word_exact_sequence_model": round(float(np.mean([w == o for (_, w), o in zip(held, seq_out)])), 3),
        "word_exact_baseline": round(float(np.mean([w == o for (_, w), o in zip(held, base_out)])), 3),
        "examples": [{"word": w, "sequence": s, "baseline": b} for (_, w), s, b in list(zip(held, seq_out, base_out))[:12]],
    }
    (ROOT / "artifacts" / "seqsim_results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "examples"}, indent=1))
    for e in res["examples"]:
        print(f"  {e['word']:<10} seq: {e['sequence']:<10} baseline: {e['baseline']}")


if __name__ == "__main__":
    main()
