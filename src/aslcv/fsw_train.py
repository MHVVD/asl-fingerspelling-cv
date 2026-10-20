"""ChicagoFSWild: benchmark today's system, then train and test the sequence model.

    python -m aslcv.fsw_train --baseline-only     # score the per-frame model + smoothing
    python -m aslcv.fsw_train --epochs 40         # train the CTC sequence model, pick by dev, score on test

Measure: letter accuracy = 1 - total edits / total reference letters (the dataset's standard measure).
Splits are signer-disjoint, so test accuracy is accuracy on people the model has never seen.
Results go to artifacts/fsw_results.json and the model to models/seq_fsw.npz; both stay local (git-ignored:
the results contain dataset labels, which can include real names). artifacts/fsw_summary.json holds the numbers only.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np

from .extract_fsw import FSW, VOCAB, letter_accuracy, unpack_sequence
from .features import majority_handedness
from .infer import LetterStream, transcript
from .model import LetterModel
from .seqmodel import FEAT_DIM, SeqModel, Vocab, sequence_features

ROOT = Path(__file__).resolve().parents[2]


def load_split(part):
    """-> list of (features (T, FEAT_DIM), label) for one partition."""
    d = np.load(FSW / f"landmarks_{part}.npz")
    items = []
    for i, label in enumerate(d["labels"]):
        raw = unpack_sequence(d, i)
        mp = str(d["mp_hand"][i])
        side = majority_handedness([{"R": "Right", "L": "Left"}[c] for c in mp if c in "RL"])
        items.append((sequence_features(raw, side), str(label)))
    return items


def baseline_decode(model, F, fps=30):
    s = LetterStream.for_fps(model.classes, fps)
    ev = []
    for t, row in enumerate(F):
        p = model.predict_proba(row[:63]) if row[65] else None
        ev += s.update(p, t=t / fps, motion=float(np.linalg.norm(row[63:65])))
    return transcript(ev).lower()


def augment(F, rng, max_rot_deg=15, noise=0.02, speed=(0.8, 1.25), slow_share=0.0, slow=(0.2, 0.6)):
    """Per clip: one in-plane rotation, landmark jitter, and a random speed change (resample in time).
    slow_share of the clips are slowed to 0.2-0.6x, like deliberate signing or a learner (journal 12)."""
    from .seqsim import augment as rot_noise
    F = rot_noise(F, rng, max_rot_deg, noise)
    k = rng.uniform(*slow) if rng.random() < slow_share else rng.uniform(*speed)
    T = max(2, int(round(len(F) / k)))
    idx = np.clip(np.round(np.linspace(0, len(F) - 1, T)).astype(int), 0, len(F) - 1)
    return F[idx]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline-only", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--slow-share", type=float, default=0.0, help="share of training clips slowed to 0.2-0.6x")
    ap.add_argument("--tag", default="", help="suffix for the model file and results key")
    a = ap.parse_args()

    out = ROOT / "artifacts" / "fsw_results.json"
    results = json.loads(out.read_text()) if out.exists() else {}
    test = load_split("test")
    refs = [l for _, l in test]

    base = LetterModel.load(ROOT / "models" / "letter_mlp_final.npz")
    hyps = [baseline_decode(base, f) for f, _ in test]
    results["baseline_per_frame_smoothing"] = {
        "test_letter_accuracy": round(letter_accuracy(refs, hyps), 4),
        "examples": [{"truth": r, "output": h} for r, h in list(zip(refs, hyps))[:15]]}
    print(f"baseline (per-frame + smoothing) test letter accuracy: {results['baseline_per_frame_smoothing']['test_letter_accuracy']:.1%}")
    if a.baseline_only:
        out.write_text(json.dumps(results, indent=1))
        return

    from .seqtrain import SeqNet, train
    vocab = Vocab(VOCAB)
    tr, dev = load_split("train"), load_split("dev")
    tr = [(f, l) for f, l in tr if l]
    net = SeqNet(FEAT_DIM, vocab.size, hidden=a.hidden, layers=a.layers)

    def val_fn(n):
        m = SeqModel.from_torch(n, vocab)
        acc = letter_accuracy([l for _, l in dev], [m.decode(f) for f, _ in dev])
        return {"dev_letter_acc": acc, "score": acc}

    t0 = time.time()
    from functools import partial
    hist = train(net, tr, vocab, epochs=a.epochs, batch=32, lr=2e-3,
                 augment=partial(augment, slow_share=a.slow_share), val_fn=val_fn)
    seq = SeqModel.from_torch(net.eval(), vocab)
    suffix = f"_{a.tag}" if a.tag else ""
    seq.save(ROOT / "models" / f"seq_fsw{suffix}.npz")
    hyps = [seq.decode(f) for f, _ in test]
    results[f"sequence_ctc{suffix}"] = {
        "test_letter_accuracy": round(letter_accuracy(refs, hyps), 4),
        "best_dev_letter_accuracy": round(max(h["dev_letter_acc"] for h in hist), 4),
        "params": seq.n_params, "epochs": a.epochs, "hidden": a.hidden, "layers": a.layers, "slow_share": a.slow_share,
        "train_minutes": round((time.time() - t0) / 60, 1), "history": hist,
        "examples": [{"truth": r, "output": h} for r, h in list(zip(refs, hyps))[:15]]}
    out.write_text(json.dumps(results, indent=1))
    r = results[f"sequence_ctc{suffix}"]
    print(f"sequence model{suffix} test letter accuracy: {r['test_letter_accuracy']:.1%} (best dev {r['best_dev_letter_accuracy']:.1%})")


if __name__ == "__main__":
    main()
