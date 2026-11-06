"""Train and evaluate the fingerspelling detector (journal 17). PyTorch on CPU for training; runtime is NumPy.

    python -m aslcv.fsdetect_train                # baselines + BiGRU, threshold on dev, scored once on test

Data: data/fsw_source/landmarks/*.npz (fsw_source.py), split by ChicagoFSWild's signer-disjoint partitions.
Scores:
  * frame level: average precision (threshold-free), and precision / recall / F1 at the dev-chosen threshold;
  * segment level: a marked fingerspelling clip is found if a detected segment overlaps it with IoU >= 0.5;
    a detected segment is correct if it overlaps some marked clip with IoU >= 0.5.
Baselines: "a hand is visible" (what the translator effectively did before) and a per-frame classifier on the
same features (no time context). The model (trained on ChicagoFSWild material) stays local; numbers go to
artifacts/fsdetect_results.json.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np

from .fsdetect import FEAT_DIM, SLOT_DIM, DETECTOR, FSDetector, detect_features, smooth, to_segments
from .fsw_source import LANDMARKS

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "artifacts" / "fsdetect_results.json"


def load_videos():
    """-> {partition: [dict(id, times, feats, label, present)]}"""
    out = {"train": [], "dev": [], "test": []}
    for f in sorted(LANDMARKS.glob("*.npz")):
        if ".part" in f.name:
            continue
        d = np.load(f)
        hands = d["hands"]
        out[str(d["partition"])].append({
            "id": f.stem, "times": d["times"], "label": d["label"].astype(np.float32),
            "feats": detect_features(hands, d["handedness"], d["size"]),
            "present": ~np.isnan(hands[:, :, 0, 0]).all(axis=1)})
    return out


def true_segments(v):
    return to_segments(v["label"], v["times"], 0.5, min_len_s=0.0, max_gap_s=0.0)


def iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def segment_scores(videos, pred_segs, thr=0.5):
    found = total = correct = n_pred = 0
    for v, pred in zip(videos, pred_segs):
        truth = true_segments(v)
        total += len(truth)
        n_pred += len(pred)
        found += sum(any(iou(t, p) >= thr for p in pred) for t in truth)
        correct += sum(any(iou(p, t) >= thr for t in truth) for p in pred)
    return {"clip_recall": round(found / max(total, 1), 3), "segment_precision": round(correct / max(n_pred, 1), 3),
            "clips": total, "segments_detected": n_pred}


def frame_scores(y, p, thr):
    from sklearn.metrics import average_precision_score
    yhat = p >= thr
    tp = float((yhat & (y > 0.5)).sum())
    prec, rec = tp / max(yhat.sum(), 1), tp / max((y > 0.5).sum(), 1)
    return {"AP": round(float(average_precision_score(y, p)), 3), "precision": round(prec, 3),
            "recall": round(rec, 3), "F1": round(2 * prec * rec / max(prec + rec, 1e-9), 3)}


def best_threshold(videos, probs):
    """Threshold maximising frame F1 on (smoothed) dev probabilities."""
    y = np.concatenate([v["label"] for v in videos])
    p = np.concatenate([smooth(q) for q in probs])
    return max(np.arange(0.05, 0.96, 0.05), key=lambda t: frame_scores(y, p, t)["F1"])


def evaluate(videos, probs, thr):
    y = np.concatenate([v["label"] for v in videos])
    p = np.concatenate([smooth(q) for q in probs])
    segs = [to_segments(q, v["times"], thr) for v, q in zip(videos, probs)]
    return {**frame_scores(y, p, thr), **segment_scores(videos, segs), "threshold": round(float(thr), 2)}


def mirror(F):
    """Horizontally flipped video: x positions, x velocities and the hands' x offset change sign (hand shapes are
    already normalised to one handedness)."""
    F = F.copy()
    for s in range(2):
        o = s * SLOT_DIM
        F[:, o + 64] = np.where(F[:, o] > 0, 1.0 - F[:, o + 64], 0.0)
        F[:, o + 67] = -F[:, o + 67]
    F[:, -2] = -F[:, -2]
    return F


def train_bigru(train, dev, hidden=64, layers=2, steps=3000, window=192, batch=32, lr=2e-3, seed=0, log=print):
    import torch
    from torch import nn
    from .seqtrain import SeqNet
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    net = SeqNet(FEAT_DIM, 1, hidden=hidden, layers=layers)
    pos = np.mean(np.concatenate([v["label"] for v in train]))
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    lengths = np.array([len(v["label"]) for v in train], float)
    pick = lengths / lengths.sum()                                  # sample videos by length
    best, best_state, history = -1.0, None, []
    for step in range(1, steps + 1):
        net.train()
        xs, ys = [], []
        for _ in range(batch):
            v = train[rng.choice(len(train), p=pick)]
            a = rng.integers(0, max(1, len(v["label"]) - window))
            x = v["feats"][a:a + window]
            xs.append(mirror(x) if rng.random() < 0.5 else x)
            ys.append(v["label"][a:a + window])
        n = min(len(x) for x in xs)
        x = torch.from_numpy(np.stack([x[:n] for x in xs]))
        y = torch.from_numpy(np.stack([y[:n] for y in ys]))
        loss = lossf(net(x)[..., 0], y)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        sched.step()
        if step % 250 == 0:
            det = FSDetector.from_torch(net)
            probs = [det.prob(v["feats"]) for v in dev]
            ap = frame_scores(np.concatenate([v["label"] for v in dev]), np.concatenate(probs), 0.5)["AP"]
            history.append({"step": step, "loss": round(float(loss), 4), "dev_AP": ap})
            log(f"  step {step}  loss {float(loss):.3f}  dev AP {ap:.3f}")
            if ap > best:
                best, best_state = ap, {k: t.clone() for k, t in net.state_dict().items()}
    net.load_state_dict(best_state)
    return FSDetector.from_torch(net), history


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--test", action="store_true", help="also score the test videos (do this once)")
    a = ap.parse_args()

    t0 = time.time()
    data = load_videos()
    for k, vs in data.items():
        n = sum(len(v["label"]) for v in vs)
        print(f"{k}: {len(vs)} videos, {n} frames, {np.mean(np.concatenate([v['label'] for v in vs])) if vs else 0:.1%}"
              " fingerspelling")
    results = {"data": {k: {"videos": len(vs), "frames": int(sum(len(v["label"]) for v in vs))} for k, vs in data.items()}}
    splits = ["dev"] + (["test"] if a.test else [])

    # baseline 0: "a hand is visible" = fingerspelling
    for s in splits:
        probs = [v["present"].astype(float) for v in data[s]]
        results.setdefault(s, {})["hand_visible"] = evaluate(data[s], probs, 0.5)

    # baseline 1: per-frame classifier on the same features
    from sklearn.linear_model import LogisticRegression
    X = np.concatenate([v["feats"] for v in data["train"]])
    y = np.concatenate([v["label"] for v in data["train"]])
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X, y)
    pf = lambda v: lr.predict_proba(v["feats"])[:, 1]
    thr = best_threshold(data["dev"], [pf(v) for v in data["dev"]])
    for s in splits:
        results[s]["per_frame"] = evaluate(data[s], [pf(v) for v in data[s]], thr)

    # the detector
    det, history = train_bigru(data["train"], data["dev"], a.hidden, a.layers, a.steps)
    det.threshold = float(best_threshold(data["dev"], [det.prob(v["feats"]) for v in data["dev"]]))
    for s in splits:
        results[s]["bigru"] = evaluate(data[s], [det.prob(v["feats"]) for v in data[s]], det.threshold)
    results["history"] = history
    results["settings"] = {"hidden": a.hidden, "layers": a.layers, "steps": a.steps,
                           "params": int(sum(v.size for v in det.p.values())), "minutes": round((time.time() - t0) / 60, 1)}
    DETECTOR.parent.mkdir(exist_ok=True)
    det.save(DETECTOR)
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=1))
    for s in splits:
        print(f"\n{s}:")
        for name, r in results[s].items():
            print(f"  {name:12s} {r}")
    print(f"\nsaved {DETECTOR} and {RESULTS}")


if __name__ == "__main__":
    main()
