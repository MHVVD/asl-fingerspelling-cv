"""Stage 5: train the letter classifier on CPU and evaluate it on held-out sessions.

    python -m aslcv.train                                        # public data, test on block b5
    python -m aslcv.train --data recorded --test-sessions 20261011-011708
    python -m aslcv.train --data public   --test-sessions 20261011-011708 --name public_on_me

--data picks what the model LEARNS from. --test-sessions picks the held-out rows, which are
looked up in all data (public + recorded), so a public-only model can be tested on your session.

Trains two models on the same split:
  * baseline: logistic regression (linear, a sanity check of how separable the features are)
  * main:     small MLP 63 -> 128 -> 64 -> 26, exported to models/letter_mlp.npz for NumPy-only inference
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np

from .dataset import load_dataset

ROOT = Path(__file__).resolve().parents[2]
DATA_DIRS = {"public": ROOT / "data" / "public", "recorded": ROOT / "data" / "recorded"}
TRAIN_CHOICES = {"public": {"public"}, "recorded": {"recorded"}, "both": {"public", "recorded"}}


def load_all():
    """All data with a per-row source tag. Returns X, y, sessions, source."""
    parts = []
    for name, d in DATA_DIRS.items():
        if Path(d).exists():
            X, y, s = load_dataset(d)
            if len(y):
                parts.append((X, y, s, np.full(len(y), name)))
    if not parts:
        raise SystemExit(f"no data found in {[str(d) for d in DATA_DIRS.values()]}")
    return tuple(np.concatenate(a) for a in zip(*parts))


def split_by_session(sessions, test_sessions):
    """Whole sessions go to one side only. Frames within a session are near-duplicates,
    so a random split would leak them across train/test and inflate accuracy."""
    is_test = np.isin(sessions, list(test_sessions))
    return np.flatnonzero(~is_test), np.flatnonzero(is_test)


def make_split(source, sessions, train_sources, test_sessions):
    """Train on rows from `train_sources` outside the test sessions; test on the test sessions
    wherever they come from. Lets every model variant be scored on the same held-out rows."""
    tr, te = split_by_session(sessions, test_sessions)
    return tr[np.isin(source[tr], list(train_sources))], te


def augment(X, y, copies=1, rng=None, max_rot_deg=15, noise=0.02):
    """Add rotated + jittered copies of each sample (originals kept first).
    In-plane rotation mimics tilted hands/cameras; small noise mimics landmark jitter.
    Rotation is about the wrist (the origin after normalization), so the wrist stays at 0."""
    rng = rng or np.random.default_rng(0)
    pts = X.reshape(-1, 21, 3)
    out_X, out_y = [X], [y]
    for _ in range(copies):
        a = np.radians(rng.uniform(-max_rot_deg, max_rot_deg, len(X)))
        c, s = np.cos(a), np.sin(a)
        rot = np.zeros((len(X), 3, 3))
        rot[:, 0, 0], rot[:, 0, 1], rot[:, 1, 0], rot[:, 1, 1], rot[:, 2, 2] = c, -s, s, c, 1
        p = np.einsum("nij,nkj->nki", rot, pts)
        p += rng.normal(0, noise, p.shape)
        p -= p[:, :1]                                  # re-anchor the wrist after jitter
        out_X.append(p.reshape(len(X), -1).astype(X.dtype))
        out_y.append(y)
    return np.concatenate(out_X), np.concatenate(out_y)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", choices=TRAIN_CHOICES, default="public", help="what the model learns from")
    ap.add_argument("--test-sessions", default="asl_alphabet:b5", help="comma-separated session ids held out for testing")
    ap.add_argument("--name", default=None, help="run name for output files (default: the --data value)")
    ap.add_argument("--hidden", default="128,64")
    ap.add_argument("--aug-copies", type=int, default=1)
    a = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from .evaluate import metrics, plot_confusion, latency_ms
    from .model import LetterModel

    name = a.name or a.data
    X, y, sessions, source = load_all()
    test_sessions = {s.strip() for s in a.test_sessions.split(",") if s.strip()}
    tr, te = make_split(source, sessions, TRAIN_CHOICES[a.data], test_sessions)
    if len(tr) == 0:
        raise SystemExit(f"no training rows left for --data {a.data} after holding out {sorted(test_sessions)}")
    if len(te) == 0:
        raise SystemExit(f"none of the test sessions {sorted(test_sessions)} are in the data; available: {sorted(set(sessions))}")
    labels = sorted(set(y))
    print(f"train on {a.data}: {len(tr)} rows | test {len(te)} rows (sessions {sorted(test_sessions)})")

    X_tr, y_tr = augment(X[tr], y[tr], copies=a.aug_copies)
    X_te, y_te = X[te], y[te]
    results = {"data": a.data, "test_sessions": sorted(test_sessions), "n_train_raw": int(len(tr)),
               "n_train_augmented": int(len(X_tr)), "n_test": int(len(te)), "models": {}}

    t = time.time()
    base = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
    results["models"]["logreg"] = {"train_seconds": round(time.time() - t, 1), **metrics(y_te, base.predict(X_te), labels)}
    print(f"  logreg  acc {results['models']['logreg']['accuracy']:.3f}  ({results['models']['logreg']['train_seconds']}s)")

    t = time.time()
    hidden = tuple(int(h) for h in a.hidden.split(","))
    # early_stopping holds out 10% of the (augmented) training rows to decide when to stop
    mlp = MLPClassifier(hidden_layer_sizes=hidden, early_stopping=True, max_iter=200, random_state=0).fit(X_tr, y_tr)
    model = LetterModel.from_sklearn(mlp)
    m = metrics(y_te, model.predict(X_te), labels)
    m.update(train_seconds=round(time.time() - t, 1), epochs=int(mlp.n_iter_), n_params=int(model.n_params),
             latency_ms_per_frame=round(latency_ms(model, X_te), 4))
    results["models"]["mlp"] = m
    print(f"  mlp     acc {m['accuracy']:.3f}  macro-F1 {m['macro_f1']:.3f}  ({m['train_seconds']}s, {m['epochs']} epochs, "
          f"{m['n_params']} params, {m['latency_ms_per_frame']:.3f} ms/frame)")

    models_dir, art = ROOT / "models", ROOT / "artifacts"
    models_dir.mkdir(exist_ok=True); art.mkdir(exist_ok=True)
    model_path = models_dir / f"letter_mlp_{name}.npz"
    model.save(model_path)
    m["model_file"] = str(model_path.relative_to(ROOT))
    m["model_kb"] = round(model_path.stat().st_size / 1024, 1)
    plot_confusion(m["confusion_matrix"], labels, art / f"confusion_{name}.png",
                   f"MLP trained on '{a.data}', tested on {', '.join(sorted(test_sessions))}  (acc {m['accuracy']:.1%})")
    for mm in results["models"].values():
        mm.pop("confusion_matrix", None)
    (art / f"train_metrics_{name}.json").write_text(json.dumps(results, indent=2))
    print("  worst letters:", ", ".join(f"{k} {v:.2f}" for k, v in sorted(m["per_letter_f1"].items(), key=lambda kv: kv[1])[:5]))
    print("  top confusions:", ", ".join(f"{c['true']}->{c['predicted']} {c['rate']:.0%}" for c in m["top_confusions"][:5]))
    print(f"saved {model_path} ({m['model_kb']} KB), artifacts/confusion_{name}.png, artifacts/train_metrics_{name}.json")


if __name__ == "__main__":
    main()
