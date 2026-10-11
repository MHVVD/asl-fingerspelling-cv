"""Stage 5: metrics, confusion matrix and latency for a trained letter model."""
import time
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def metrics(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_letter = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 4),
        "per_letter_f1": {l: round(float(f), 3) for l, f in zip(labels, per_letter)},
        "top_confusions": top_confusions(cm, labels),
        "confusion_matrix": cm.tolist(),
    }


def top_confusions(cm, labels, k=10):
    """Most frequent (true -> predicted) mistakes, as a share of that true letter's samples."""
    rows = cm.sum(axis=1, keepdims=True).clip(min=1)
    rate = cm / rows
    np.fill_diagonal(rate, 0)
    flat = np.argsort(rate, axis=None)[::-1][:k]
    out = []
    for i in flat:
        t, p = divmod(int(i), len(labels))
        if rate[t, p] > 0:
            out.append({"true": labels[t], "predicted": labels[p], "rate": round(float(rate[t, p]), 3)})
    return out


def plot_confusion(cm, labels, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.asarray(cm, float)
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for t in range(len(labels)):
        for p in range(len(labels)):
            if norm[t, p] >= 0.05:
                ax.text(p, t, f"{norm[t, p]:.0%}".replace("100%", "1"), ha="center", va="center", fontsize=6,
                        color="white" if norm[t, p] > 0.5 else "black")
    fig.colorbar(im, fraction=0.046, label="share of true letter")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def latency_ms(model, X, repeats=2000):
    """Mean time for one single-sample prediction, which is what a live frame costs."""
    x = X[0]
    model.predict_proba(x)
    t = time.perf_counter()
    for i in range(repeats):
        model.predict_proba(X[i % len(X)])
    return (time.perf_counter() - t) / repeats * 1000
