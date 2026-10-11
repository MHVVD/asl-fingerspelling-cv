"""Stage 5: the runtime letter classifier, pure NumPy.

Training uses scikit-learn, but at runtime we only need the learned weights: a forward pass
through a small MLP is a few matrix multiplications. Keeping torch/sklearn out of the runtime
keeps start-up fast and memory low on a laptop CPU (importing torch alone takes ~20 s here).

    probs = softmax(relu(relu(x·W1 + b1)·W2 + b2)·W3 + b3)
"""
from pathlib import Path
import numpy as np


class LetterModel:
    def __init__(self, weights, biases, classes):
        self.weights = [np.asarray(w, np.float32) for w in weights]
        self.biases = [np.asarray(b, np.float32) for b in biases]
        self.classes = np.asarray(classes)

    @classmethod
    def from_sklearn(cls, clf):
        """Copy the learned weights out of a fitted sklearn MLPClassifier (relu hidden layers, softmax output)."""
        if clf.activation != "relu" or clf.out_activation_ != "softmax":
            raise ValueError("expects a multi-class MLPClassifier with relu activation")
        return cls(clf.coefs_, clf.intercepts_, clf.classes_)

    def predict_proba(self, X):
        X = np.asarray(X, np.float32)
        single = X.ndim == 1
        h = X[None] if single else X
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            h = np.maximum(h @ W + b, 0.0)                    # hidden layer + ReLU
        logits = h @ self.weights[-1] + self.biases[-1]
        logits -= logits.max(axis=1, keepdims=True)           # numerically stable softmax
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        return p[0] if single else p

    def predict(self, X):
        return self.classes[np.argmax(self.predict_proba(X), axis=-1)]

    def top(self, x, k=1):
        """Best letter and its confidence for one feature vector (or the top-k list if k > 1)."""
        p = self.predict_proba(x)
        idx = np.argsort(p)[::-1][:k]
        pairs = [(str(self.classes[i]), float(p[i])) for i in idx]
        return pairs[0] if k == 1 else pairs

    @property
    def n_params(self):
        return sum(w.size for w in self.weights) + sum(b.size for b in self.biases)

    def save(self, path):
        arrays = {f"W{i}": w for i, w in enumerate(self.weights)}
        arrays.update({f"b{i}": b for i, b in enumerate(self.biases)})
        np.savez(path, classes=self.classes, n_layers=len(self.weights), **arrays)

    @classmethod
    def load(cls, path):
        d = np.load(Path(path))
        n = int(d["n_layers"])
        return cls([d[f"W{i}"] for i in range(n)], [d[f"b{i}"] for i in range(n)], d["classes"])
