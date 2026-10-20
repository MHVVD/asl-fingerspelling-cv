"""Sequence model for the video translator: hand-point sequences -> words, in pure NumPy at runtime.

Instead of guessing a letter per frame and smoothing (infer.LetterStream), a small recurrent network reads
the whole sequence of hand features and outputs, for every frame, a probability for each letter plus a
"blank" (no new letter here). It is trained with CTC (connectionist temporal classification), which only
needs the word for each clip, not which frame shows which letter: the model learns the timing itself.

Decoding (greedy CTC): take the most likely symbol per frame, merge repeats, drop blanks.
    frames:  - h h - e e l l - l l o -      ->   "hello"   (the blank between the l's keeps both)

Architecture (trained in seqtrain.py with PyTorch, exported here):
    per-frame features (66) -> Linear + ReLU (hidden) -> bidirectional GRU x layers -> Linear -> symbols
"""
from pathlib import Path
import numpy as np

from .features import normalize_landmarks, WRIST, MIDDLE_MCP

FEAT_DIM = 63 + 2 + 1      # normalized hand shape, wrist velocity (x, y) in hand sizes, hand-present flag


class Vocab:
    """Index 0 is the CTC blank; symbols start at 1."""
    def __init__(self, symbols):
        self.symbols = list(symbols)
        self.index = {s: i + 1 for i, s in enumerate(self.symbols)}
        self.blank = 0
        self.size = len(self.symbols) + 1

    def encode(self, text):
        return [self.index[c] for c in text if c in self.index]

    def decode(self, ids):
        return "".join(self.symbols[i - 1] for i in ids if i != self.blank)


def sequence_features(raw_seq, handedness="Right"):
    """raw_seq: per frame a (21,3) array of pixel-space landmarks, or None when no hand was found.
    Returns (T, FEAT_DIM). Wrist velocity is measured against the last frame that had a hand, in
    hand sizes (wrist to middle knuckle), so it means the same near or far from the camera."""
    out = np.zeros((len(raw_seq), FEAT_DIM), np.float32)
    prev = None
    for t, raw in enumerate(raw_seq):
        if raw is None:
            continue
        raw = np.asarray(raw, float)
        out[t, :63] = normalize_landmarks(raw, handedness)
        size = max(float(np.linalg.norm(raw[MIDDLE_MCP, :2] - raw[WRIST, :2])), 1e-6)
        if prev is not None:
            out[t, 63:65] = (raw[WRIST, :2] - prev[WRIST, :2]) / size
        out[t, 65] = 1.0
        prev = raw
    return out


def ctc_greedy_decode(logits, vocab):
    best = np.argmax(logits, axis=-1)
    collapsed = [int(b) for i, b in enumerate(best) if i == 0 or b != best[i - 1]]
    return vocab.decode(collapsed)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _gru_direction(x, w_ih, w_hh, b_ih, b_hh, reverse=False):
    """One GRU direction over x (T, I), PyTorch's equations and gate order (r, z, n)."""
    T, H = x.shape[0], w_hh.shape[1]
    gi = x @ w_ih.T + b_ih                     # input contributions for all steps at once
    h = np.zeros(H, np.float32)
    out = np.zeros((T, H), np.float32)
    steps = range(T - 1, -1, -1) if reverse else range(T)
    for t in steps:
        gh = w_hh @ h + b_hh
        r = _sigmoid(gi[t, :H] + gh[:H])
        z = _sigmoid(gi[t, H:2 * H] + gh[H:2 * H])
        n = np.tanh(gi[t, 2 * H:] + r * gh[2 * H:])
        h = (1 - z) * n + z * h
        out[t] = h
    return out


class SeqModel:
    def __init__(self, params, vocab, layers):
        self.p = {k: np.asarray(v, np.float32) for k, v in params.items()}
        self.vocab, self.layers = vocab, layers

    @classmethod
    def from_torch(cls, net, vocab):
        params = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
        return cls(params, vocab, net.gru.num_layers)

    def logits(self, feats):
        p = self.p
        h = np.maximum(np.asarray(feats, np.float32) @ p["inp.weight"].T + p["inp.bias"], 0)
        for l in range(self.layers):
            fwd = _gru_direction(h, p[f"gru.weight_ih_l{l}"], p[f"gru.weight_hh_l{l}"],
                                 p[f"gru.bias_ih_l{l}"], p[f"gru.bias_hh_l{l}"])
            bwd = _gru_direction(h, p[f"gru.weight_ih_l{l}_reverse"], p[f"gru.weight_hh_l{l}_reverse"],
                                 p[f"gru.bias_ih_l{l}_reverse"], p[f"gru.bias_hh_l{l}_reverse"], reverse=True)
            h = np.concatenate([fwd, bwd], axis=1)
        return h @ p["out.weight"].T + p["out.bias"]

    def decode(self, feats):
        return ctc_greedy_decode(self.logits(feats), self.vocab)

    @property
    def n_params(self):
        return int(sum(v.size for v in self.p.values()))

    def save(self, path):
        np.savez(path, layers=self.layers, symbols=np.array(self.vocab.symbols), **self.p)

    @classmethod
    def load(cls, path):
        d = np.load(Path(path))
        params = {k: d[k] for k in d.files if k not in ("layers", "symbols")}
        return cls(params, Vocab(d["symbols"].tolist()), int(d["layers"]))
