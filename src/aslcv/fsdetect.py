"""Fingerspelling detection (journal 17): which moments of a signing video are fingerspelling?

The letter models read *all* signing as letters (journal 15), so the translator decodes junk from ordinary signs.
This detector looks at both hands over time and gives each frame a probability of being fingerspelling; frames
above a threshold are joined into segments, and only those segments are read as letters.

Per frame, for the upper (more raised) hand and the lower hand:
  present flag, hand shape (63, wrist-centred and scaled, left hands mirrored), wrist position as a share of the
  frame, hand size as a share of the frame height, wrist velocity in hand sizes per frame;
plus where the lower hand is relative to the upper one. Fingerspelling tends to be one raised hand, held fairly
still near the shoulder, changing shape quickly, with the other hand down; the network learns its own version of
that from ChicagoFSWild's source videos (fsw_source.py).

Network: Linear+ReLU -> BiGRU -> Linear, trained with PyTorch (fsdetect_train.py), run here in NumPy.
"""
from pathlib import Path
import numpy as np

from .features import normalize_landmarks
from .seqmodel import bigru_forward

ROOT = Path(__file__).resolve().parents[2]
DETECTOR = ROOT / "models" / "fsdetect.npz"
SLOT_DIM = 1 + 63 + 2 + 1 + 2
FEAT_DIM = 2 * SLOT_DIM + 2
WRIST, MIDDLE_MCP = 0, 9


def _hand_size(pts):
    return float(np.linalg.norm(pts[MIDDLE_MCP, :2] - pts[WRIST, :2])) or 1.0


def detect_features(hands, handedness, size):
    """hands: (T, 2, 21, 3) pixel-space landmarks, NaN where absent; handedness: (T, 2) "Left"/"Right"/"";
    size: (width, height). Returns (T, FEAT_DIM) float32."""
    T = len(hands)
    w, h = float(size[0]), float(size[1])
    F = np.zeros((T, FEAT_DIM), np.float32)
    prev = [None, None]
    for t in range(T):
        present = [k for k in range(2) if not np.isnan(hands[t, k, 0, 0])]
        present.sort(key=lambda k: hands[t, k, WRIST, 1])                 # upper hand (smaller y) first
        slots = present + [None] * (2 - len(present))
        cur = [None, None]
        for s, k in enumerate(slots):
            if k is None:
                continue
            pts = hands[t, k]
            size_px = _hand_size(pts)
            o = s * SLOT_DIM
            F[t, o] = 1.0
            F[t, o + 1:o + 64] = normalize_landmarks(pts, handedness[t, k] or "Right")
            F[t, o + 64:o + 66] = pts[WRIST, 0] / w, pts[WRIST, 1] / h
            F[t, o + 66] = size_px / h
            if prev[s] is not None:
                F[t, o + 67:o + 69] = (pts[WRIST, :2] - prev[s]) / size_px
            cur[s] = pts[WRIST, :2]
        if slots[1] is not None:
            up, lo = hands[t, slots[0]], hands[t, slots[1]]
            F[t, -2:] = (lo[WRIST, :2] - up[WRIST, :2]) / _hand_size(up)
        prev = cur
    return F


def smooth(p, k=5):
    """Moving average over k frames (k odd)."""
    if len(p) < k:
        return p
    pad = np.pad(p, k // 2, mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def to_segments(prob, times, threshold=0.5, min_len_s=0.3, max_gap_s=0.3):
    """Per-frame probabilities -> [(start_s, end_s)] of fingerspelling: frames above threshold, short gaps bridged,
    very short runs dropped."""
    on = smooth(prob) >= threshold
    segs, start = [], None
    for i, v in enumerate(on):
        if v and start is None:
            start = i
        if not v and start is not None:
            segs.append([times[start], times[i - 1]])
            start = None
    if start is not None:
        segs.append([times[start], times[-1]])
    merged = []
    for s in segs:
        if merged and s[0] - merged[-1][1] <= max_gap_s:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    return [(float(a), float(b)) for a, b in merged if b - a >= min_len_s]


class FSDetector:
    def __init__(self, params, layers, threshold=0.5):
        self.p = {k: np.asarray(v, np.float32) for k, v in params.items()}
        self.layers, self.threshold = layers, threshold

    @classmethod
    def from_torch(cls, net, threshold=0.5):
        return cls({k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}, net.gru.num_layers, threshold)

    def prob(self, feats):
        z = bigru_forward(self.p, self.layers, feats)[:, 0]
        return 1.0 / (1.0 + np.exp(-z))

    def segments(self, feats, times):
        return to_segments(self.prob(feats), times, self.threshold)

    def save(self, path):
        np.savez(path, layers=self.layers, threshold=self.threshold, **self.p)

    @classmethod
    def load(cls, path=DETECTOR):
        d = dict(np.load(path))
        layers, threshold = int(d.pop("layers")), float(d.pop("threshold"))
        return cls(d, layers, threshold)
