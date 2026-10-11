"""Stage 4: recording state machine + on-disk format for labeled landmark features.

One .npz per recording session in data/recorded/: X (n,63), y (n,) letters, session id.
Splitting by session (not by random frame) later avoids train/test leakage, since
consecutive frames of one recording are near-duplicates.
"""
import string
import time
from collections import defaultdict
from pathlib import Path
import numpy as np

LETTERS = set(string.ascii_uppercase)


class Recorder:
    def __init__(self, out_dir, session=None, target=150):
        self.out_dir = Path(out_dir)
        self.session = session or time.strftime("%Y%m%d-%H%M%S")
        self.target = target
        self.letter = None
        self.recording = False
        self._samples = []                 # list of (letter, features, raw_landmarks, model_label), in order
        self._batches = []                 # (letter, start_index) per recording burst
        self.counts = defaultdict(int)

    def select(self, letter):
        letter = letter.upper()
        if letter not in LETTERS:
            raise ValueError(f"not a letter: {letter!r}")
        self.letter = letter
        self.recording = False

    def toggle(self):
        if self.letter is None:
            return
        self.recording = not self.recording
        if self.recording:
            self._batches.append((self.letter, len(self._samples)))

    def add(self, features, raw=None, model_label=None):
        """Call once per frame that has a detected hand. Returns True if stored.
        raw (21,3 pixel-space landmarks) and model_label (MediaPipe's Left/Right guess) are kept for
        auditing, so a pipeline bug can be traced and the features recomputed without re-recording."""
        if not self.recording or self.letter is None:
            return False
        if self.counts[self.letter] >= self.target:
            self.recording = False
            return False
        self._samples.append((self.letter, np.asarray(features, dtype=np.float32), raw, model_label))
        self.counts[self.letter] += 1
        if self.counts[self.letter] >= self.target:
            self.recording = False  # auto-stop at target
        return True

    def undo(self):
        """Drop the most recent recording burst."""
        if not self._batches:
            return
        letter, start = self._batches.pop()
        removed = self._samples[start:]
        del self._samples[start:]
        for sample in removed:
            self.counts[sample[0]] -= 1

    def save(self):
        if not self._samples:
            return None
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"{self.session}.npz"
        X = np.stack([s[1] for s in self._samples])
        y = np.array([s[0] for s in self._samples])
        extra = {}
        if all(s[2] is not None for s in self._samples):
            extra["raw"] = np.stack([np.asarray(s[2], np.float32) for s in self._samples])
        if all(s[3] is not None for s in self._samples):
            extra["model_hand"] = np.array([s[3] for s in self._samples])
        np.savez(path, X=X, y=y, session=np.array(self.session), **extra)
        return path


def load_dataset(directory):
    """Load all session files. Returns X (n,63), y (n,), sessions (n,)."""
    Xs, ys, ss = [], [], []
    for p in sorted(Path(directory).glob("*.npz")):
        d = np.load(p)
        Xs.append(d["X"]); ys.append(d["y"])
        s = d["session"]
        # recorder files hold one id for the whole session; public-data files hold one per row
        ss.append(np.full(len(d["y"]), str(s)) if s.ndim == 0 else s.astype(str))
    if not Xs:
        return np.empty((0, 63), np.float32), np.empty(0, str), np.empty(0, str)
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ss)
