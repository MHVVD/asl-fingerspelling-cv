"""Tutor logic (camera-free, unit-tested): which letter to practise next, and how one attempt goes.

Planner: spaced repetition. Mastery per letter = (correct + 1) / (attempts + 2), so a new letter starts
at 0.5. The next letter is drawn at random with weight (1 - mastery)^2 + 0.02: weak letters come back
often, mastered ones occasionally. Never the same letter twice in a row. Progress persists as JSON.

Attempt: the target counts as signed once the smoothed prediction has been that letter, with
confidence >= min_conf, continuously for hold_s seconds. No success within timeout_s = missed.
"""
import json
import string
from pathlib import Path
import numpy as np

PROGRESS_PATH = Path(__file__).resolve().parents[2] / "data" / "progress.json"


class Planner:
    def __init__(self, letters=string.ascii_uppercase, path=None, seed=None):
        self.letters = [c for c in letters.upper() if c in string.ascii_uppercase]
        self.rng = np.random.default_rng(seed)
        self.stats = {k: {"attempts": 0, "correct": 0, "streak": 0} for k in string.ascii_uppercase}
        if path and Path(path).exists():
            self.stats.update(json.loads(Path(path).read_text()))
        self.last = None

    def mastery(self, letter):
        s = self.stats[letter]
        return (s["correct"] + 1) / (s["attempts"] + 2)

    def next(self):
        options = [k for k in self.letters if k != self.last] or self.letters
        w = np.array([(1 - self.mastery(k)) ** 2 + 0.02 for k in options])
        self.last = str(self.rng.choice(options, p=w / w.sum()))
        return self.last

    def record(self, letter, ok):
        s = self.stats[letter]
        s["attempts"] += 1
        s["correct"] += int(ok)
        s["streak"] = s["streak"] + 1 if ok else 0

    def save(self, path=PROGRESS_PATH):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.stats, indent=1))

    def summary(self):
        tried = [k for k in self.letters if self.stats[k]["attempts"]]
        return {"tried": len(tried), "mastered": sum(self.mastery(k) >= 0.75 for k in tried),
                "weakest": sorted(tried, key=self.mastery)[:5]}


class Attempt:
    def __init__(self, target, hold_s=0.6, timeout_s=12.0, min_conf=0.6):
        self.target, self.hold_s, self.timeout_s, self.min_conf = target, hold_s, timeout_s, min_conf
        self.t0 = self._hold_start = None
        self.result = None          # None while trying, then "success" or "missed"
        self.progress = 0.0         # 0..1 of the hold, for the on-screen bar
        self.elapsed = 0.0

    def update(self, seen, t):
        """seen: (letter, confidence) from the smoothed stream, or (None, 0.0) without a hand."""
        if self.result:
            return self.result
        if self.t0 is None:
            self.t0 = t
        self.elapsed = t - self.t0
        letter, conf = seen
        if letter == self.target and conf >= self.min_conf:
            if self._hold_start is None:
                self._hold_start = t
            self.progress = min(1.0, (t - self._hold_start) / self.hold_s)
            if t - self._hold_start >= self.hold_s - 1e-9:
                self.result = "success"
        else:
            self._hold_start, self.progress = None, 0.0
            if self.elapsed >= self.timeout_s:
                self.result = "missed"
        return self.result
