"""Stage 6: turn noisy per-frame predictions into a clean stream of letters and spaces.

Used by both modes:
  * tutor:            `current` gives a steady letter + confidence to compare with the target
  * video translator: `update()` returns events ("letter H at 2.4 s", "space") that become subtitles

Rules (all thresholds are constructor parameters):
  1. Smooth: exponential moving average (EMA) of the probability vectors. New frames weigh `alpha`.
  2. Commit: a letter is emitted once it has been the top smoothed letter with confidence >= `enter`
     for `hold` consecutive frames. Short flickers never reach `hold`.
  3. Double letters: after emitting X, X can only be emitted again after a "release": its smoothed
     confidence drops below `release`, the wrist jumps by more than `motion_release` hand-sizes
     in one frame (the small bounce signers use for double letters), or the hand briefly disappears.
  4. Spaces: no hand for `gap` frames ends a word (one space, never leading or doubled).
     Shorter dropouts (detector misses a frame or two) are ignored.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class Event:
    kind: str        # "letter" or "space"
    value: str       # the letter, or " "
    t_start: float   # when this letter started being the steady candidate (s)
    t_commit: float  # when it was emitted (s)
    confidence: float = 0.0


class LetterStream:
    def __init__(self, classes, alpha=0.4, enter=0.7, hold=6, release=0.45,
                 gap=10, short_dropout=3, motion_release=0.15):
        self.classes = list(classes)
        self.alpha, self.enter, self.hold, self.release = alpha, enter, hold, release
        self.gap, self.short_dropout, self.motion_release = gap, short_dropout, motion_release
        self.reset()

    @classmethod
    def for_fps(cls, classes, fps, hold_s=0.2, gap_s=0.33, dropout_s=0.1, **kw):
        """Same behaviour in seconds at any frame rate (defaults equal the tuned 30 fps values)."""
        return cls(classes, hold=max(2, round(hold_s * fps)), gap=max(2, round(gap_s * fps)),
                   short_dropout=max(1, round(dropout_s * fps)), **kw)

    def reset(self):
        self.smoothed = None
        self._candidate, self._run, self._run_start = None, 0, 0.0
        self._last_letter, self._armed = None, True
        self._missing = 0
        self._last_kind = None   # kind of the last emitted event

    @property
    def current(self):
        """(letter, confidence) of the smoothed prediction right now, or (None, 0.0) without a hand."""
        if self.smoothed is None or self._missing > 0:
            return None, 0.0
        i = int(np.argmax(self.smoothed))
        return self.classes[i], float(self.smoothed[i])

    def update(self, probs, t=0.0, motion=0.0):
        """Feed one frame. probs: class probabilities, or None if no hand was detected.
        motion: wrist displacement since the last frame, in hand-size units. Returns new events."""
        if probs is None:
            return self._no_hand(t)

        events = []
        if self._missing >= self.short_dropout:
            self._armed = True                       # hand left and came back: repeats allowed
        if self._missing >= self.gap:
            self.smoothed = None                     # new word: forget the old letter's momentum
        self._missing = 0

        probs = np.asarray(probs, dtype=float)
        self.smoothed = probs if self.smoothed is None else self.alpha * probs + (1 - self.alpha) * self.smoothed
        i = int(np.argmax(self.smoothed))
        letter, conf = self.classes[i], float(self.smoothed[i])

        # release the last emitted letter so it can repeat (double letters)
        if self._last_letter is not None and not self._armed:
            last_conf = self.smoothed[self.classes.index(self._last_letter)]
            bounced = self.motion_release is not None and motion > self.motion_release
            if last_conf < self.release or bounced:
                self._armed = True
                self._run = 0                          # the repeat has to be held again from now

        # count consecutive confident frames for the current candidate
        if conf >= self.enter:
            if letter == self._candidate:
                self._run += 1
            else:
                self._candidate, self._run, self._run_start = letter, 1, t
        else:
            self._candidate, self._run = None, 0

        if self._run >= self.hold and (letter != self._last_letter or self._armed):
            events.append(Event("letter", letter, self._run_start, t, conf))
            self._last_letter, self._armed, self._last_kind = letter, False, "letter"
            self._run = 0
        return events

    def _no_hand(self, t):
        self._missing += 1
        self._candidate, self._run = None, 0
        if self._missing == self.gap and self._last_kind == "letter":
            self._last_kind = "space"
            return [Event("space", " ", t, t)]
        return []


def transcript(events):
    """Events -> text, without a trailing space."""
    return "".join(e.value for e in events).strip()
