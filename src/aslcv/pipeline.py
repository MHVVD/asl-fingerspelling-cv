"""Stage 6: one frame in -> letter probabilities out. Shared by the tutor and the video translator.

    frame -> MediaPipe hand -> handedness decision -> normalized features -> LetterModel -> probabilities
                            -> wrist motion (for double-letter bounces)
"""
from collections import Counter
from dataclasses import dataclass
import numpy as np

from .features import normalize_landmarks, resolve_handedness, WRIST, MIDDLE_MCP


def wrist_motion(prev_px, cur_px):
    """How far the wrist moved since the last frame, in hand sizes (wrist to middle knuckle).
    Dividing by hand size makes it the same whether the signer is near or far from the camera."""
    if prev_px is None:
        return 0.0
    size = np.linalg.norm(cur_px[MIDDLE_MCP, :2] - cur_px[WRIST, :2])
    return float(np.linalg.norm(cur_px[WRIST, :2] - prev_px[WRIST, :2]) / max(size, 1e-6))


class HandednessVote:
    """Running majority of MediaPipe's Left/Right labels. One signer keeps one hand, so the
    majority so far is far more reliable than any single frame (see journal 05)."""
    def __init__(self):
        self.counts = Counter()

    def update(self, label):
        self.counts[label] += 1
        return self.counts.most_common(1)[0][0]


@dataclass
class FrameResult:
    probs: np.ndarray | None   # class probabilities, None when no hand
    hand: object | None        # the detected Hand (for drawing), None when no hand
    motion: float              # wrist movement since the previous detected frame, in hand sizes
    side: str | None           # which hand the features were normalized as


class FramePipeline:
    def __init__(self, detector, model, hand="Right"):
        """hand: "Right"/"Left" for a known signer (tutor), "auto" to vote over frames (any video)."""
        self.detector, self.model, self.hand = detector, model, hand
        self.vote = HandednessVote()
        self._prev_px = None

    @property
    def classes(self):
        return list(self.model.classes)

    def process(self, frame_bgr, t_ms=None):
        hands = self.detector.detect(frame_bgr) if t_ms is None else self.detector.detect(frame_bgr, t_ms=t_ms)
        if not hands:
            return FrameResult(None, None, 0.0, None)
        h = hands[0]
        side = self.vote.update(h.handedness) if self.hand == "auto" else resolve_handedness(h.handedness, self.hand)
        feats = normalize_landmarks(h.landmarks_px, side)
        motion = wrist_motion(self._prev_px, h.landmarks_px)
        self._prev_px = h.landmarks_px
        return FrameResult(self.model.predict_proba(feats), h, motion, side)
