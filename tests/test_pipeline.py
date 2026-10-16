import numpy as np
from aslcv.pipeline import FramePipeline, HandednessVote, wrist_motion
from aslcv.landmarks import Hand


def hand_at(dx=0.0, label="Right", scale=100.0):
    pts = np.zeros((21, 3)); pts[9] = [0, -scale, 0]          # middle MCP one hand-size above wrist
    pts[:, 0] += 300 + dx; pts[:, 1] += 300
    return Hand(pts / [640, 480, 640], pts, label, 0.9)


def test_wrist_motion_in_hand_sizes():
    assert wrist_motion(None, hand_at().landmarks_px) == 0.0
    assert abs(wrist_motion(hand_at().landmarks_px, hand_at(dx=50).landmarks_px) - 0.5) < 1e-9


def test_handedness_vote_resists_occasional_wrong_labels():
    v = HandednessVote()
    for label in ["Right"] * 20 + ["Left"] * 5:
        side = v.update(label)
    assert side == "Right"


class FakeDetector:
    def __init__(self, hands): self.hands, self.i = hands, 0
    def detect(self, frame):
        h = self.hands[self.i]; self.i += 1
        return [h] if h is not None else []


class FakeModel:
    classes = np.array(list("AB"))
    def __init__(self): self.seen = []
    def predict_proba(self, x):
        self.seen.append(x); return np.array([0.9, 0.1])


def test_pipeline_outputs_probs_motion_and_none_without_hand():
    frames = [hand_at(), hand_at(dx=20), None]
    pipe = FramePipeline(FakeDetector(frames), FakeModel(), hand="Right")
    f = np.zeros((480, 640, 3), np.uint8)
    r1, r2, r3 = pipe.process(f), pipe.process(f), pipe.process(f)
    assert np.allclose(r1.probs, [0.9, 0.1]) and r1.motion == 0.0
    assert abs(r2.motion - 0.2) < 1e-9
    assert r3.probs is None and r3.hand is None


def test_auto_hand_uses_vote_not_single_frame_label():
    model = FakeModel()
    frames = [hand_at(label="Right")] * 10 + [hand_at(label="Left")]
    pipe = FramePipeline(FakeDetector(frames), model, hand="auto")
    f = np.zeros((480, 640, 3), np.uint8)
    for _ in frames:
        pipe.process(f)
    # the last frame was labeled Left but the vote says Right, so it must NOT be mirrored
    assert np.allclose(model.seen[-1], model.seen[0])


def hand_at_height(y_px, label="Right"):
    pts = np.zeros((21, 3)); pts[9] = [0, -100, 0]
    pts[:, 0] += 300; pts[:, 1] += y_px
    return Hand(pts / [640, 480, 640], pts, label, 0.9)


def test_signing_hand_is_the_raised_one_not_the_resting_one():
    """Real clip (journal 11): MediaPipe locked onto the signer's resting hand at the waist
    while the other hand fingerspelled at shoulder height."""
    resting, signing = hand_at_height(450), hand_at_height(200)

    class TwoHands:
        def detect(self, frame, t_ms=None):
            return [resting, signing]          # resting hand listed first

    pipe = FramePipeline(TwoHands(), FakeModel(), hand="Right")
    r = pipe.process(np.zeros((480, 640, 3), np.uint8))
    assert r.hand is signing
