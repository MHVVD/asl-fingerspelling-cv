"""Stage 3: turn raw hand landmarks into position/size/handedness-independent features."""
import numpy as np

NUM_LANDMARKS = 21
FEATURE_DIM = NUM_LANDMARKS * 3
WRIST, MIDDLE_MCP = 0, 9


def resolve_handedness(model_label, setting="Right"):
    """Which hand to treat the detection as. MediaPipe's per-frame label is unreliable for some poses
    (e.g. a downward-pointing right hand in P/Q gets called "Left"), and a wrong label mirrors the
    hand into a different shape. So a fixed setting ("Right"/"Left") overrides it; "auto" trusts it."""
    return model_label if setting == "auto" else setting


def majority_handedness(labels, default="Right"):
    """For a whole video of one signer: vote over all frames instead of trusting each one."""
    labels = list(labels)
    if not labels:
        return default
    return max(set(labels), key=labels.count)


def normalize_landmarks(landmarks, handedness="Right"):
    """landmarks: (21,3) array of x,y,z. Returns a flat 63-d feature vector.

    1. Mirror left hands (flip x) so one model handles both hands.
    2. Translate so the wrist is the origin (removes where the hand is in frame).
    3. Divide by wrist->middle-finger-base distance (removes hand size / camera distance).
    """
    pts = np.asarray(landmarks, dtype=np.float64).copy()
    if handedness == "Left":
        pts[:, 0] = -pts[:, 0]
    pts -= pts[WRIST]
    scale = np.linalg.norm(pts[MIDDLE_MCP])
    pts /= scale if scale > 1e-9 else 1.0
    return pts.reshape(-1)
