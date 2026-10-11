import numpy as np
from aslcv.features import normalize_landmarks, FEATURE_DIM


def make_hand(seed=0):
    return np.random.default_rng(seed).random((21, 3))


def test_output_shape():
    assert normalize_landmarks(make_hand()).shape == (FEATURE_DIM,)


def test_translation_invariant():
    h = make_hand()
    shifted = h + np.array([0.3, -0.2, 0.1])
    assert np.allclose(normalize_landmarks(h), normalize_landmarks(shifted))


def test_scale_invariant():
    h = make_hand()
    assert np.allclose(normalize_landmarks(h), normalize_landmarks(h * 2.5))


def test_wrist_is_origin():
    f = normalize_landmarks(make_hand()).reshape(21, 3)
    assert np.allclose(f[0], 0)


def test_left_hand_mirrored_to_right():
    h = make_hand()
    mirrored = h.copy()
    mirrored[:, 0] = -mirrored[:, 0]
    right = normalize_landmarks(h, handedness="Right")
    left = normalize_landmarks(mirrored, handedness="Left")
    assert np.allclose(right, left)


def test_fixed_hand_setting_ignores_unreliable_model_label():
    from aslcv.features import resolve_handedness
    # MediaPipe mislabels a downward-pointing right hand as "Left": a fixed setting must win
    assert resolve_handedness("Left", "Right") == "Right"
    assert resolve_handedness("Right", "Left") == "Left"
    assert resolve_handedness("Left", "auto") == "Left"


def test_majority_handedness_over_a_video():
    from aslcv.features import majority_handedness
    assert majority_handedness(["Right"] * 8 + ["Left"] * 2) == "Right"
    assert majority_handedness([]) == "Right"  # default when no hands seen
