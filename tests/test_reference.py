import string
import numpy as np
import pytest
from aslcv.reference import reference_pose, render_panel, LETTER_SPECS
from aslcv.features import normalize_landmarks

TIP = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}


def test_all_26_letters_defined():
    assert set(LETTER_SPECS) == set(string.ascii_uppercase)


@pytest.mark.parametrize("letter", list(string.ascii_uppercase))
def test_pose_is_valid_landmark_array(letter):
    pose = reference_pose(letter)
    assert pose.shape == (21, 3)
    assert np.all(np.isfinite(pose))
    assert np.all(np.isfinite(normalize_landmarks(pose)))


def test_f_thumb_touches_index():
    p = reference_pose("F")
    assert np.linalg.norm(p[TIP["thumb"]] - p[TIP["index"]]) < 0.05


def test_l_thumb_is_far_from_index():
    p = reference_pose("L")
    assert np.linalg.norm(p[TIP["thumb"]] - p[TIP["index"]]) > 0.8


def test_extended_finger_is_above_curled_one():
    p = reference_pose("I")  # pinky up, index curled; y grows downward
    assert p[TIP["pinky"], 1] < p[TIP["index"], 1] - 0.4


def test_sideways_letter_points_sideways():
    p = reference_pose("G")
    vec = p[TIP["index"]] - p[0]
    assert abs(vec[0]) > abs(vec[1])


@pytest.mark.parametrize("letter", [None, "A", "J", "Z"])
def test_render_panel_shape(letter):
    img = render_panel(letter, width=300, height=480)
    assert img.shape == (480, 300, 3) and img.dtype == np.uint8
