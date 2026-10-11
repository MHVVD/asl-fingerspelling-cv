import numpy as np
import cv2
from aslcv.extract_public import discover_images, session_for, detect_with_fallback
from aslcv.landmarks import Hand


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.zeros((20, 20, 3), np.uint8))


def test_discover_finds_letters_only_with_frame_numbers(tmp_path):
    root = tmp_path / "asl_alphabet_train" / "asl_alphabet_train"
    for name in ["A/A1.jpg", "A/A1200.jpg", "B/B7.jpg", "del/del3.jpg", "nothing/nothing1.jpg", "space/space2.jpg"]:
        _touch(root / name)
    _touch(tmp_path / "asl_alphabet_test" / "asl_alphabet_test" / "A_test.jpg")  # not a class folder
    items = discover_images(tmp_path)
    assert sorted((label, idx) for _, label, idx in items) == [("A", 1), ("A", 1200), ("B", 7)]


def test_limit_per_class(tmp_path):
    for i in range(1, 6):
        _touch(tmp_path / "C" / f"C{i}.jpg")
    assert len(discover_images(tmp_path, limit_per_class=2)) == 2


def test_session_blocks_group_consecutive_frames():
    assert session_for(1, 500) == session_for(500, 500) == "asl_alphabet:b0"
    assert session_for(501, 500) == "asl_alphabet:b1"


class FakeDetector:
    """Finds a hand only when the image is larger than `min_size` (i.e. after padding)."""
    def __init__(self, min_size):
        self.min_size, self.calls = min_size, 0

    def detect(self, img):
        self.calls += 1
        if img.shape[0] >= self.min_size:
            lm = np.full((21, 3), 0.5)
            return [Hand(lm, lm * img.shape[0], "Right", 0.9)]
        return []


def test_fallback_not_used_when_plain_works():
    det = FakeDetector(min_size=0)
    hand, padded = detect_with_fallback(det, np.zeros((200, 200, 3), np.uint8), pad=0.25)
    assert hand is not None and padded is False and det.calls == 1


def test_fallback_pads_when_plain_fails():
    det = FakeDetector(min_size=250)
    hand, padded = detect_with_fallback(det, np.zeros((200, 200, 3), np.uint8), pad=0.25)
    assert hand is not None and padded is True and det.calls == 2


def test_no_hand_anywhere():
    hand, padded = detect_with_fallback(FakeDetector(min_size=10**6), np.zeros((200, 200, 3), np.uint8), pad=0.25)
    assert hand is None
