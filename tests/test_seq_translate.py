import numpy as np

from aslcv.seq_translate import find_segments, text_to_events
from aslcv.infer import transcript


def test_segments_split_on_long_gaps_and_ignore_short_dropouts():
    P = [True] * 10 + [False] * 2 + [True] * 10 + [False] * 20 + [True] * 10
    segs = find_segments(P, fps=30, max_gap_s=0.33, min_len_s=0.15)
    assert segs == [(0, 22), (42, 52)]          # the 2-frame dropout is bridged, the 20-frame gap splits


def test_tiny_segments_are_dropped():
    P = [True] * 3 + [False] * 30 + [True] * 10
    assert find_segments(P, fps=30, max_gap_s=0.33, min_len_s=0.15) == [(33, 43)]


def test_long_segments_are_split_at_the_calmest_frame():
    P = [True] * 600
    motion = [1.0] * 600
    motion[250] = 0.0                            # the stillest moment inside the search window
    segs = find_segments(P, fps=30, max_gap_s=0.33, min_len_s=0.15, max_len_s=10, motion=motion)
    assert segs[0] == (0, 251) and segs[1][0] == 251 and segs[-1][1] == 600


def test_text_to_events_spreads_letters_and_marks_words():
    ev = text_to_events("hi you", 1.0, 2.0)
    assert transcript(ev) == "HI YOU"
    letters = [e for e in ev if e.kind == "letter"]
    assert letters[0].t_commit >= 1.0 and letters[-1].t_commit <= 2.0
    assert all(a.t_commit < b.t_commit for a, b in zip(letters, letters[1:]))
    assert ev[-1].kind == "space" and ev[-1].t_commit == 2.0     # a segment ends a word


def test_fingerspelling_mask_marks_detected_stretch_with_padding():
    from types import SimpleNamespace
    from aslcv.seq_translate import fingerspelling_mask

    class Stub:                          # "fingerspelling" whenever the first hand is raised (y < 100)
        threshold = 0.5

        def prob(self, F):
            return (F[:, 65] < 100 / 360).astype(float)

    pts = lambda y: np.c_[np.full(21, 50.0), np.r_[y, np.full(8, y - 10), y - 20, np.full(11, y - 10)], np.zeros(21)]
    times = np.arange(90) / 30.0                                       # 3 s at 30 fps
    hands = [[SimpleNamespace(landmarks_px=pts(50.0 if 1.0 <= t < 2.0 else 300.0), handedness="Right")]
             for t in times]
    mask, segs = fingerspelling_mask(hands, times, (640, 360), Stub(), pad_s=0.2)
    assert len(segs) == 1 and abs(segs[0][0] - 1.0) < 0.1 and abs(segs[0][1] - 2.0) < 0.15
    assert mask[int(0.85 * 30)] and not mask[int(0.5 * 30)] and not mask[int(2.6 * 30)]


def test_signer_box_ignores_stray_detections():
    from aslcv.seq_translate import signer_box
    rng = np.random.default_rng(0)
    hand = lambda x, y: np.c_[rng.normal(x, 8, 21), rng.normal(y, 8, 21)]
    cluster = [hand(950 + rng.normal(0, 25), 560 + rng.normal(0, 25)) for _ in range(30)]
    strays = [hand(60, 400), hand(500, 100), hand(1200, 50)]
    box = signer_box(cluster + strays, [20.0] * 33, 1280, 720)
    assert box is not None
    x0, y0, x1, y1 = box
    assert 800 < x0 < 950 < x1 < 1150 and 300 < y0 < 560 < y1 <= 720


def test_signer_box_none_when_hands_fill_the_frame():
    from aslcv.seq_translate import signer_box
    rng = np.random.default_rng(1)
    pts = [np.c_[rng.normal(x, 5, 21), rng.normal(y, 5, 21)] for x, y in rng.uniform([0, 0], [1280, 720], (40, 2))]
    assert signer_box(pts, [150.0] * 40, 1280, 720) is None          # big hands everywhere: not an inset
