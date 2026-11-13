import numpy as np

from aslcv.benchmark import same_word, score_video, substring_distance
from aslcv.landmarks import Hand
from aslcv.seq_translate import load_hands, save_hands


def test_substring_distance():
    assert substring_distance("karen", "xxaskarnyy") == 1          # "karn" -> one deletion
    assert substring_distance("fema", "zzfemazz") == 0
    assert substring_distance("hiv", "") == 3


def test_same_word_allows_plural():
    assert same_word("bacteria", "bacteria") and same_word("antibiotics", "antibiotic")
    assert not same_word("bus", "us")


def test_score_video_windows():
    terms = [{"t": 10.0, "word": "karen", "verdict": "spelled"}, {"t": 50.0, "word": "fema", "verdict": "unclear"},
             {"t": 80.0, "word": "harvey", "verdict": "spelled"}]
    captions = [(9.0, ["ask", "karen"]), (78.0, ["hurricane", "harvey"])]
    segments = [{"start_s": 12.0, "end_s": 14.0, "text": "askarn"},           # within [8, 18]: read
                {"start_s": 95.0, "end_s": 96.0, "text": "harvey"}]          # outside [78, 88]: not read
    spoken = [{"start_s": 12.0, "text": "karen"}, {"start_s": 95.0, "text": "harvey"}, {"start_s": 40.0, "text": "bus"}]
    r = score_video("v", terms, captions, segments, spoken, 0.5)
    assert (r["terms_spelled"], r["terms_read"], r["terms_spoken"]) == (2, 1, 1)
    assert (r["spoken"], r["spoken_correct"]) == (3, 1)        # "bus": no caption; "harvey": caption 17 s earlier


def test_hands_round_trip(tmp_path):
    h = Hand(np.random.rand(21, 3).astype(np.float32), np.random.rand(21, 3).astype(np.float32) * 100, "Left", 0.9)
    hp = {"hands": [[h], [], [h, h]], "times": [0.0, 0.033, 0.067], "size": (640, 360), "src_fps": 30.0, "step": 1,
          "fps": 30.0}
    save_hands(hp, tmp_path / "h.npz")
    back = load_hands(tmp_path / "h.npz")
    assert [len(x) for x in back["hands"]] == [1, 0, 2] and back["size"] == (640, 360)
    assert np.allclose(back["hands"][2][1].landmarks_px, h.landmarks_px) and back["hands"][0][0].handedness == "Left"
    assert abs(back["hands"][0][0].score - 0.9) < 1e-6
    # each hand owns its points: a view into the loaded array would keep a whole copy alive per hand (journal 18)
    assert all(x.landmarks_px.base is None and x.landmarks.base is None for hs in back["hands"] for x in hs)
