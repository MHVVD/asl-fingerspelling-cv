import numpy as np
from aslcv.reference import reference_pose
from aslcv.features import normalize_landmarks
from aslcv.hints import hand_measures, build_table, hints
from aslcv.tutor import Planner, Attempt


def feat(letter, rot_deg=0.0):
    f = normalize_landmarks(reference_pose(letter)).reshape(21, 3)
    a = np.radians(rot_deg)
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    return (f @ R.T).ravel()


def table_for(letters):
    X = np.stack([feat(k, r) for k in letters for r in (-10, 0, 10)])
    y = np.array([k for k in letters for _ in range(3)])
    return build_table(X, y)


# ---- hints ----
def test_measures_tell_straight_from_curled():
    m = hand_measures(feat("L"))
    assert m["fingers"]["index"] < 60 and m["fingers"]["middle"] > 150
    assert m["thumb_out"] > hand_measures(feat("A"))["thumb_out"]
    assert m["direction"] == "up" and hand_measures(feat("G"))["direction"] == "side"


def test_correct_sign_gets_no_hints():
    t = table_for("BL")
    assert hints(hand_measures(feat("L")), t["L"]) == []


def test_wrong_fingers_get_specific_hints():
    t = table_for("BL")
    h = hints(hand_measures(feat("B")), t["L"])             # showing B, target L
    assert "Curl your middle finger down" in h and len(h) <= 2


def test_orientation_hint_comes_first():
    t = table_for("GL")
    h = hints(hand_measures(feat("L")), t["G"])
    assert h[0] == "Point your fingers sideways"


def test_classifier_hint_when_it_looks_like_another_letter():
    t = table_for("BL")
    h = hints(hand_measures(feat("L")), t["L"], target="L", seen=("V", 0.9))
    assert h[-1] == "That looks like V"
    assert "That looks like" not in " ".join(hints(hand_measures(feat("L")), t["L"], target="L", seen=("L", 0.9)))


# ---- planner (spaced repetition) ----
def test_planner_never_repeats_and_respects_letter_set():
    p = Planner(letters="ABC", seed=0)
    seq = [p.next() for _ in range(30)]
    assert set(seq) <= set("ABC") and all(a != b for a, b in zip(seq, seq[1:]))


def test_planner_prefers_weak_letters(tmp_path):
    p = Planner(letters="ABCD", seed=1)
    for _ in range(10):
        p.record("A", True); p.record("B", True); p.record("C", True); p.record("D", False)
    picks = [p.next() for _ in range(300)]
    assert picks.count("D") > max(picks.count(k) for k in "ABC")
    p.save(tmp_path / "progress.json")
    q = Planner(letters="ABCD", path=tmp_path / "progress.json")
    assert q.stats["D"]["attempts"] == 10 and q.mastery("A") > q.mastery("D")


# ---- one attempt ----
def test_attempt_succeeds_after_holding_the_target():
    a = Attempt("L", hold_s=0.6, timeout_s=12)
    t = 0.0
    while a.result is None and t < 2:
        a.update(("L", 0.9), t); t += 1 / 30
    assert a.result == "success" and 0.6 <= t < 1.0


def test_attempt_hold_resets_when_sign_breaks():
    a = Attempt("L", hold_s=0.6)
    for i in range(10): a.update(("L", 0.9), i / 30)
    a.update(("B", 0.9), 10 / 30)
    assert a.progress == 0.0 and a.result is None


def test_low_confidence_does_not_count_and_timeout_fails():
    a = Attempt("L", hold_s=0.6, timeout_s=3, min_conf=0.6)
    for i in range(200):
        a.update(("L", 0.4), i / 30)
    assert a.result == "missed"


def test_spread_hint_tells_u_from_v():
    t = table_for("UV")
    assert t["V"]["spread"] == "apart" and t["U"]["spread"] == "together"
    assert hints(hand_measures(feat("U")), t["V"]) == ["Spread your index and middle fingers apart"]
    assert hints(hand_measures(feat("V")), t["U"]) == ["Hold your index and middle fingers together"]


def test_no_point_up_hint():
    """Direction follows the palm, not the fingers, so 'point up' would mislead e.g. a correct G."""
    t = table_for("GL")
    assert not any("Point your fingers up" in h for h in hints(hand_measures(feat("G")), t["L"]))
