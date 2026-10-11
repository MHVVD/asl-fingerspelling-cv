import numpy as np
from aslcv.dataset import Recorder, load_dataset
from aslcv.features import FEATURE_DIM


def feat(v=0.0):
    return np.full(FEATURE_DIM, v)


def test_no_samples_when_not_recording(tmp_path):
    r = Recorder(tmp_path, session="s1", target=3)
    r.select("A")
    assert r.add(feat()) is False
    assert r.counts["A"] == 0


def test_records_until_target_then_stops(tmp_path):
    r = Recorder(tmp_path, session="s1", target=3)
    r.select("A")
    r.toggle()
    assert [r.add(feat(i)) for i in range(5)] == [True, True, True, False, False]
    assert r.counts["A"] == 3
    assert r.recording is False  # auto-stops at target


def test_save_and_load_roundtrip_with_session_ids(tmp_path):
    r1 = Recorder(tmp_path, session="s1", target=2)
    r1.select("A"); r1.toggle(); r1.add(feat(1)); r1.add(feat(2))
    r1.select("B"); r1.toggle(); r1.add(feat(3))
    r1.save()
    r2 = Recorder(tmp_path, session="s2", target=1)
    r2.select("C"); r2.toggle(); r2.add(feat(4))
    r2.save()
    X, y, sessions = load_dataset(tmp_path)
    assert X.shape == (4, FEATURE_DIM)
    assert sorted(y.tolist()) == ["A", "A", "B", "C"]
    assert set(sessions.tolist()) == {"s1", "s2"}


def test_undo_removes_last_batch_of_current_letter(tmp_path):
    r = Recorder(tmp_path, session="s1", target=5)
    r.select("A"); r.toggle(); r.add(feat()); r.add(feat())
    r.toggle()
    r.undo()
    assert r.counts["A"] == 0


def test_invalid_letter_rejected(tmp_path):
    r = Recorder(tmp_path, session="s1")
    try:
        r.select("1")
    except ValueError:
        return
    assert False, "expected ValueError"


def test_load_dataset_accepts_per_sample_session_arrays(tmp_path):
    """Public-data files store one session id per row; recorder files store a single id."""
    np.savez(tmp_path / "public.npz", X=np.zeros((3, FEATURE_DIM)), y=np.array(["A", "A", "B"]),
             session=np.array(["p:b0", "p:b0", "p:b1"]))
    X, y, sessions = load_dataset(tmp_path)
    assert sessions.tolist() == ["p:b0", "p:b0", "p:b1"]


def test_raw_landmarks_and_model_label_are_saved_for_auditing(tmp_path):
    r = Recorder(tmp_path, session="s1", target=2)
    r.select("P"); r.toggle()
    r.add(feat(1), raw=np.ones((21, 3)), model_label="Left")
    r.add(feat(2), raw=np.zeros((21, 3)), model_label="Right")
    d = np.load(r.save())
    assert d["raw"].shape == (2, 21, 3)
    assert d["model_hand"].tolist() == ["Left", "Right"]


def test_hand_mismatch_warns_only_when_most_recent_frames_disagree():
    from aslcv.record import HandMismatch
    m = HandMismatch(declared="Right", window=10, share=0.8)
    for _ in range(10):
        m.update("Right")
    assert m.warning is False
    for _ in range(3):
        m.update("Left")            # a few odd labels (e.g. P/Q) must not warn
    assert m.warning is False
    for _ in range(10):
        m.update("Left")            # consistently the other hand: warn
    assert m.warning is True
