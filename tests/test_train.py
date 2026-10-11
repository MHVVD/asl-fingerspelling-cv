import numpy as np
from aslcv.train import split_by_session, augment


def test_split_keeps_sessions_whole():
    sessions = np.array(["a:b0"] * 5 + ["a:b1"] * 5 + ["a:b5"] * 5)
    tr, te = split_by_session(sessions, test_sessions={"a:b5"})
    assert set(sessions[tr]) == {"a:b0", "a:b1"} and set(sessions[te]) == {"a:b5"}
    assert len(set(tr) & set(te)) == 0


def test_augment_keeps_wrist_at_origin_and_adds_copies():
    rng = np.random.default_rng(0)
    X = rng.random((10, 63)); X.reshape(10, 21, 3)[:, 0] = 0
    y = np.array(list("ABCDEFGHIJ"))
    Xa, ya = augment(X, y, copies=2, rng=rng)
    assert Xa.shape == (30, 63) and len(ya) == 30
    assert np.allclose(Xa.reshape(-1, 21, 3)[:, 0], 0, atol=1e-6)
    assert (ya[:10] == y).all() and np.allclose(Xa[:10], X)  # originals kept unchanged


def test_make_split_trains_on_chosen_sources_and_tests_on_session_from_any_source():
    from aslcv.train import make_split
    source = np.array(["public"] * 4 + ["recorded"] * 4)
    sessions = np.array(["p:b0", "p:b0", "p:b5", "p:b5", "s1", "s1", "s2", "s2"])
    # public-only model, tested on my session s2
    tr, te = make_split(source, sessions, train_sources={"public"}, test_sessions={"s2"})
    assert set(tr) == {0, 1, 2, 3} and set(te) == {6, 7}
    # my data only
    tr, te = make_split(source, sessions, train_sources={"recorded"}, test_sessions={"s2"})
    assert set(tr) == {4, 5} and set(te) == {6, 7}
    # both, test sessions never leak into training
    tr, te = make_split(source, sessions, train_sources={"public", "recorded"}, test_sessions={"s2"})
    assert set(tr) == {0, 1, 2, 3, 4, 5} and not set(tr) & set(te)
