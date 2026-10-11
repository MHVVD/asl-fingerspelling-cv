import numpy as np
from sklearn.neural_network import MLPClassifier
from aslcv.model import LetterModel


def _toy(n=300, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.choice(list("ABC"), n)
    centers = {"A": 0.0, "B": 2.0, "C": 4.0}
    X = rng.normal(0, 0.3, (n, 63)) + np.array([centers[c] for c in y])[:, None]
    return X.astype(np.float32), y


def test_numpy_model_matches_sklearn(tmp_path):
    X, y = _toy()
    clf = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=300, random_state=0).fit(X, y)
    LetterModel.from_sklearn(clf).save(tmp_path / "m.npz")
    m = LetterModel.load(tmp_path / "m.npz")
    assert np.allclose(m.predict_proba(X), clf.predict_proba(X), atol=1e-5)
    assert (m.predict(X) == clf.predict(X)).all()


def test_single_vector_input(tmp_path):
    X, y = _toy()
    clf = MLPClassifier(hidden_layer_sizes=(8,), max_iter=200, random_state=0).fit(X, y)
    m = LetterModel.from_sklearn(clf)
    probs = m.predict_proba(X[0])
    assert probs.shape == (3,) and abs(probs.sum() - 1) < 1e-6
    letter, conf = m.top(X[0])
    assert letter in "ABC" and 0 <= conf <= 1


def test_model_does_not_import_torch():
    import subprocess, sys
    code = "import sys; import aslcv.model; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, env={"PYTHONPATH": "src"})
