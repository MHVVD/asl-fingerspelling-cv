import numpy as np
import pytest
from aslcv.seqmodel import ctc_greedy_decode, sequence_features, FEAT_DIM, Vocab


def test_vocab_roundtrip_and_blank():
    v = Vocab("abc")
    assert v.blank == 0 and v.size == 4
    assert v.decode(v.encode("cab")) == "cab"


def test_ctc_greedy_collapses_repeats_and_drops_blanks():
    v = Vocab("ab")
    frames = [0, 1, 1, 0, 1, 2, 2, 0]            # - a a - a b b -
    logits = np.full((len(frames), v.size), -5.0)
    logits[np.arange(len(frames)), frames] = 5.0
    assert ctc_greedy_decode(logits, v) == "aab"  # blank between the a's keeps both


def test_sequence_features_shape_presence_and_motion():
    hand = np.zeros((21, 3)); hand[9] = [0, -100, 0]; hand[8] = [30, -180, 0]
    seq = [hand + [200, 300, 0], None, hand + [250, 300, 0]]   # second frame: no hand
    F = sequence_features(seq)
    assert F.shape == (3, FEAT_DIM)
    assert F[0, -1] == 1 and F[1, -1] == 0 and np.all(F[1, :-1] == 0)
    # wrist moved 50 px = 0.5 hand sizes to the right since the last seen hand
    assert np.isclose(F[2, 63], 0.5) and np.isclose(F[2, 64], 0.0)


def test_numpy_model_matches_torch():
    torch = pytest.importorskip("torch")
    from aslcv.seqtrain import SeqNet
    from aslcv.seqmodel import SeqModel
    torch.manual_seed(0)
    net = SeqNet(n_in=FEAT_DIM, n_out=5, hidden=16, layers=2).eval()
    x = np.random.default_rng(0).normal(size=(37, FEAT_DIM)).astype(np.float32)
    with torch.no_grad():
        ref = net(torch.from_numpy(x)[None]).numpy()[0]
    np_model = SeqModel.from_torch(net, Vocab("abcd"))
    assert np.allclose(np_model.logits(x), ref, atol=1e-4)
