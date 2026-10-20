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


def test_speed_resample_scales_length_and_hand_velocity():
    """Slowing a clip must also slow the hand: per-frame wrist velocity shrinks by the same factor (journal 12)."""
    from aslcv.fsw_train import resample_speed
    F = np.zeros((10, FEAT_DIM), np.float32)
    F[:, 63] = 0.4; F[:, 65] = 1.0                 # wrist moving 0.4 hand sizes per frame
    F[4, :] = 0.0                                   # one frame without a hand
    slow = resample_speed(F, 0.5)                   # half speed
    assert len(slow) == 20
    moving = slow[slow[:, 65] == 1]
    assert np.allclose(moving[:, 63], 0.2)          # half the distance per frame
    assert np.all(slow[slow[:, 65] == 0][:, :65] == 0)
    fast = resample_speed(F, 2.0)
    assert len(fast) == 5 and np.allclose(fast[fast[:, 65] == 1][:, 63], 0.8)


def test_repeat_symbol_encoding_and_decoding():
    from aslcv.seqmodel import encode_repeats, expand_repeats
    assert encode_repeats("mall") == "mal+"
    assert encode_repeats("bookkeeper") == "bo+k+e+per"
    assert encode_repeats("aaa") == "a+"            # triples are rare; treated as a double
    assert encode_repeats("ab") == "ab"
    assert expand_repeats("mal+") == "mall"
    assert expand_repeats("+ab+") == "abb"          # a leading + has nothing to repeat
    assert expand_repeats("a +") == "a "            # never repeat a space


def test_decode_expands_repeat_symbol():
    v = Vocab("ab+")
    ids = [v.index[c] for c in "a"] + [0] + [v.index["+"]] + [v.index["b"]]
    logits = np.full((len(ids), v.size), -5.0); logits[np.arange(len(ids)), ids] = 5.0
    from aslcv.seqmodel import SeqModel
    assert ctc_greedy_decode(logits, v) == "a+b"
    m = SeqModel.__new__(SeqModel); m.vocab = v; m.logits = lambda f: logits
    assert m.decode(None) == "aab"
