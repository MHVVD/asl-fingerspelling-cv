import numpy as np

from aslcv.fsdetect import FEAT_DIM, SLOT_DIM, FSDetector, detect_features, smooth, to_segments
from aslcv.fsdetect_train import iou, mirror, segment_scores


def hand(x, y, size=20.0):
    pts = np.zeros((21, 3), np.float32)
    pts[:, 0], pts[:, 1] = x, y
    pts[9, :2] = x, y - size                       # middle-finger base: sets the hand size
    pts[8, :2] = x + 5, y - 2 * size
    return pts


def frames(*per_frame):
    H = np.full((len(per_frame), 2, 21, 3), np.nan, np.float32)
    sides = np.full((len(per_frame), 2), "", dtype="<U5")
    for t, hs in enumerate(per_frame):
        for k, (h, s) in enumerate(hs):
            H[t, k], sides[t, k] = h, s
    return H, sides


def test_upper_hand_goes_first_and_absent_slots_are_zero():
    H, sides = frames([(hand(100, 300), "Left"), (hand(200, 100), "Right")], [(hand(210, 110), "Right")])
    F = detect_features(H, sides, (640, 360))
    assert F.shape == (2, FEAT_DIM)
    assert F[0, 0] == 1 and F[0, SLOT_DIM] == 1
    assert np.isclose(F[0, 64], 200 / 640) and np.isclose(F[0, 65], 100 / 360)     # upper = the y=100 hand
    assert np.isclose(F[0, 66], 20 / 360)
    assert np.allclose(F[0, -2:], [(100 - 200) / 20, (300 - 100) / 20])           # lower hand, in hand sizes
    assert F[1, SLOT_DIM] == 0 and np.allclose(F[1, -2:], 0)
    assert np.allclose(F[1, 67:69], [10 / 20, 10 / 20])                            # wrist velocity


def test_mirror_flips_x_only():
    H, sides = frames([(hand(100, 300), "Left"), (hand(200, 100), "Right")])
    F = detect_features(H, sides, (640, 360))
    M = mirror(F)
    assert np.isclose(M[0, 64], 1 - F[0, 64]) and M[0, 65] == F[0, 65]
    assert M[0, -2] == -F[0, -2] and M[0, -1] == F[0, -1]
    assert np.allclose(M[0, 1:64], F[0, 1:64])


def test_to_segments_bridges_gaps_and_drops_blips():
    times = np.arange(40) / 10.0
    p = np.zeros(40)
    p[5:15] = 1          # 1 s
    p[17:25] = 1         # after a 0.2 s gap -> merged
    p[33:34] = 1         # 0.1 s blip -> dropped
    segs = to_segments(p, times, 0.5, min_len_s=0.3, max_gap_s=0.3)
    assert len(segs) == 1 and segs[0][0] < 0.6 and segs[0][1] > 2.2


def test_smooth_keeps_length():
    assert len(smooth(np.arange(10.0))) == 10


def test_segment_scores():
    v = {"label": np.r_[np.zeros(10), np.ones(10), np.zeros(10)], "times": np.arange(30) / 10.0}
    s = segment_scores([v], [[(1.0, 1.9), (2.5, 2.9)]])
    assert s == {"clip_recall": 1.0, "segment_precision": 0.5, "clips": 1, "segments_detected": 2}
    assert iou((0, 2), (1, 3)) == 1 / 3


def test_numpy_detector_matches_torch(tmp_path):
    import torch
    from aslcv.seqtrain import SeqNet
    torch.manual_seed(0)
    net = SeqNet(FEAT_DIM, 1, hidden=8, layers=2).eval()
    x = np.random.default_rng(0).normal(size=(12, FEAT_DIM)).astype(np.float32)
    with torch.no_grad():
        ref = torch.sigmoid(net(torch.from_numpy(x[None]))[0, :, 0]).numpy()
    det = FSDetector.from_torch(net, threshold=0.4)
    assert np.allclose(det.prob(x), ref, atol=1e-5)
    det.save(tmp_path / "d.npz")
    back = FSDetector.load(tmp_path / "d.npz")
    assert back.threshold == 0.4 and np.allclose(back.prob(x), ref, atol=1e-5)
