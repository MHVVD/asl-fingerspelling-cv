import numpy as np
from aslcv.landmarks import Hand
from aslcv.extract_fsw import select_signing_track, pack_sequences, unpack_sequence, letter_accuracy, clean_label


def hand(cx, cy, size=40.0, w=640, h=360):
    px = np.zeros((21, 3)); px[:, 0] = cx; px[:, 1] = cy
    px[9, 1] = cy - size                     # middle knuckle one hand-size above the wrist
    px[8, :2] = [cx + 10, cy - 2 * size]
    return Hand(px / [w, h, w], px, "Right", 0.9)


def test_raised_track_is_chosen_for_the_whole_clip():
    # frames: both hands; then only the resting hand is detected (signing hand missed)
    up, down = hand(400, 120), hand(150, 320)
    frames = [[up, down], [down, up], [down]]
    chosen = select_signing_track(frames)
    assert chosen[0] is up and chosen[1] is up
    assert chosen[2] is None              # do NOT switch to the resting hand when the signing hand is missed


def test_tracks_follow_motion_not_list_order():
    a0, a1 = hand(400, 120), hand(410, 125)     # signing hand moving slightly
    b0, b1 = hand(150, 320), hand(152, 321)
    chosen = select_signing_track([[b0, a0], [a1, b1]])
    assert chosen == [a0, a1]


def test_annotated_box_overrides_height():
    up, down = hand(400, 120), hand(150, 320)
    frames = [[up, down], [up, down]]
    boxes = [[(110, 220, 190, 340)], None]       # annotation: the LOWER hand is the signing one
    chosen = select_signing_track(frames, boxes)
    assert chosen == [down, down]


def test_single_hand_clip_and_empty_frames():
    h = hand(300, 200)
    assert select_signing_track([[h], [], [h]]) == [h, None, h]


def test_pack_roundtrip_with_missing_frames():
    seqs = [[np.ones((21, 3)), None, 2 * np.ones((21, 3))], [None]]
    packed = pack_sequences(seqs, labels=["ab", "c"])
    out = unpack_sequence(packed, 0)
    assert len(out) == 3 and out[1] is None and np.allclose(out[2], 2)
    assert unpack_sequence(packed, 1) == [None]


def test_letter_accuracy_is_one_minus_total_edits_over_total_letters():
    assert letter_accuracy(["hello", "ab"], ["hello", "ab"]) == 1.0
    assert abs(letter_accuracy(["hello", "ab"], ["helo", "b"]) - (1 - 2 / 7)) < 1e-9


def test_clean_label_keeps_vocab_only():
    assert clean_label("asl lit.") == "asl lit"
    assert clean_label("r&d") == "rd"
    assert clean_label("don't") == "don't"
