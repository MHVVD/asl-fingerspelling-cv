import numpy as np
from aslcv.infer import LetterStream, transcript

CLASSES = list("ABLO")


def p(letter, conf=0.95):
    """A probability vector peaked on `letter`."""
    v = np.full(len(CLASSES), (1 - conf) / (len(CLASSES) - 1))
    v[CLASSES.index(letter)] = conf
    return v


def run(stream, frames, fps=30):
    events = []
    for i, f in enumerate(frames):
        probs, motion = (f if isinstance(f, tuple) else (f, 0.0))
        events += stream.update(probs, t=i / fps, motion=motion)
    return events


def letters(events):
    return "".join(e.value for e in events)


def test_steady_letter_emitted_once():
    s = LetterStream(CLASSES)
    ev = run(s, [p("A")] * 40)
    assert letters(ev) == "A"


def test_needs_hold_before_committing():
    s = LetterStream(CLASSES, hold=6)
    assert run(s, [p("A")] * 3) == []


def test_flicker_is_ignored():
    s = LetterStream(CLASSES)
    ev = run(s, [p("A"), p("B")] * 20)
    assert ev == []


def test_transition_between_letters():
    s = LetterStream(CLASSES)
    ev = run(s, [p("A")] * 20 + [p("B")] * 20)
    assert letters(ev) == "AB"


def test_double_letter_needs_release_by_confidence_dip():
    s = LetterStream(CLASSES)
    uncertain = np.full(len(CLASSES), 1 / len(CLASSES))
    ev = run(s, [p("L")] * 20 + [uncertain] * 6 + [p("L")] * 20)
    assert letters(ev) == "LL"


def test_double_letter_by_motion_bounce():
    s = LetterStream(CLASSES, motion_release=0.15)
    frames = [p("L")] * 20 + [(p("L"), 0.4)] + [p("L")] * 20
    assert letters(run(s, frames)) == "LL"


def test_motion_release_can_be_disabled():
    s = LetterStream(CLASSES, motion_release=None)
    frames = [p("L")] * 20 + [(p("L"), 0.4)] + [p("L")] * 20
    assert letters(run(s, frames)) == "L"


def test_no_hand_gap_inserts_one_space_between_words():
    s = LetterStream(CLASSES, gap=10)
    ev = run(s, [None] * 15 + [p("A")] * 20 + [None] * 30 + [p("B")] * 20 + [None] * 30)
    assert [e.kind for e in ev] == ["letter", "space", "letter", "space"]
    assert transcript(ev) == "A B"


def test_short_dropout_does_not_break_a_letter_or_add_space():
    s = LetterStream(CLASSES, gap=10)
    ev = run(s, [p("A")] * 4 + [None] * 2 + [p("A")] * 20)
    assert [e.kind for e in ev] == ["letter"] and letters(ev) == "A"


def test_same_letter_allowed_again_after_word_gap():
    s = LetterStream(CLASSES, gap=10)
    ev = run(s, [p("A")] * 20 + [None] * 20 + [p("A")] * 20)
    assert transcript(ev) == "A A"


def test_event_timestamps():
    s = LetterStream(CLASSES, hold=6)
    ev = run(s, [None] * 30 + [p("O")] * 30, fps=30)
    e = ev[0]
    assert 0.95 <= e.t_start <= 1.05      # O appears at frame 30 = 1.0 s
    assert e.t_start < e.t_commit < 1.5   # committed a few frames later


def test_current_display_state():
    s = LetterStream(CLASSES)
    run(s, [p("B")] * 20)
    letter, conf = s.current
    assert letter == "B" and conf > 0.9


def test_thresholds_defined_in_seconds_scale_with_fps():
    s30 = LetterStream.for_fps(CLASSES, fps=30)
    s60 = LetterStream.for_fps(CLASSES, fps=60)
    assert (s30.hold, s30.gap, s30.short_dropout) == (6, 10, 3)      # same as the tuned defaults
    assert (s60.hold, s60.gap, s60.short_dropout) == (12, 20, 6)
    assert LetterStream.for_fps(CLASSES, fps=5).hold >= 2            # never a single-frame commit
