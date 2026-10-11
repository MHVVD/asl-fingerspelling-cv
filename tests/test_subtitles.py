from aslcv.infer import Event
from aslcv.subtitles import events_to_cues, format_srt, caption_at, srt_time


def L(ch, t0, t1):
    return Event("letter", ch, t0, t1, 0.9)


def S(t):
    return Event("space", " ", t, t)


EVENTS = [L("H", 1.0, 1.2), L("I", 1.5, 1.7), S(2.3), L("Y", 3.0, 3.2), L("O", 3.5, 3.7), L("U", 4.0, 4.2)]


def test_words_become_cues_with_times():
    cues = events_to_cues(EVENTS, linger=1.0)
    assert [c.text for c in cues] == ["HI", "YOU"]
    assert cues[0].start == 1.0 and cues[0].end == 2.3        # ends when the word ends (space)
    assert cues[1].start == 3.0 and cues[1].end == 5.2        # last word: last letter + linger


def test_cue_end_never_overlaps_next_cue():
    cues = events_to_cues([L("A", 1.0, 1.2), S(1.4), L("B", 1.5, 1.7)], linger=3.0)
    assert cues[0].end <= cues[1].start


def test_srt_format():
    assert srt_time(3661.5) == "01:01:01,500"
    srt = format_srt(events_to_cues(EVENTS, linger=1.0))
    assert srt.startswith("1\n00:00:01,000 --> 00:00:02,300\nHI\n\n2\n00:00:03,000 --> 00:00:05,200\nYOU\n")


def test_caption_grows_as_letters_commit():
    assert caption_at(EVENTS, 1.1) == ""
    assert caption_at(EVENTS, 1.8) == "HI"
    assert caption_at(EVENTS, 3.8) == "HI YO"


def test_caption_keeps_the_tail_on_a_word_boundary():
    ev = [L(c, i, i + 0.1) if c != " " else S(i) for i, c in enumerate("THE QUICK BROWN FOX")]
    assert caption_at(ev, 100, max_chars=10) == "BROWN FOX"
