"""--speak glue in translate.py: which segments get spoken, and a full speak() run with fakes."""
import json

import numpy as np

from aslcv import narrate, translate
from aslcv.seq_translate import segment_confidence
from aslcv.subtitles import Cue


def seg(t, text, conf):
    return {"start_s": t, "end_s": t + 1, "text": text, "confidence": conf}


VOCAB = {"campylobacter", "bacteria", "strain", "human", "jejuni"}


def test_low_confidence_segments_are_skipped():
    r = {"segments": [seg(1, "campylobacter", 0.9), seg(5, "lim", 0.5), seg(7, " ", 0.9)]}
    keep, skipped = translate.speech_segments(r, "sequence", min_confidence=0.6)
    assert keep == [{"start_s": 1, "text": "campylobacter"}]
    assert skipped == [{"start_s": 5, "text": "lim", "reason": "confidence 0.5"}]      # blanks are neither


def test_only_spotted_words_are_spoken_each_at_its_time():
    r = {"segments": [{"start_s": 10.0, "end_s": 13.0, "text": "strainhumanicampylobacterjejunis", "confidence": 0.8},
                      seg(20, "reveidisemittar", 0.7)]}
    keep, skipped = translate.speech_segments(r, "sequence", 0.6, VOCAB)
    assert [k["text"] for k in keep] == ["strain", "campylobacter", "jejuni"]       # "human" < 6 letters
    assert keep[0]["start_s"] == 10.0 and 10.0 < keep[1]["start_s"] < keep[2]["start_s"] < 13.0
    assert skipped == [{"start_s": 20, "text": "reveidisemittar", "reason": "no known word"}]


def test_frame_engine_cues_are_spotted_too():
    r = {"cues": [Cue(1.0, 2.0, "BACTERIA"), Cue(3.0, 4.0, "XQZT")]}
    keep, skipped = translate.speech_segments(r, "frame", min_confidence=0.9, vocab=VOCAB)
    assert keep == [{"start_s": 1.0, "text": "bacteria"}] and skipped[0]["reason"] == "no known word"


def test_segment_confidence_ignores_blank_frames():
    blank = 0
    logits = np.log(np.array([[0.9, 0.05, 0.05], [0.2, 0.8, 0.0001], [0.4, 0.6, 0.0001]]))
    assert segment_confidence(logits, blank) == 0.7
    assert segment_confidence(np.log(np.array([[0.9, 0.1, 1e-9]])), blank) == 0.0


class FakeSynth:
    voice_id, model_id = "v", "m"

    def __init__(self):
        self.stats = {"requests": 0, "cache_hits": 0, "characters": 0}

    def synthesize(self, text):
        self.stats["requests"] += 1
        self.stats["characters"] += len(text)
        return text.encode()


def test_speak_writes_report_and_lowercases_text(tmp_path, monkeypatch):
    calls = {}
    monkeypatch.setattr(narrate, "video_duration", lambda v: 10.0)
    monkeypatch.setattr(narrate, "decode_mp3", lambda d: np.zeros(len(d) * 4410, np.float32))
    monkeypatch.setattr(narrate, "write_mp3", lambda s, out: calls.setdefault("mp3", out))
    monkeypatch.setattr(narrate, "mux", lambda v, a, out: calls.setdefault("mux", (v, a, out)))
    synth = FakeSynth()
    r = {"segments": [seg(1, "FSIS", 0.9), seg(4, "zzzz", 0.2)]}
    rep, mp3, mp4 = translate.speak(r, "sequence", synth, "in.mp4", tmp_path, "clip", 0.5)
    assert mp3.name == "clip_spoken.mp3" and mp4.name == "clip_captioned_spoken.mp4"
    assert calls["mux"] == ("in.mp4", mp3, mp4)
    assert rep["clips"] == 1 and rep["characters"] == 4 and rep["skipped"][0]["text"] == "zzzz"
    saved = json.loads((tmp_path / "clip_spoken.json").read_text())
    assert saved["spoken"][0]["text"] == "fsis"


def test_speak_with_nothing_to_say_writes_nothing(tmp_path):
    rep, mp3, mp4 = translate.speak({"segments": [seg(1, "zzzz", 0.2)]}, "sequence", FakeSynth(), "in.mp4",
                                    tmp_path, "clip", 0.5)
    assert rep["clips"] == 0 and mp3 is None and not list(tmp_path.iterdir())
