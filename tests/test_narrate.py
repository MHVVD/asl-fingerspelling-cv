"""narrate.py: timing placement and the overlap policy (pure), plus an ffmpeg round trip when ffmpeg exists."""
import shutil
import subprocess

import numpy as np
import pytest

from aslcv import narrate
from aslcv.narrate import RATE, NarrationError, mix, plan_timeline
from aslcv.tts import TTSError


def test_clips_start_at_their_segment_when_there_is_room():
    p = plan_timeline([(1.0, 0.5, "a"), (3.0, 0.5, "b")])
    assert [x.start_s for x in p] == [1.0, 3.0] and all(x.delay_s == 0 for x in p)


def test_overlapping_clip_is_queued_not_trimmed():
    p = plan_timeline([(1.0, 1.5, "campylobacter"), (2.0, 0.5, "bed")], min_gap_s=0.1)
    assert p[1].start_s == pytest.approx(2.6) and p[1].delay_s == pytest.approx(0.6)
    assert p[1].duration_s == 0.5                                    # full length kept
    for a, b in zip(p, p[1:]):
        assert b.start_s >= a.start_s + a.duration_s


def test_queue_delay_disappears_after_a_pause():
    p = plan_timeline([(0.0, 2.0, "a"), (1.0, 2.0, "b"), (10.0, 1.0, "c")], min_gap_s=0.0)
    assert [x.start_s for x in p] == [0.0, 2.0, 10.0]


def test_unsorted_input_is_placed_in_time_order():
    p = plan_timeline([(5.0, 1.0, "late"), (1.0, 1.0, "early")])
    assert [x.text for x in p] == ["early", "late"]


def test_mix_puts_samples_at_the_right_time_and_covers_the_video():
    clips = [np.ones(RATE // 10, np.float32) * 0.5]
    track = mix(clips, plan_timeline([(1.0, 0.1, "a")]), total_s=3.0)
    assert len(track) == 3 * RATE
    assert track[RATE - 1] == 0 and track[RATE] == 0.5 and track[RATE + RATE // 10] == 0


def test_mix_extends_past_the_video_if_a_queued_clip_runs_over():
    clips = [np.ones(RATE, np.float32) * 0.1]
    assert len(mix(clips, plan_timeline([(2.5, 1.0, "a")]), total_s=3.0)) == int(3.5 * RATE)


class FakeSynth:
    def __init__(self, fail_on=None):
        self.texts, self.fail_on = [], fail_on

    def synthesize(self, text):
        if text == self.fail_on:
            raise TTSError("boom")
        self.texts.append(text)
        return text.encode()


def fake_decode(data):
    return np.ones(len(data) * RATE // 10, np.float32) * 0.1        # 0.1 s per character


def test_narrate_places_each_segment_and_reports_delays():
    written = {}
    segs = [{"start_s": 1.0, "text": "bed"}, {"start_s": 1.1, "text": "fsis"}, {"start_s": 9.0, "text": "  "}]
    r = narrate.narrate(segs, FakeSynth(), 12.0, "x.mp3", decode=fake_decode,
                        write=lambda s, out: written.update(out=out, n=len(s)))
    assert written == {"out": "x.mp3", "n": 12 * RATE}
    assert [s["text"] for s in r["spoken"]] == ["bed", "fsis"]       # the blank segment is skipped
    assert r["queued"] == 1 and r["max_delay_s"] == pytest.approx(0.3, abs=0.01)


def test_api_failure_writes_nothing():
    written = []
    with pytest.raises(TTSError):
        narrate.narrate([{"start_s": 0, "text": "ok"}, {"start_s": 2, "text": "bad"}], FakeSynth(fail_on="bad"),
                        5.0, "x.mp3", decode=fake_decode, write=lambda s, out: written.append(out))
    assert written == []


def test_missing_ffmpeg_is_a_friendly_error(monkeypatch):
    monkeypatch.setattr(narrate.shutil, "which", lambda name: None)
    with pytest.raises(NarrationError, match="apt install ffmpeg"):
        narrate.require_ffmpeg()


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_ffmpeg_round_trip_and_mux(tmp_path):
    tone = (0.3 * np.sin(2 * np.pi * 440 * np.arange(RATE) / RATE)).astype(np.float32)
    mp3 = tmp_path / "n_spoken.mp3"
    narrate.write_mp3(tone, mp3)
    back = narrate.decode_mp3(mp3.read_bytes())
    assert abs(len(back) - RATE) < RATE * 0.1                        # mp3 adds a little padding
    video = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=1",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)], check=True)
    out = tmp_path / "v_captioned_spoken.mp4"
    narrate.mux(video, mp3, out)
    streams = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0",
                              str(out)], capture_output=True, text=True).stdout.split()
    assert streams == ["video", "audio"]
    assert not list(tmp_path.glob("*.part*"))


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_failed_mux_leaves_no_file(tmp_path):
    out = tmp_path / "out.mp4"
    with pytest.raises(NarrationError):
        narrate.mux(tmp_path / "missing.mp4", tmp_path / "missing.mp3", out)
    assert not out.exists() and not list(tmp_path.glob("*.part*"))
