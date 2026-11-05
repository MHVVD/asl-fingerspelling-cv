"""Narration track for the video translator (journal 16): spoken words placed at their segment's start time.

    segments (start_s, text) -> synthesize each text -> decode -> place on one timeline -> <name>_spoken.mp3
                                                                                        -> mux -> <name>_captioned_spoken.mp4

Text-to-speech goes through any object with `synthesize(text) -> MP3 bytes` (tts.Synthesizer), so this module
knows nothing about ElevenLabs. ffmpeg decodes the clips and writes the outputs; every output is written to a
`.part` file and renamed only when complete, so a failure never leaves a corrupt file behind.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import shutil
import subprocess
import numpy as np

RATE = 44100
MIN_GAP_S = 0.1          # silence kept between two queued clips


class NarrationError(Exception):
    """A failure that should be shown to the user as one message, without a traceback."""


@dataclass
class Placement:
    text: str
    wanted_s: float      # segment start: where the word should be heard
    start_s: float       # where it is placed
    duration_s: float

    @property
    def delay_s(self):
        return self.start_s - self.wanted_s


def plan_timeline(items, min_gap_s=MIN_GAP_S):
    """[(wanted_start_s, duration_s, text)] -> [Placement], never overlapping.

    Overlap policy: QUEUE. A clip that would start before the previous one has finished waits until it has.
    Trimming would cut words in half, and speeding up distorts the voice and still overlaps when the gap is
    tiny. A spoken word (~0.5-1 s) is shorter than fingerspelling it (~2 s), so delays stay small and vanish
    at the next pause; the delays are reported so they can be checked.
    """
    out, free_at = [], float("-inf")
    for wanted, duration, text in sorted(items, key=lambda x: x[0]):
        start = max(wanted, free_at)
        out.append(Placement(text, wanted, start, duration))
        free_at = start + duration + min_gap_s
    return out


def mix(clips, placements, total_s, rate=RATE):
    """Mono float32 track of max(total_s, end of last clip), clips written at their placement."""
    end = max([total_s] + [p.start_s + len(c) / rate for c, p in zip(clips, placements)])
    track = np.zeros(int(np.ceil(end * rate)), np.float32)
    for c, p in zip(clips, placements):
        i = int(round(p.start_s * rate))
        track[i:i + len(c)] += c[:len(track) - i]
    return np.clip(track, -1.0, 1.0)


def require_ffmpeg():
    if not shutil.which("ffmpeg"):
        raise NarrationError("ffmpeg is needed for --speak but was not found. Install it: sudo apt install ffmpeg "
                             "(macOS: brew install ffmpeg)")


def _ffmpeg(args, data=None):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], input=data,
                       capture_output=True)
    if r.returncode != 0:
        raise NarrationError(f"ffmpeg failed: {r.stderr.decode(errors='ignore').strip()[-300:]}")
    return r.stdout


def decode_mp3(data, rate=RATE):
    """MP3 bytes -> mono float32 samples at `rate`."""
    pcm = _ffmpeg(["-i", "pipe:0", "-f", "s16le", "-ac", "1", "-ar", str(rate), "pipe:1"], data)
    return np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0


def _part(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(f"{path.stem}.part{path.suffix}")        # keep the extension: ffmpeg picks the format by it


def write_mp3(samples, out, rate=RATE):
    part = _part(out)
    pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes()
    try:
        _ffmpeg(["-f", "s16le", "-ar", str(rate), "-ac", "1", "-i", "pipe:0", "-c:a", "libmp3lame", "-q:a", "3",
                 str(part)], pcm)
        part.replace(out)
    finally:
        part.unlink(missing_ok=True)


def mux(video, audio, out):
    """Video stream copied as-is + the narration as its only audio (the original soundtrack is replaced so it
    does not talk over the narration)."""
    part = _part(out)
    try:
        _ffmpeg(["-i", str(video), "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                 "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(part)])
        part.replace(out)
    finally:
        part.unlink(missing_ok=True)


def video_duration(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def narrate(segments, synthesizer, total_s, out_mp3, decode=decode_mp3, write=write_mp3):
    """segments: [{"start_s", "text"}] (already filtered) -> writes out_mp3, returns a report.
    All clips are synthesized and decoded before anything is written, so an API error leaves no output."""
    texts = sorted((s["start_s"], s["text"].strip()) for s in segments if s["text"].strip())  # plan_timeline's order
    clips = [decode(synthesizer.synthesize(text)) for _, text in texts]
    placements = plan_timeline([(t, len(c) / RATE, text) for (t, text), c in zip(texts, clips)])
    write(mix(clips, placements, total_s), out_mp3)
    delays = [p.delay_s for p in placements]
    return {"spoken": [asdict(p) for p in placements], "clips": len(placements),
            "queued": sum(d > 1e-6 for d in delays), "max_delay_s": round(max(delays, default=0.0), 2),
            "speech_s": round(sum(p.duration_s for p in placements), 2)}
