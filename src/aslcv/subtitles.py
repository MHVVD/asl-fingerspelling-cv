"""Video translator, part 1: letter/space events -> subtitle cues, SRT text and the running caption."""
from dataclasses import dataclass


@dataclass
class Cue:
    start: float
    end: float
    text: str


def events_to_cues(events, linger=1.0):
    """One cue per word. A word starts when its first letter started forming and ends at the
    space event that closed it, or `linger` seconds after its last letter if it is the last word."""
    cues, word, start, last = [], "", None, None
    for e in events:
        if e.kind == "letter":
            if not word:
                start = e.t_start
            word += e.value
            last = e.t_commit
        elif word:
            cues.append(Cue(start, e.t_commit, word))
            word = ""
    if word:
        cues.append(Cue(start, last + linger, word))
    for a, b in zip(cues, cues[1:]):          # never let a cue run into the next one
        a.end = min(a.end, b.start)
    return cues


def srt_time(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_srt(cues):
    return "".join(f"{i}\n{srt_time(c.start)} --> {srt_time(c.end)}\n{c.text}\n\n" for i, c in enumerate(cues, 1))


def caption_at(events, t, max_chars=32):
    """Text committed up to time t, for the burned-in caption. Long text keeps its tail,
    cut at a word boundary so the newest letters are always visible."""
    text = "".join(e.value for e in events if e.t_commit <= t).strip()
    if len(text) <= max_chars:
        return text
    tail = text[-max_chars:]
    cut = tail.find(" ")
    return tail[cut + 1:] if cut != -1 else tail
