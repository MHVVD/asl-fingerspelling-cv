"""Which decoded text is worth speaking (journal 16).

The sequence model transcribes *all* signing as letters, so a segment mixes real fingerspelled words with letter
strings decoded from lexical signs: "strainhumanicampylobacterjejunis". Neither the model's confidence nor a
character-model score separates the two at segment level (journal 16), so --speak by default speaks only the
words it can find in the decoded text, from a vocabulary (the system word list plus an optional --vocab file of
topic words): a segment that is exactly one known word, or words of at least MIN_LEN letters inside a longer
one, each at its estimated time inside the segment.
"""
from pathlib import Path
import re

SYSTEM_WORDS = Path("/usr/share/dict/words")
MIN_LEN = 6              # shorter dictionary words turn up by chance inside junk ("oust", "cane", "goof")


def load_vocab(extra=None, system=SYSTEM_WORDS):
    """Lower-case a-z words from the system list and/or a file (one word per line, # comments)."""
    words = set()
    for path in [system, extra]:
        if path and Path(path).exists():
            for line in Path(path).read_text(errors="ignore").splitlines():
                w = line.split("#")[0].strip().lower()
                if re.fullmatch(r"[a-z]+", w):
                    words.add(w)
    return words


def spot_words(text, vocab, min_len=MIN_LEN):
    """-> [(word, letter_index)]. Text that is exactly one known word is kept whatever its length ("hello");
    otherwise greedy longest match, left to right, ignoring spaces the decoder put in, and letters that
    start no word of >= min_len are skipped."""
    letters = text.replace(" ", "").lower()
    if letters in vocab:
        return [(letters, 0)]
    found, i = [], 0
    while i < len(letters):
        word = next((letters[i:j] for j in range(len(letters), i + min_len - 1, -1) if letters[i:j] in vocab), None)
        if word:
            found.append((word, i))
            i += len(word)
        else:
            i += 1
    return found


def words_in_segment(seg, vocab, min_len=MIN_LEN):
    """Spotted words with a time each: letters are spread evenly over the segment, as in the captions."""
    n = max(1, len(seg["text"].replace(" ", "")))
    span = seg["end_s"] - seg["start_s"]
    return [{"start_s": round(seg["start_s"] + i / n * span, 2), "text": w}
            for w, i in spot_words(seg["text"], vocab, min_len)]
