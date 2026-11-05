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


class Vocab:
    """Known words. General words (system list) must have >= min_len letters to be spotted inside a longer
    string; topic words (--vocab) are deliberate, so any of 3+ letters count ("fsis")."""

    def __init__(self, words=(), topic=(), min_len=MIN_LEN):
        self.topic = set(topic)
        self.words = set(words) | self.topic
        self.min_len = min_len

    def __contains__(self, word):
        return word in self.words

    def __len__(self):
        return len(self.words)

    def spottable(self, word):
        return word in self.topic or (word in self.words and len(word) >= self.min_len)


def _read_words(path):
    words = set()
    if path and Path(path).exists():
        for line in Path(path).read_text(errors="ignore").splitlines():
            w = line.split("#")[0].strip().lower()
            if re.fullmatch(r"[a-z]+", w):
                words.add(w)
    return words


def load_vocab(extra=None, system=SYSTEM_WORDS):
    """System word list + topic words from `extra` (one per line, # comments); lower-case a-z only."""
    return Vocab(_read_words(system), {w for w in _read_words(extra) if len(w) >= 3})


def spot_words(text, vocab):
    """-> [(word, letter_index)]. Text that is exactly one known word is kept whatever its length ("hello");
    otherwise greedy longest match, left to right, ignoring spaces the decoder put in; letters that start
    no spottable word are skipped."""
    if not isinstance(vocab, Vocab):
        vocab = Vocab(vocab)
    letters = text.replace(" ", "").lower()
    if letters in vocab:
        return [(letters, 0)]
    found, i = [], 0
    while i < len(letters):
        word = next((letters[i:j] for j in range(len(letters), i, -1) if vocab.spottable(letters[i:j])), None)
        if word:
            found.append((word, i))
            i += len(word)
        else:
            i += 1
    return found


def words_in_segment(seg, vocab):
    """Spotted words with a time each: letters are spread evenly over the segment, as in the captions."""
    n = max(1, len(seg["text"].replace(" ", "")))
    span = seg["end_s"] - seg["start_s"]
    return [{"start_s": round(seg["start_s"] + i / n * span, 2), "text": w}
            for w, i in spot_words(seg["text"], vocab)]
