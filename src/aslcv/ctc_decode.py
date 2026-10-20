"""Spelling-aware decoding for the CTC sequence model (journal 14).

Greedy decoding takes the single most likely symbol per frame and merges repeats, so "mall" needs a confident
blank between the two l's, which fast signers rarely give. Prefix beam search keeps the `beam` most likely
spellings at every frame (summing over all the frame paths that produce each spelling), and adds a character
language model's opinion:

    score(spelling) = log P_ctc(spelling | video) + alpha * log P_lm(spelling) + beta * letters + beta_space * spaces

beta rewards letters (it counters the language model's preference for short outputs). Spaces get their own
beta_space (default 0): sharing beta made inserting a space almost free ("campyl obacter", journal 15).

The character model is an n-gram with Witten-Bell smoothing, trained on training-set labels plus an English
word list (never on dev/test labels). alpha and beta are tuned on the dev signers.
"""
from collections import defaultdict
import numpy as np

from .seqmodel import REPEAT, expand_repeats

NEG = -1e30
END = "$"


class CharLM:
    def __init__(self, texts, order=4):
        self.order = order
        self.alphabet = sorted({c for t in texts for c in t} | {END})
        self.counts = [defaultdict(lambda: defaultdict(int)) for _ in range(order)]   # [k][context of len k][char]
        for t in texts:
            s = "^" * (order - 1) + t + END
            for i in range(order - 1, len(s)):
                for k in range(order):
                    self.counts[k][s[i - k:i]][s[i]] += 1
        self._cache = {}

    def _p(self, context, c):
        """Witten-Bell interpolation, from the full context down to a uniform distribution."""
        p = 1.0 / len(self.alphabet)
        for k in range(0, len(context) + 1):
            h = context[len(context) - k:] if k else ""
            nxt = self.counts[k].get(h)
            if not nxt:
                continue
            total, types = sum(nxt.values()), len(nxt)
            p = (nxt.get(c, 0) + types * p) / (total + types)
        return p

    def logp(self, prefix, c):
        context = ("^" * (self.order - 1) + prefix)[-(self.order - 1):]
        key = (context, c)
        if key not in self._cache:
            self._cache[key] = float(np.log(self._p(context, c))) if c in self.alphabet else -20.0
        return self._cache[key]


def _lae(a, b):
    if a == NEG:
        return b
    if b == NEG:
        return a
    return max(a, b) + np.log1p(np.exp(-abs(a - b)))


def beam_decode(logits, vocab, lm=None, alpha=0.5, beta=0.0, beam=16, prune=1e-3, beta_space=0.0):
    """CTC prefix beam search with optional character-LM fusion. Returns the decoded text
    (with repeat symbols expanded)."""
    lp_all = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)        # log-softmax
    blank = vocab.blank
    syms = vocab.symbols

    def text_of(prefix):
        return expand_repeats("".join(syms[i - 1] for i in prefix))

    beams = {(): (0.0, NEG)}                     # prefix -> (log p ending in blank, log p ending in symbol)
    for lp in lp_all:
        cands = [s for s in range(vocab.size) if s != blank and lp[s] > np.log(prune)]
        nxt = defaultdict(lambda: [NEG, NEG])
        for prefix, (pb, pnb) in beams.items():
            total = _lae(pb, pnb)
            nb = nxt[prefix]
            nb[0] = _lae(nb[0], total + lp[blank])
            text = text_of(prefix) if lm else ""
            for s in cands:
                ch = syms[s - 1]
                step_bonus = beta_space if ch == " " else beta
                if lm:
                    lm_char = (text[-1] if text and text[-1] != " " else ch) if ch == REPEAT else ch
                    bonus = alpha * lm.logp(text, lm_char) + step_bonus
                else:
                    bonus = step_bonus
                new = prefix + (s,)
                if prefix and s == prefix[-1]:
                    nxt[new][1] = _lae(nxt[new][1], pb + lp[s] + bonus)     # a real repeat needs a blank first
                    nb[1] = _lae(nb[1], pnb + lp[s])                         # same symbol continuing
                else:
                    nxt[new][1] = _lae(nxt[new][1], total + lp[s] + bonus)
        beams = dict(sorted(nxt.items(), key=lambda kv: -_lae(*kv[1]))[:beam])

    def final(kv):
        prefix, (pb, pnb) = kv
        score = _lae(pb, pnb)
        if lm:
            score += alpha * lm.logp(text_of(prefix), END)
        return score
    best = max(beams.items(), key=final)[0]
    return text_of(best)
