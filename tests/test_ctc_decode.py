import numpy as np
from aslcv.ctc_decode import CharLM, beam_decode
from aslcv.seqmodel import Vocab, ctc_greedy_decode


def logits_from_probs(rows, vocab):
    """rows: list of {symbol: prob}; '-' is the blank."""
    P = np.full((len(rows), vocab.size), 1e-4)
    for t, r in enumerate(rows):
        for sym, p in r.items():
            P[t, 0 if sym == "-" else vocab.index[sym]] = p
    P /= P.sum(1, keepdims=True)
    return np.log(P)


def test_lm_probabilities_are_normalised_and_learn_spelling():
    lm = CharLM(["mall", "ball", "call", "hall"] * 5, order=4)
    total = sum(np.exp(lm.logp("ma", c)) for c in lm.alphabet)
    assert abs(total - 1.0) < 1e-6
    assert lm.logp("mal", "l") > lm.logp("mal", "x")


def test_without_lm_beam_matches_greedy_on_clear_input():
    v = Vocab("abc")
    L = logits_from_probs([{"a": .9}, {"-": .9}, {"b": .9}, {"c": .9}], v)
    assert beam_decode(L, v, lm=None) == ctc_greedy_decode(L, v) == "abc"


def test_lm_recovers_a_double_letter_greedy_loses():
    v = Vocab("abhlm")
    # m a l [l-or-blank, l slightly ahead] l : greedy merges to "mal"
    L = logits_from_probs([{"m": .9}, {"a": .9}, {"l": .9}, {"l": .55, "-": .45}, {"l": .9}], v)
    assert ctc_greedy_decode(L, v) == "mal"
    lm = CharLM(["mall", "hall", "ball", "all"] * 20, order=4)
    assert beam_decode(L, v, lm=lm, alpha=1.0, beta=0.5) == "mall"


def test_repeat_symbol_is_expanded_for_the_lm_and_the_output():
    v = Vocab("al+")
    L = logits_from_probs([{"a": .9}, {"l": .9}, {"+": .9}], v)
    assert beam_decode(L, v, lm=None) == "all"
