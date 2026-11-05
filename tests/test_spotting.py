from aslcv.spotting import Vocab, load_vocab, spot_words, words_in_segment


def test_greedy_longest_match_ignores_decoder_spaces():
    vocab = {"campylobacter", "bacter", "bacteria", "website"}
    assert spot_words("campyl obacterbacteriatco", vocab) == [("campylobacter", 0), ("bacteria", 13)]


def test_short_words_need_min_len():
    assert spot_words("goofeces", {"goof", "feces"}) == []
    assert spot_words("goofeces", Vocab({"goof", "feces"}, min_len=4)) == [("goof", 0)]   # greedy: the f is used up


def test_topic_words_count_at_any_length():
    v = Vocab({"sisters"}, topic={"fsis"})
    assert spot_words("fsiswebsite", v) == [("fsis", 0)]
    assert spot_words("xsistersx", v) == [("sisters", 1)]


def test_word_time_follows_its_letters():
    seg = {"start_s": 0.0, "end_s": 10.0, "text": "xxxxxbacteria"}
    assert words_in_segment(seg, {"bacteria"}) == [{"start_s": round(5 / 13 * 10, 2), "text": "bacteria"}]


def test_load_vocab_merges_files_and_cleans(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_text("Apple\nit's\n")
    b.write_text("# topic words\nJejuni  # a bacterium\n\n")
    v = load_vocab(b, system=a)
    assert v.words == {"apple", "jejuni"} and v.topic == {"jejuni"}
    assert len(load_vocab(None, system=tmp_path / "missing")) == 0


def test_a_whole_known_word_is_kept_even_if_short():
    assert spot_words("HELLO", {"hello"}) == [("hello", 0)]
    assert spot_words("xhello", {"hello"}) == []            # inside a longer string it would need 6 letters
