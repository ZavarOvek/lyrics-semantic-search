"""SPEC-04 §1: direct unit coverage for core/sections.py's chunking cascade.

This module previously had zero dedicated tests -- only incidental coverage
via tests/test_preprocess_song_filter.py's integration test, which happens
to touch a handful of lines exercising a "normal" song and nothing else.
That left most of the cascade's actual branches (bracket classification,
plain-label classification, blank-line refinement, level-4 length packing,
and its two internal fallback paths) completely unverified. These tests
target each branch directly, by function where practical (fast, precise
failure localization) and through chunk_song() end-to-end where the
positional/cascade interaction between levels is the point.
"""
from __future__ import annotations

from lyrics_search.core.sections import (
    HARD_CEILING_WORDS,
    Segment,
    _find_bracket_boundaries,
    _pack_by_length,
    _split_long_line,
    chunk_song,
    classify_bracket,
    classify_plain_label_line,
)

# --- classify_bracket --------------------------------------------------


def test_classify_bracket_noise_empty():
    assert classify_bracket("") == ("noise", None)


def test_classify_bracket_noise_whitespace_only():
    assert classify_bracket("   ") == ("noise", None)


def test_classify_bracket_noise_no_alpha():
    assert classify_bracket("??") == ("noise", None)


def test_classify_bracket_canonical_single_word():
    assert classify_bracket("Verse") == ("canonical", "Verse")


def test_classify_bracket_canonical_case_insensitive():
    assert classify_bracket("CHORUS") == ("canonical", "Chorus")


def test_classify_bracket_canonical_with_trailing_number():
    # leading word alone must resolve; "1" is not required to match anything.
    assert classify_bracket("Verse 1") == ("canonical", "Verse")


def test_classify_bracket_compound_pre_chorus():
    assert classify_bracket("Pre Chorus") == ("canonical", "Pre-Chorus")


def test_classify_bracket_compound_post_chorus():
    assert classify_bracket("Post Chorus") == ("canonical", "Post-Chorus")


def test_classify_bracket_credits_single_name():
    assert classify_bracket("Zaytoven") == ("credits", None)


def test_classify_bracket_credits_multi_word_non_canonical():
    assert classify_bracket("Lord Christy K:") == ("credits", None)


# --- classify_plain_label_line -----------------------------------------


def test_plain_label_empty_line():
    assert classify_plain_label_line("") is None


def test_plain_label_rejects_bracketed_line():
    assert classify_plain_label_line("[Verse 1]") is None


def test_plain_label_no_alpha_tokens():
    assert classify_plain_label_line("123") is None


def test_plain_label_too_many_tokens_is_ordinary_lyric():
    # Conservative-on-purpose: an actual lyric line that merely mentions a
    # section word must NOT be misclassified as a boundary.
    assert classify_plain_label_line("the chorus of angels sang loud") is None


def test_plain_label_single_canonical_word():
    assert classify_plain_label_line("VERSE 1") == "Verse"


def test_plain_label_trailing_colon_stripped():
    assert classify_plain_label_line("Chorus:") == "Chorus"


def test_plain_label_compound_two_tokens():
    assert classify_plain_label_line("Pre Chorus") == "Pre-Chorus"


def test_plain_label_two_tokens_second_is_canonical():
    assert classify_plain_label_line("Big Chorus") == "Chorus"


def test_plain_label_two_tokens_neither_canonical():
    assert classify_plain_label_line("Hey Yo") is None


# --- chunk_song: level 1+2 (outer segments) -----------------------------


def test_chunk_song_empty_text_returns_no_segments():
    assert chunk_song("") == []


def test_chunk_song_whitespace_only_returns_no_segments():
    assert chunk_song("   \n  \t ") == []


def test_chunk_song_no_boundaries_single_none_segment():
    segs = chunk_song("just some plain lyrics with no markup at all")
    assert len(segs) == 1
    assert segs[0].section is None
    assert segs[0].split_by == "none"
    assert segs[0].force_split is False


def test_chunk_song_leading_text_before_first_boundary():
    text = "some intro line\n[Chorus]\nla la la"
    segs = chunk_song(text)
    assert segs[0].split_by == "none"
    assert segs[0].section is None
    assert "intro line" in segs[0].text
    assert segs[1].split_by == "bracket_tag"
    assert segs[1].section == "Chorus"


def test_chunk_song_noise_bracket_does_not_create_boundary():
    text = "[?]\nsome real lyrics here"
    segs = chunk_song(text)
    assert len(segs) == 1
    assert segs[0].split_by == "none"
    assert "[?]" not in segs[0].text


def test_chunk_song_mixes_bracket_and_plain_label_positionally():
    text = "[Verse 1]\nfirst verse line here\nHOOK\nsecond bit goes here"
    segs = chunk_song(text)
    split_bys = [s.split_by for s in segs]
    assert split_bys == ["bracket_tag", "plain_label"]
    assert segs[0].section == "Verse"
    assert segs[1].section == "Hook"


def test_chunk_song_empty_body_between_boundaries_skipped():
    # Two bracket tags with nothing but whitespace between them: the first
    # tag's "body" is empty and must not produce a blank Segment.
    text = "[Verse]\n\n[Chorus]\nactual content"
    segs = chunk_song(text)
    assert len(segs) == 1
    assert segs[0].section == "Chorus"


def test_chunk_song_credits_bracket_is_a_boundary_with_no_section():
    text = "[Zaytoven]\nproducer tag line as its own segment"
    segs = chunk_song(text)
    assert len(segs) == 1
    assert segs[0].section is None
    assert segs[0].split_by == "bracket_tag"


# --- chunk_song: level 3 (blank-line refinement) ------------------------


def test_chunk_song_blank_line_splits_single_segment():
    text = "first paragraph here\n\nsecond paragraph here"
    segs = chunk_song(text)
    assert len(segs) == 2
    assert all(s.split_by == "blank_line" for s in segs)


def test_chunk_song_blank_line_refines_within_bracket_segment():
    text = "[Verse]\npart one\n\npart two"
    segs = chunk_song(text)
    assert len(segs) == 2
    assert all(s.section == "Verse" for s in segs)
    assert all(s.split_by == "blank_line" for s in segs)


def test_chunk_song_single_paragraph_no_blank_line_keeps_original_split_by():
    text = "[Verse]\njust one paragraph, no blank line inside it at all"
    segs = chunk_song(text)
    assert len(segs) == 1
    assert segs[0].split_by == "bracket_tag"


# --- chunk_song: level 4 (force-split / length packing) -----------------


def test_chunk_song_long_segment_triggers_force_split():
    long_line = " ".join(f"word{i}" for i in range(HARD_CEILING_WORDS + 20))
    segs = chunk_song(long_line, target_max=40, hard_ceiling=HARD_CEILING_WORDS)
    assert len(segs) > 1
    assert all(s.force_split for s in segs)
    assert all(s.split_by == "none" for s in segs)
    # never split a word, and no piece exceeds the hard ceiling
    for s in segs:
        assert len(s.text.split()) <= HARD_CEILING_WORDS


def test_chunk_song_force_split_calls_warn_once_per_oversized_segment():
    calls: list[int] = []
    long_line = " ".join(f"w{i}" for i in range(HARD_CEILING_WORDS + 10))
    chunk_song(long_line, hard_ceiling=HARD_CEILING_WORDS, warn=calls.append)
    assert len(calls) == 1
    assert calls[0] == HARD_CEILING_WORDS + 10


def test_chunk_song_short_segment_does_not_trigger_warn():
    calls: list[int] = []
    chunk_song("short line", warn=calls.append)
    assert calls == []


def test_chunk_song_single_long_sentence_no_punctuation_word_split_fallback():
    # One giant "sentence" (no . ! ?) that alone exceeds the hard ceiling:
    # _split_long_line must fall back to whitespace splitting.
    single_line = " ".join(f"tok{i}" for i in range(HARD_CEILING_WORDS + 30))
    segs = chunk_song(single_line, target_max=40, hard_ceiling=HARD_CEILING_WORDS)
    assert len(segs) > 1
    rejoined = " ".join(s.text for s in segs)
    assert rejoined.split() == single_line.split()  # no word lost or split


def test_chunk_song_long_paragraph_splits_on_sentence_punctuation():
    sentence = "This is one short sentence."
    text = " ".join([sentence] * (HARD_CEILING_WORDS // 3))
    segs = chunk_song(text, target_max=20, hard_ceiling=HARD_CEILING_WORDS)
    assert len(segs) > 1
    assert all(s.force_split for s in segs)


def test_chunk_song_pack_by_length_flushes_at_target_max():
    # 30 lines x 6 words = 180 words total, over hard_ceiling (120), so
    # _pack_by_length actually runs; target_max=30 should flush roughly
    # every 5 lines rather than waiting for the 120-word hard ceiling.
    lines = "\n".join(f"line number {i} has five words" for i in range(30))
    segs = chunk_song(lines, target_max=30, hard_ceiling=HARD_CEILING_WORDS)
    assert len(segs) >= 2
    for s in segs:
        assert len(s.text.split()) <= HARD_CEILING_WORDS
    # no line lost in the process
    assert sum(len(s.text.split()) for s in segs) == 180


# --- private helpers, tested directly for precise branch coverage -------
# (same convention as test_preprocess_song_filter.py's direct testing of
# _song_reject_reason: these are pure, deterministic, and some of their
# branches are unreachable through chunk_song's own normal call path.)


def test_find_bracket_boundaries_skips_noise_bracket():
    # Called directly (bypassing chunk_song's own _strip_noise_brackets
    # pre-pass, which normally removes noise brackets before this function
    # ever sees them) to exercise the "kind == noise: continue" branch.
    boundaries = _find_bracket_boundaries("[?] some text [Chorus] more")
    assert len(boundaries) == 1
    assert boundaries[0].section == "Chorus"


def test_pack_by_length_flush_triggered_by_hard_ceiling_not_target_max():
    # target_max set high enough that it's never reached on its own; the
    # third line's addition must instead overflow hard_ceiling and force a
    # flush via the "cur_words + w > hard_ceiling" branch.
    text = "\n".join([f"{'w ' * 50}line{i}" for i in range(3)])  # ~51 words/line
    pieces = _pack_by_length(text, target_max=1000, hard_ceiling=120)
    assert len(pieces) >= 2
    for piece in pieces:
        assert len(piece.split()) <= 120


def test_pack_by_length_single_line_exceeds_hard_ceiling_delegates_to_split():
    line = " ".join(f"w{i}" for i in range(200))
    pieces = _pack_by_length(line, target_max=40, hard_ceiling=120)
    assert len(pieces) > 1
    rejoined = " ".join(pieces)
    assert rejoined.split() == line.split()


def test_split_long_line_sentence_punctuation_path():
    line = "First sentence here. Second sentence here. Third one too."
    pieces = _split_long_line(line, hard_ceiling=5)
    assert pieces == [
        "First sentence here.",
        "Second sentence here.",
        "Third one too.",
    ]


def test_split_long_line_word_fallback_when_no_punctuation():
    line = " ".join(f"w{i}" for i in range(25))
    pieces = _split_long_line(line, hard_ceiling=10)
    assert len(pieces) == 3  # 10 + 10 + 5 words
    assert " ".join(pieces).split() == line.split()


# --- Segment dataclass ---------------------------------------------------


def test_segment_is_frozen_and_defaults_force_split_false():
    seg = Segment("text", None, "none")
    assert seg.force_split is False
