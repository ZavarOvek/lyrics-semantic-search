"""SPEC-04 §1: hypothesis-based property tests.

Complements the example-based unit tests elsewhere (test_fusion.py,
test_sections.py, test_dedupe.py) with randomized property checks for two
kinds of invariant the example tests can't fully demonstrate on their own:

- RRF fusion's algebraic properties (rank-only dependence, order/tie
  behavior, deterministic well-formed output) across many random ranked-list
  shapes, not just the handful of hand-picked examples in test_fusion.py.
- Idempotency: applying a pure transformation to its own already-stable
  output must be a no-op. Checked here for chunk_song (core/sections.py),
  dedupe_blocks (core/dedupe.py), and normalize_for_dedupe (core/text.py).
"""

from __future__ import annotations

import string

from hypothesis import given
from hypothesis import strategies as st

from lyrics_search.core.dedupe import dedupe_blocks
from lyrics_search.core.fusion import reciprocal_rank_fusion
from lyrics_search.core.sections import (
    HARD_CEILING_WORDS,
    chunk_song,
    classify_plain_label_line,
)
from lyrics_search.core.text import normalize_for_dedupe

song_id_strategy = st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=5)
ranked_list_strategy = st.lists(song_id_strategy, min_size=0, max_size=8, unique=True)

# ---------------------------------------------------------------------
# RRF (reciprocal_rank_fusion) properties
# ---------------------------------------------------------------------


@given(st.lists(ranked_list_strategy, min_size=1, max_size=4))
def test_rrf_output_ids_equal_union_of_input_ids(ranked_lists):
    fused = reciprocal_rank_fusion(ranked_lists)
    fused_ids = {song_id for song_id, _ in fused}
    expected_ids = {song_id for lst in ranked_lists for song_id in lst}
    assert fused_ids == expected_ids


@given(st.lists(ranked_list_strategy, min_size=1, max_size=4))
def test_rrf_output_has_no_duplicate_ids(ranked_lists):
    fused = reciprocal_rank_fusion(ranked_lists)
    ids = [song_id for song_id, _ in fused]
    assert len(ids) == len(set(ids))


@given(ranked_list_strategy)
def test_rrf_single_list_preserves_order_exactly(ranked_list):
    fused = reciprocal_rank_fusion([ranked_list])
    assert [song_id for song_id, _ in fused] == ranked_list


@given(ranked_list_strategy)
def test_rrf_all_scores_strictly_positive(ranked_list):
    fused = reciprocal_rank_fusion([ranked_list])
    assert all(score > 0 for _, score in fused)


@given(ranked_list_strategy)
def test_rrf_idempotent_order_under_self_duplication(ranked_list):
    """Fusing a list with an exact copy of itself must not change the
    relative order: every score simply doubles uniformly, so the fused
    ranking is identical to fusing the list once alone -- a form of
    idempotency (repeating identical evidence doesn't reshuffle anything)."""
    once = reciprocal_rank_fusion([ranked_list])
    twice = reciprocal_rank_fusion([ranked_list, ranked_list])
    assert [song_id for song_id, _ in once] == [song_id for song_id, _ in twice]


# ---------------------------------------------------------------------
# chunk_song idempotency
# ---------------------------------------------------------------------

# Words built only from letters (no whitespace/brackets/blank lines inside
# a "word"), so joining them with single spaces produces text whose word
# count is exactly len(words) and which contains no bracket_tag or
# blank_line boundary. A word list of length <=2 can still be classified
# as a plain_label boundary on its own (e.g. ["BRIDGE"] or ["Pre", "Chorus"]
# -- classify_plain_label_line's own definition of markup), which would
# make chunk_song legitimately collapse it to zero segments (an empty body
# after the label) rather than reproduce it as one segment -- filtered out
# below since that's genuine markup, not the "no markup" case this test
# is about.
plain_word = st.text(alphabet=string.ascii_letters, min_size=1, max_size=10)
word_list_no_label = st.lists(plain_word, min_size=1, max_size=HARD_CEILING_WORDS).filter(
    lambda words: classify_plain_label_line(" ".join(words)) is None
)


@given(word_list_no_label)
def test_chunk_song_idempotent_on_already_minimal_segment(words):
    """A segment already short enough to survive level 4 untouched
    (word count <= hard_ceiling, no markup) must chunk to exactly itself
    again when fed back in -- chunk_song should be a no-op on its own
    stable output, not just on the original raw input."""
    text = " ".join(words)
    first = chunk_song(text)
    assert len(first) == 1

    second = chunk_song(first[0].text)
    assert len(second) == 1
    assert second[0].text == first[0].text
    assert second[0].section == first[0].section
    assert second[0].split_by == first[0].split_by
    assert second[0].force_split is False


# ---------------------------------------------------------------------
# dedupe_blocks idempotency
# ---------------------------------------------------------------------


@given(st.lists(st.text(min_size=0, max_size=20), min_size=0, max_size=15))
def test_dedupe_blocks_idempotent_on_kept_subset(texts):
    """Running dedupe_blocks again on exactly the texts it already kept
    must keep all of them and flag none as duplicates -- a second pass over
    already-deduplicated input is a no-op."""
    keep, _dup = dedupe_blocks(texts)
    kept_texts = [texts[i] for i in keep]

    keep2, dup2 = dedupe_blocks(kept_texts)
    assert keep2 == list(range(len(kept_texts)))
    assert dup2 == []


# ---------------------------------------------------------------------
# normalize_for_dedupe idempotency
# ---------------------------------------------------------------------


@given(st.text(max_size=100))
def test_normalize_for_dedupe_idempotent(text):
    once = normalize_for_dedupe(text)
    twice = normalize_for_dedupe(once)
    assert once == twice
