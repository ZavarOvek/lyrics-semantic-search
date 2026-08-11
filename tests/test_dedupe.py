"""SPEC-04 §1: direct unit coverage for core/dedupe.py.

Previously only exercised incidentally (via real corpora that happened to
contain duplicate blocks) -- no test directly asserted the duplicate-branch
behavior itself.
"""

from __future__ import annotations

from lyrics_search.core.dedupe import dedupe_blocks


def test_no_duplicates_keeps_everything():
    keep, dup = dedupe_blocks(["one", "two", "three"])
    assert keep == [0, 1, 2]
    assert dup == []


def test_exact_repeat_is_flagged_as_duplicate():
    keep, dup = dedupe_blocks(["chorus line", "verse line", "chorus line"])
    assert keep == [0, 1]
    assert dup == [2]


def test_normalization_insensitive_duplicate():
    # dedupe_blocks hashes via normalize_for_dedupe (case/punct/whitespace
    # insensitive) -- these two should collide despite differing surface form.
    keep, dup = dedupe_blocks(["Hello, World!", "hello   world"])
    assert keep == [0]
    assert dup == [1]


def test_empty_input_returns_empty():
    assert dedupe_blocks([]) == ([], [])


def test_first_occurrence_wins_order_preserved():
    keep, dup = dedupe_blocks(["a", "b", "a", "a", "c"])
    assert keep == [0, 1, 4]
    assert dup == [2, 3]
