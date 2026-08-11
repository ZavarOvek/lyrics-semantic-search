"""Tests for `core/lexical_leak.py` -- the IDF-weighted leak gate.

The cases that matter are the two that motivate the module: a paraphrase
full of shared *common* words must score low, and a paraphrase sharing
one rare word must be caught by `max_shared_idf` even when `ratio` looks
healthy. Both are asserted explicitly below rather than left implied by
arithmetic coverage.
"""

from __future__ import annotations

from lyrics_search.core.lexical_leak import LeakScore, idf_weighted_overlap

# A miniature corpus IDF table: ordinary words cheap, proper nouns dear.
IDF = {
    "the": 1.5,
    "night": 3.0,
    "love": 3.5,
    "leaving": 4.0,
    "station": 5.0,
    "cordoba": 10.0,
    "marisol": 11.0,
}


def test_no_shared_terms_scores_zero() -> None:
    score = idf_weighted_overlap(["love", "night"], ["station", "cordoba"], IDF)
    assert score.ratio == 0.0
    assert score.weight_shared == 0.0
    assert score.max_shared_idf == 0.0
    assert score.shared == ()


def test_ratio_is_shared_weight_over_total_weight() -> None:
    score = idf_weighted_overlap(["the", "night"], ["night"], IDF)
    assert score.weight_total == 4.5
    assert score.weight_shared == 3.0
    assert score.ratio == 3.0 / 4.5
    assert score.shared == ("night",)


def test_common_words_shared_stay_low_but_rare_word_is_caught() -> None:
    """The whole point of the module, as two contrasting queries."""
    source = ["the", "night", "station", "cordoba"]

    honest = idf_weighted_overlap(["the", "night", "love"], source, IDF)
    leaky = idf_weighted_overlap(["the", "love", "leaving", "cordoba"], source, IDF)

    # The honest paraphrase shares more *words* with the source than the
    # leaky one does, yet carries no anchor.
    assert len(honest.shared) == 2
    assert len(leaky.shared) == 2
    assert honest.max_shared_idf == 3.0
    assert leaky.max_shared_idf == 10.0


def test_unknown_terms_are_excluded_from_both_weights() -> None:
    score = idf_weighted_overlap(["night", "zzzz"], ["night", "zzzz"], IDF)
    assert score.unknown == ("zzzz",)
    # "zzzz" occurs in the source too, but weighs nothing either way.
    assert score.weight_total == 3.0
    assert score.weight_shared == 3.0
    assert score.ratio == 1.0
    assert score.shared == ("night",)


def test_empty_query_scores_zero_rather_than_raising() -> None:
    score = idf_weighted_overlap([], ["night"], IDF)
    assert score == LeakScore(0.0, 0.0, 0.0, 0.0, (), ())


def test_query_of_only_unknown_terms_scores_zero() -> None:
    score = idf_weighted_overlap(["zzzz", "qqqq"], ["zzzz"], IDF)
    assert score.ratio == 0.0
    assert score.weight_total == 0.0
    assert score.unknown == ("qqqq", "zzzz")


def test_repeated_terms_count_once() -> None:
    once = idf_weighted_overlap(["night"], ["night"], IDF)
    twice = idf_weighted_overlap(["night", "night"], ["night", "night"], IDF)
    assert once == twice


def test_shared_and_unknown_are_sorted_tuples() -> None:
    score = idf_weighted_overlap(["station", "night", "qqqq", "zzzz"], ["station", "night"], IDF)
    assert score.shared == ("night", "station")
    assert score.unknown == ("qqqq", "zzzz")
