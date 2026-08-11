"""EVAL-PREP §4: core/metrics.py.

The four properties EVAL-PREP names as required are each pinned by a
hypothesis test below, marked `# REQUIRED PROPERTY n`. The example-based
tests around them exist for a different reason: a property test says a
metric behaves consistently, it does not say the metric is the one that
was asked for. Hand-computed values are what catch an off-by-one in the
nDCG discount or a denominator that quietly became min(|relevant|, k).
"""

from __future__ import annotations

import math
import string

import pytest
from hypothesis import given
from hypothesis import strategies as st

from lyrics_search.core.metrics import (
    mean_ci,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)

song_id_strategy = st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=4)
ranked_strategy = st.lists(song_id_strategy, min_size=0, max_size=10, unique=True)


@st.composite
def ranked_and_relevant(draw):
    """A ranked list plus a relevant set that overlaps it some of the
    time -- relevant songs that were never retrieved are the normal case,
    not an edge case, so the set is not drawn as a subset."""
    ranked = draw(ranked_strategy)
    pool = ranked + draw(st.lists(song_id_strategy, min_size=0, max_size=4))
    relevant = draw(
        st.sets(st.sampled_from(pool) if pool else song_id_strategy, min_size=0, max_size=5)
    )
    return ranked, relevant


# --- REQUIRED PROPERTY 1: nDCG stays within [0, 1] ---------------------


@given(ranked_and_relevant(), st.integers(min_value=0, max_value=15))
def test_ndcg_is_always_within_the_unit_interval(pair, k):
    ranked, relevant = pair
    value = ndcg_at_k(ranked, relevant, k)
    assert 0.0 <= value <= 1.0
    assert not math.isnan(value)


# --- REQUIRED PROPERTY 2: promoting a relevant song never hurts --------


@st.composite
def promotable_case(draw):
    """A case where a relevant song sits at position i and can be swapped
    with whatever is at some better position j < i."""
    ranked = draw(st.lists(song_id_strategy, min_size=2, max_size=10, unique=True))
    relevant = draw(st.sets(st.sampled_from(ranked), min_size=1, max_size=len(ranked)))
    relevant_positions = [i for i, s in enumerate(ranked) if s in relevant and i > 0]
    if not relevant_positions:
        return ranked, relevant, None, None
    i = draw(st.sampled_from(relevant_positions))
    j = draw(st.integers(min_value=0, max_value=i - 1))
    return ranked, relevant, i, j


@given(promotable_case(), st.integers(min_value=1, max_value=12))
def test_promoting_a_relevant_song_never_decreases_any_metric(case, k):
    ranked, relevant, i, j = case
    if i is None:
        return
    promoted = list(ranked)
    promoted[i], promoted[j] = promoted[j], promoted[i]

    assert recall_at_k(promoted, relevant, k) >= recall_at_k(ranked, relevant, k)
    assert reciprocal_rank(promoted, relevant) >= reciprocal_rank(ranked, relevant)
    assert ndcg_at_k(promoted, relevant, k) >= ndcg_at_k(ranked, relevant, k)


# --- REQUIRED PROPERTY 3: empty result lists score 0 -------------------


@given(st.sets(song_id_strategy, min_size=0, max_size=5), st.integers(min_value=0, max_value=10))
def test_an_empty_ranked_list_scores_zero_everywhere(relevant, k):
    assert recall_at_k([], relevant, k) == 0.0
    assert reciprocal_rank([], relevant) == 0.0
    assert ndcg_at_k([], relevant, k) == 0.0
    assert mean_reciprocal_rank([([], relevant)]) == 0.0


def test_an_empty_relevant_set_scores_zero_rather_than_dividing_by_zero():
    ranked = ["a", "b", "c"]
    assert recall_at_k(ranked, [], 3) == 0.0
    assert reciprocal_rank(ranked, []) == 0.0
    assert ndcg_at_k(ranked, [], 3) == 0.0


def test_mean_reciprocal_rank_over_nothing_is_zero():
    assert mean_reciprocal_rank([]) == 0.0


# --- REQUIRED PROPERTY 4: Recall@k is monotonic in k -------------------


@given(ranked_and_relevant(), st.integers(min_value=0, max_value=12))
def test_recall_is_monotonic_in_k(pair, k):
    ranked, relevant = pair
    assert recall_at_k(ranked, relevant, k + 1) >= recall_at_k(ranked, relevant, k)


@given(ranked_and_relevant(), st.integers(min_value=0, max_value=12))
def test_recall_is_within_the_unit_interval(pair, k):
    ranked, relevant = pair
    assert 0.0 <= recall_at_k(ranked, relevant, k) <= 1.0


# --- hand-computed values ----------------------------------------------


def test_recall_denominator_is_the_full_relevant_set_not_the_window():
    # 3 relevant songs, only 1 reachable inside k=1 -- recall must be
    # 1/3, not 1/1.
    assert recall_at_k(["a", "b", "c"], {"a", "b", "c"}, 1) == pytest.approx(1 / 3)
    assert recall_at_k(["a", "b", "c"], {"a", "b", "c"}, 3) == 1.0


def test_recall_counts_only_the_top_k_prefix():
    assert recall_at_k(["x", "y", "a"], {"a"}, 2) == 0.0
    assert recall_at_k(["x", "y", "a"], {"a"}, 3) == 1.0


def test_reciprocal_rank_uses_the_first_relevant_hit_one_based():
    assert reciprocal_rank(["a"], {"a"}) == 1.0
    assert reciprocal_rank(["x", "a"], {"a"}) == 0.5
    assert reciprocal_rank(["x", "y", "a", "b"], {"a", "b"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


def test_mean_reciprocal_rank_averages_per_query():
    per_query = [(["a"], {"a"}), (["x", "a"], {"a"}), (["x"], {"a"})]
    assert mean_reciprocal_rank(per_query) == pytest.approx((1.0 + 0.5 + 0.0) / 3)


def test_ndcg_is_one_when_every_relevant_song_is_on_top():
    assert ndcg_at_k(["a", "b", "x", "y"], {"a", "b"}, 10) == 1.0


def test_ndcg_hand_computed_discount():
    # one relevant song at rank 2: DCG = 1/log2(3), IDCG = 1/log2(2) = 1.
    assert ndcg_at_k(["x", "a"], {"a"}, 10) == pytest.approx(1 / math.log2(3))


def test_ndcg_ideal_is_capped_at_k_unlike_recall():
    # 3 relevant, window of 1: the best achievable inside k=1 is one hit,
    # so a single hit at rank 1 is nDCG 1.0 while recall is only 1/3.
    ranked, relevant = ["a", "b", "c"], {"a", "b", "c"}
    assert ndcg_at_k(ranked, relevant, 1) == 1.0
    assert recall_at_k(ranked, relevant, 1) == pytest.approx(1 / 3)


def test_ndcg_defaults_to_k_of_ten():
    ranked = ["x"] * 9 + ["a"]
    assert ndcg_at_k(ranked, {"a"}) == ndcg_at_k(ranked, {"a"}, 10)
    assert ndcg_at_k(ranked, {"a"}, 9) == 0.0


def test_negative_k_is_rejected_rather_than_silently_slicing():
    # ranked[:-1] would quietly drop the last hit instead of failing.
    with pytest.raises(ValueError, match="non-negative"):
        recall_at_k(["a"], {"a"}, -1)
    with pytest.raises(ValueError, match="non-negative"):
        ndcg_at_k(["a"], {"a"}, -1)


# --- mean_ci -----------------------------------------------------------


def test_mean_ci_brackets_the_mean():
    values = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    mean, lo, hi = mean_ci(values)
    assert mean == pytest.approx(0.6)
    assert lo <= mean <= hi


def test_mean_ci_is_deterministic():
    values = [0.0, 1.0] * 15
    assert mean_ci(values) == mean_ci(values)


def test_mean_ci_stays_inside_the_metric_range():
    """The reason for bootstrapping rather than a t-interval: at n=8 with
    every value at 1.0 a normal approximation still reports a band, and
    at a boundary it reports one the metric cannot occupy."""
    mean, lo, hi = mean_ci([1.0] * 8)
    assert (mean, lo, hi) == (1.0, 1.0, 1.0)
    mean, lo, hi = mean_ci([0.0] * 8)
    assert (mean, lo, hi) == (0.0, 0.0, 0.0)


def test_mean_ci_narrows_as_n_grows():
    small = mean_ci([0.0, 1.0] * 5)
    large = mean_ci([0.0, 1.0] * 200)
    assert (large[2] - large[1]) < (small[2] - small[1])


def test_mean_ci_of_a_single_value_is_degenerate_not_fabricated():
    assert mean_ci([0.7]) == (0.7, 0.7, 0.7)


def test_mean_ci_refuses_an_empty_slice():
    with pytest.raises(ValueError, match="n=0"):
        mean_ci([])


def test_mean_ci_rejects_an_impossible_confidence():
    with pytest.raises(ValueError, match="confidence"):
        mean_ci([0.5, 0.5], confidence=1.0)
