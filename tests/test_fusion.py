"""SPEC-03 §3.3: Reciprocal Rank Fusion property/unit tests."""
from __future__ import annotations

from lyrics_search.core.fusion import RRF_DEFAULT_K, reciprocal_rank_fusion


def test_single_list_preserves_order():
    ranked = ["a", "b", "c"]
    fused = reciprocal_rank_fusion([ranked])
    assert [song_id for song_id, _ in fused] == ["a", "b", "c"]


def test_document_first_in_both_lists_stays_first():
    list_a = ["x", "a", "b"]
    list_b = ["x", "c", "d"]
    fused = reciprocal_rank_fusion([list_a, list_b])
    assert fused[0][0] == "x"


def test_score_matches_rrf_formula():
    fused = reciprocal_rank_fusion([["a", "b"]], k=60)
    scores = dict(fused)
    assert scores["a"] == 1.0 / (60 + 1)
    assert scores["b"] == 1.0 / (60 + 2)


def test_scores_sum_across_lists():
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "a"]], k=60)
    scores = dict(fused)
    expected_a = 1.0 / (60 + 1) + 1.0 / (60 + 2)
    expected_b = 1.0 / (60 + 2) + 1.0 / (60 + 1)
    assert scores["a"] == expected_a
    assert scores["b"] == expected_b
    # both ranked 1st once and 2nd once -> exactly tied
    assert scores["a"] == scores["b"]


def test_missing_from_one_list_only_contributes_its_present_terms():
    fused = reciprocal_rank_fusion([["a", "b"], ["a"]], k=60)
    scores = dict(fused)
    assert scores["a"] == 1.0 / (60 + 1) + 1.0 / (60 + 1)
    assert scores["b"] == 1.0 / (60 + 2)
    assert scores["a"] > scores["b"]


def test_hybrid_with_one_empty_branch_returns_other_branch_ranking():
    fused = reciprocal_rank_fusion([["a", "b", "c"], []])
    assert [song_id for song_id, _ in fused] == ["a", "b", "c"]


def test_both_branches_empty_returns_empty():
    assert reciprocal_rank_fusion([[], []]) == []


def test_default_k_is_60():
    assert RRF_DEFAULT_K == 60


def test_tie_break_is_deterministic_by_song_id():
    # both "b" and "a" tied at rank 1 in two disjoint single-item lists
    fused = reciprocal_rank_fusion([["b"], ["a"]])
    assert [song_id for song_id, _ in fused] == ["a", "b"]
