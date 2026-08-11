"""SPEC-03 §3.3: Reciprocal Rank Fusion for hybrid (dense + lexical) retrieval.

RRF, not a weighted sum of raw scores: cosine similarity (dense) and BM25
(lexical) live on incomparable scales that can't be correctly combined
without per-corpus calibration. RRF instead fuses on *rank*, which is
scale-invariant by construction:

    score(d) = sum over ranked lists r that contain d of  1 / (k + rank_r(d))

`rank_r(d)` is 1-based (the top hit in list r has rank 1). `k` dampens the
influence of low ranks and is configurable (default 60, the commonly-cited
value from the original RRF paper, also reused as HybridRetriever's default).

Pure function, no I/O: takes already-ranked song_id lists in, returns a
fused ranking out. Callers (HybridRetriever) are responsible for resolving
each song_id back to a Hit/best_chunk after fusion.
"""

from __future__ import annotations

from collections.abc import Sequence

RRF_DEFAULT_K = 60


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]],
    *,
    k: int = RRF_DEFAULT_K,
) -> list[tuple[str, float]]:
    """Fuse multiple ranked lists of song_ids into one.

    Each input list is assumed to already be sorted best-first (rank 1 =
    index 0). A song_id missing from a given list simply doesn't contribute
    that list's term to its score (not treated as an infinitely-bad rank).

    Returns (song_id, fused_score) pairs sorted by fused_score descending,
    with a deterministic tie-break: on an exact score tie, the song_id that
    appears at a better (numerically smaller) rank in more of the input
    lists sorts first; if still tied, sorted by song_id ascending for full
    determinism.
    """
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}  # for tie-breaking only
    rank_count: dict[str, int] = {}  # how many lists contain this id

    for ranked_list in ranked_lists:
        for rank0, song_id in enumerate(ranked_list):
            rank = rank0 + 1
            scores[song_id] = scores.get(song_id, 0.0) + 1.0 / (k + rank)
            rank_count[song_id] = rank_count.get(song_id, 0) + 1
            if song_id not in best_rank or rank < best_rank[song_id]:
                best_rank[song_id] = rank

    def sort_key(song_id: str) -> tuple[float, int, int, str]:
        # Higher score first, then present-in-more-lists first, then better
        # best_rank first, then song_id ascending -- all fully deterministic.
        return (-scores[song_id], -rank_count[song_id], best_rank[song_id], song_id)

    ordered_ids = sorted(scores.keys(), key=sort_key)
    return [(song_id, scores[song_id]) for song_id in ordered_ids]
