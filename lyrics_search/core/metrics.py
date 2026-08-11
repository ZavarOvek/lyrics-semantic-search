"""EVAL-PREP §4: retrieval metrics as pure functions.

Recall@k, reciprocal rank (MRR is its mean), and nDCG@k. No I/O, no
dependency on how the query set was designed -- these take a ranked list
of song_ids and a set of relevant song_ids, and nothing else. That
independence is the point: the query scheme is decided separately and
must drop into this without any of it being rewritten.

Relevance is binary. The eval-set format (EVAL-PREP §5) records
`relevant_song_ids` as a flat list with no grades, so there is no graded
signal to use; nDCG with binary gains is still worth having over plain
precision because it is the one metric here that rewards *how much*
better a rank is, not just whether a relevant hit landed inside the
window.

Ranking is at song level, after `core/aggregate.py` has collapsed chunks
to songs, so `ranked` contains each song_id at most once. That is
assumed, not validated -- aggregation guarantees it.

Conventions, all four of which EVAL-PREP §4 pins down as property tests:
  - An empty `ranked` scores 0.0 for every metric. Not NaN, not an
    exception: a query that returned nothing is a query that retrieved
    nothing, and it has to be averageable together with the rest.
  - An empty `relevant` also scores 0.0. A query with no relevant song
    cannot be scored, and silently dropping it from a mean would inflate
    the mean; the loader (EVAL-PREP §5) is where such a record should be
    rejected, so reaching here at all is already unusual.
  - Recall@k is non-decreasing in k, by construction (the numerator is a
    set intersection over a growing prefix, the denominator is fixed).
  - Moving a relevant song to a better rank never lowers any of the
    three.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np


def recall_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of the relevant songs that appear in the top k.

    Denominator is |relevant|, not min(|relevant|, k) -- so a query with
    more relevant songs than k cannot reach 1.0. That is the honest
    reading of "recall": the cap is a property of the query set, and
    hiding it inside the metric would make a hard query look solved.
    """
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    relevant = set(relevant)
    if not relevant or not ranked:
        return 0.0
    found = relevant.intersection(ranked[:k])
    return len(found) / len(relevant)


def reciprocal_rank(ranked: Sequence[str], relevant: Iterable[str]) -> float:
    """1 / (1-based rank of the first relevant song), or 0.0 if none."""
    relevant = set(relevant)
    if not relevant:
        return 0.0
    for rank0, song_id in enumerate(ranked):
        if song_id in relevant:
            return 1.0 / (rank0 + 1)
    return 0.0


def mean_reciprocal_rank(
    per_query: Iterable[tuple[Sequence[str], Iterable[str]]],
) -> float:
    """MRR over (ranked, relevant) pairs. 0.0 over an empty iterable."""
    values = [reciprocal_rank(ranked, relevant) for ranked, relevant in per_query]
    return float(np.mean(values)) if values else 0.0


def _dcg(gains: Sequence[float]) -> float:
    # Standard discount: gain_i / log2(i + 2), i 0-based.
    return float(sum(g / np.log2(i + 2) for i, g in enumerate(gains)))


def ndcg_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int = 10) -> float:
    """Binary-gain nDCG@k, in [0, 1].

    The ideal ranking puts min(|relevant|, k) relevant songs at the top,
    so the denominator reflects the best achievable *within k* -- unlike
    recall_at_k, whose denominator deliberately does not. The difference
    is intentional: nDCG answers "how good is this ordering, given the
    window", recall answers "how much of the answer did we get at all".
    """
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    relevant = set(relevant)
    if not relevant or not ranked or k == 0:
        return 0.0

    gains = [1.0 if song_id in relevant else 0.0 for song_id in ranked[:k]]
    ideal_gains = [1.0] * min(len(relevant), k)

    idcg = _dcg(ideal_gains)
    # Unreachable via the public API: the guard above leaves `relevant`
    # non-empty and k >= 1, so ideal_gains has at least one 1.0 and
    # _dcg([1.0]) == 1.0. Kept as a free guard against future edits to the
    # branches above; not covered because faking it would test a mock.
    if idcg == 0.0:  # pragma: no cover
        return 0.0
    return _dcg(gains) / idcg


def mean_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    n_boot: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Mean and a bootstrap percentile confidence interval (EVAL-PREP §6).

    Returns (mean, lo, hi).

    Bootstrap rather than a normal-approximation interval because these
    are per-query metric values on [0, 1] that pile up at the boundaries
    -- Recall@1 is literally 0 or 1 per query -- and a slice can easily
    be a couple of dozen queries. A t-interval there routinely runs off
    the end of [0, 1] and reports a bound the metric cannot take.

    `seed` is fixed so the same slice reports the same interval twice; an
    eval table that shifts between runs of the same data is not a table
    anyone can check.

    Raises on an empty `values`: an empty slice has no mean to report,
    and the caller should show `n=0` rather than a fabricated number.
    """
    if not values:
        raise ValueError("mean_ci: no values -- report n=0 instead of an interval")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    if arr.size == 1:
        return mean, mean, mean

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot_means = arr[draws].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(boot_means, [tail, 1.0 - tail])
    return mean, float(lo), float(hi)
