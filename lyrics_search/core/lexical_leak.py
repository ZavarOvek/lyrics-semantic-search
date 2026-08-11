"""SUBAGENTS.md: how much of a generated query leaked from its source text.

The known-item eval track generates queries by paraphrasing a lyrics
fragment. If a paraphrase keeps the fragment's distinctive wording, BM25
finds the song by exact string match and the track measures the
generation procedure rather than the retrieval task. So every generated
query has to be checked against the fragment it came from, and the ones
that leaked have to be regenerated.

Raw word overlap is the wrong instrument. A shared common word ("the",
"night", "love") tells a lexical engine almost nothing -- thousands of
songs contain it. A shared rare word ("cordoba", "believin") is a near
unique key into one song. Counting both the same way would flag honest
paraphrases and pass the dangerous ones, because honest paraphrases are
*full* of shared common words and that is fine.

So overlap is weighted by IDF, taken from the fitted `tfidf` state -- the
same statistic the lexical baseline itself uses to decide what matters,
which is what makes the measurement mean something rather than being a
plausible-looking proxy.

Two numbers come out, and they answer different questions:

  - `ratio` -- the share of the query's total rarity mass that also
    occurs in the source. A diagnostic, good for tracking whether a
    prompt is improving across a batch.
  - `max_shared_idf` -- the rarest single term the query and source
    share. This is the gate. One rare shared word is enough to hand BM25
    an exact anchor, and a query can have a comfortable `ratio` while
    still carrying one fatal word, because a single term's weight gets
    diluted by every ordinary word around it.

Terms with no IDF entry are reported in `unknown` and excluded from both
weights rather than given a fallback weight. A term absent from the
corpus vocabulary cannot be matched by the lexical baseline at all, so it
can neither leak nor mask a leak; folding it into the denominator would
only dilute `ratio` and make leakage read as smaller than it is.

Both inputs are already-tokenised term sequences. Tokenisation belongs to
the caller precisely because it must be the *corpus* analyzer -- pairing
these IDF weights with a different tokeniser would silently compare terms
that were never the ones weighed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class LeakScore:
    """Result of one query-versus-source comparison.

    `ratio` is 0.0 when no query term carries a weight, which happens
    when the query is empty or made entirely of unknown terms. Zero, not
    NaN: a query that shares no weighted rarity has leaked nothing, and
    the value has to stay averageable across a batch.
    """

    ratio: float
    weight_shared: float
    weight_total: float
    max_shared_idf: float
    shared: tuple[str, ...]
    unknown: tuple[str, ...]


def idf_weighted_overlap(
    query_terms: Iterable[str],
    source_terms: Iterable[str],
    idf: Mapping[str, float],
) -> LeakScore:
    """Weigh the query terms that also occur in the source by their IDF.

    Terms are compared as sets, so a word repeated in the query counts
    once. Repetition is a property of phrasing, not of how much of an
    anchor the word gives a lexical engine -- BM25 saturates on term
    frequency anyway, and counting a word twice would let a clumsy
    paraphrase look twice as leaky as a terse one carrying the same
    word.
    """
    query_set = set(query_terms)
    source_set = set(source_terms)

    weighted = {t for t in query_set if t in idf}
    unknown = query_set - weighted
    shared = {t for t in weighted if t in source_set}

    weight_total = sum(idf[t] for t in weighted)
    weight_shared = sum(idf[t] for t in shared)

    return LeakScore(
        ratio=weight_shared / weight_total if weight_total else 0.0,
        weight_shared=weight_shared,
        weight_total=weight_total,
        max_shared_idf=max((idf[t] for t in shared), default=0.0),
        shared=tuple(sorted(shared)),
        unknown=tuple(sorted(unknown)),
    )
