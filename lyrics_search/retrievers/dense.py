"""SPEC-03 §3: DenseRetriever -- embed the query, search a chunk-level
vector index, aggregate matching chunks up to songs.

Requires an already-loaded-and-warmed embedder and an already-loaded index
(see retrievers/loading.py) -- this class does no loading/fitting/building
itself, per SPEC-03 §1.
"""

from __future__ import annotations

from lyrics_search.contracts import Chunk, SearchResult
from lyrics_search.core.aggregate import aggregate_chunks_to_songs
from lyrics_search.core.vectors import vector_norm

# Matches runners/embed.py's ZERO_NORM_EPS: a query vector at ~zero norm
# carries no retrievable signal (e.g. every query word was out of a
# tfidf/fasttext-avg vocabulary) -- treated as OOV rather than searched,
# so the index isn't queried with a meaningless all-near-zero vector.
QUERY_OOV_NORM_EPS = 1e-6


class DenseRetriever:
    name = "dense"

    def __init__(self, embedder, index, chunk_lookup: dict[str, Chunk]):
        self._embedder = embedder
        self._index = index
        self._chunk_lookup = chunk_lookup

    def search(self, query: str, top_k: int) -> SearchResult:
        normalized = query.strip()
        if not normalized:
            return SearchResult(hits=[], status="empty_query", query_norm=0.0)

        # EVAL-PREP §1: encode() returns a dense array or a sparse matrix,
        # so row 0 is a 1-D array in one case and a 1xN sparse row in the
        # other. vector_norm() covers both, and the index takes either --
        # dot_scores() densifies the single query row, which is bounded by
        # the vocabulary size (~300 KiB at tfidf's full-corpus dim), not by
        # corpus size.
        vec = self._embedder.encode([normalized], is_query=True)[0]
        norm = vector_norm(vec)
        if norm <= QUERY_OOV_NORM_EPS:
            return SearchResult(hits=[], status="query_out_of_vocabulary", query_norm=norm)

        raw_hits = self._index.search(vec, top_k)
        chunk_scores = []
        for chunk_id, score in raw_hits:
            if chunk_id not in self._chunk_lookup:
                raise KeyError(
                    f"chunk_id {chunk_id!r} returned by index but missing from "
                    f"chunk_lookup -- the index and chunks.jsonl are out of sync."
                )
            chunk_scores.append((self._chunk_lookup[chunk_id], score))

        hits = aggregate_chunks_to_songs(chunk_scores)
        return SearchResult(hits=hits, status="ok", query_norm=norm)
