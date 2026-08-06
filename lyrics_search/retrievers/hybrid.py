"""SPEC-03 §3: HybridRetriever -- fuse DenseRetriever and LexicalRetriever
via Reciprocal Rank Fusion (core/fusion.py), NOT weighted score summation.

RRF is used specifically because dense (cosine-ish inner product) and
lexical (BM25) scores live on incomparable raw scales -- summing them with
hand-picked weights would be arbitrary and corpus-dependent, whereas RRF
only needs each branch's *rank order*, which is directly comparable.

branch_statuses records each branch's own SearchResult.status so a partial
degradation (e.g. dense query OOV for tfidf, but lexical still found
keyword matches) is visible in the result rather than silently absorbed
into one overall "ok".

best_chunk resolution when a song appears in both branches: dense wins
(its chunk-level score is what actually drove the aggregation-to-song
step in DenseRetriever, so it's the more information-rich pick); a
lexical-only song uses LexicalRetriever's own best_chunk resolution.

query_norm on the combined SearchResult is dense's query_norm (the
continuous, more information-rich of the two -- lexical's is only a
match *count*) -- this is a hybrid-specific convention, documented here
since SearchResult.query_norm's docstring describes the single-retriever
cases only.
"""
from __future__ import annotations

from lyrics_search.contracts import Hit, SearchResult
from lyrics_search.core.fusion import RRF_DEFAULT_K, reciprocal_rank_fusion


class HybridRetriever:
    name = "hybrid-rrf"

    def __init__(self, dense_retriever, lexical_retriever, *, rrf_k: int = RRF_DEFAULT_K):
        self._dense = dense_retriever
        self._lexical = lexical_retriever
        self._rrf_k = rrf_k

    def search(self, query: str, top_k: int) -> SearchResult:
        dense_result = self._dense.search(query, top_k)
        lexical_result = self._lexical.search(query, top_k)
        branch_statuses = {"dense": dense_result.status, "lexical": lexical_result.status}

        ranked_lists = []
        if dense_result.status == "ok" and dense_result.hits:
            ranked_lists.append([h.song_id for h in dense_result.hits])
        if lexical_result.status == "ok" and lexical_result.hits:
            ranked_lists.append([h.song_id for h in lexical_result.hits])

        if ranked_lists:
            status = "ok"
        elif branch_statuses["dense"] == "empty_query" and branch_statuses["lexical"] == "empty_query":
            status = "empty_query"
        else:
            status = "query_out_of_vocabulary"

        dense_chunk_by_song = {h.song_id: h.best_chunk for h in dense_result.hits}
        lexical_chunk_by_song = {h.song_id: h.best_chunk for h in lexical_result.hits}

        fused = reciprocal_rank_fusion(ranked_lists, k=self._rrf_k)
        hits = [
            Hit(
                song_id=song_id,
                score=score,
                best_chunk=dense_chunk_by_song.get(song_id) or lexical_chunk_by_song[song_id],
            )
            for song_id, score in fused
        ]

        return SearchResult(
            hits=hits,
            status=status,
            query_norm=dense_result.query_norm,
            branch_statuses=branch_statuses,
        )
