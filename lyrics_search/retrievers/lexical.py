"""SPEC-03 §3: LexicalRetriever -- BM25 over whole-song text_raw.

Deliberately different granularity from DenseRetriever: dense searches
deduplicated *chunks*, lexical searches whole *songs*. A literal keyword
match anywhere in a song is a meaningful lexical signal even if the
matching passage happened to get dropped/deduped at chunk level, so
re-chunking text_raw for BM25 would throw away real recall for no benefit.

SPEC-03 §1's "never fit at query time" rule is about *learned* state
(tfidf's vocabulary, a neural model's weights) -- BM25's per-term IDF and
per-doc-length statistics are a deterministic, parameter-free function of
the corpus text itself (no held-out data, no randomness, nothing to
overfit), and are cheap enough to recompute from raw.jsonl (itself a
`build`-stage output, not something query-time invents) rather than
persist to disk: measured on the 500-song dev corpus, tokenizing + BM25Okapi
construction together took ~0.14s; linearly extrapolated to the ~30k-song
full corpus that's still only a few seconds, one-time per process -- the
same cost bucket as loading a neural embedder, not a per-query cost. So
unlike tfidf's vocabulary (persisted via save_fit/load_fit,
SPEC-02-PATCH item 3), LexicalRetriever rebuilds its BM25Okapi index fresh
every process start directly from the already-built raw.jsonl. This is a
deliberate design decision, not an oversight.

best_chunk resolution: BM25 itself only ranks whole songs, but Hit
requires a representative Chunk. For each matched song, this class scores
that song's own chunks by summing the corpus-wide BM25 idf weight of every
query token the chunk's text actually contains, and picks the
highest-scoring chunk (ties broken by earliest position) -- i.e. "which
part of this song most resembles the query", using the same idf weights
that ranked the song in the first place.
"""

from __future__ import annotations

from lyrics_search.contracts import Chunk, Hit, RawSong, SearchResult
from lyrics_search.core.text import tokenize_words


class LexicalRetriever:
    name = "lexical-bm25"

    def __init__(self, song_corpus: dict[str, RawSong], chunk_lookup: dict[str, Chunk]):
        from rank_bm25 import BM25Okapi

        self._song_ids = list(song_corpus.keys())
        tokenized_docs = [tokenize_words(song_corpus[sid].text_raw) for sid in self._song_ids]
        self._bm25 = BM25Okapi(tokenized_docs)

        self._song_chunks: dict[str, list[Chunk]] = {}
        for chunk in chunk_lookup.values():
            self._song_chunks.setdefault(chunk.song_id, []).append(chunk)
        for chunks in self._song_chunks.values():
            chunks.sort(key=lambda c: c.position)

    def _best_chunk_for_song(self, song_id: str, query_tokens: list[str]) -> Chunk:
        chunks = self._song_chunks.get(song_id)
        if not chunks:
            raise KeyError(
                f"song_id {song_id!r} matched by BM25 but has no chunks in "
                f"chunk_lookup -- chunks.jsonl and raw.jsonl are out of sync."
            )

        def chunk_score(chunk: Chunk) -> float:
            chunk_tokens = set(tokenize_words(chunk.text))
            return sum(self._bm25.idf.get(t, 0.0) for t in query_tokens if t in chunk_tokens)

        return max(chunks, key=lambda c: (chunk_score(c), -c.position))

    def search(self, query: str, top_k: int) -> SearchResult:
        normalized = query.strip()
        if not normalized:
            return SearchResult(hits=[], status="empty_query", query_norm=0.0)

        query_tokens = tokenize_words(normalized)
        matched = sum(1 for t in query_tokens if t in self._bm25.idf)
        if matched == 0:
            return SearchResult(hits=[], status="query_out_of_vocabulary", query_norm=0.0)

        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(
            ((sid, float(s)) for sid, s in zip(self._song_ids, scores) if s > 0.0),
            key=lambda pair: (-pair[1], pair[0]),
        )[:top_k]

        hits = [
            Hit(song_id=sid, score=score, best_chunk=self._best_chunk_for_song(sid, query_tokens))
            for sid, score in ranked
        ]
        return SearchResult(hits=hits, status="ok", query_norm=float(matched))
