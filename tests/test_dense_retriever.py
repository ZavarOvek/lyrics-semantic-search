"""SPEC-03 §3: DenseRetriever tests -- embed, search index, aggregate to
songs; empty-query and OOV-query status handling."""
from __future__ import annotations

import numpy as np
import pytest

from lyrics_search.contracts import Chunk
from lyrics_search.embedders.tfidf import TfidfEmbedder
from lyrics_search.indexes.numpy_index import NumpyIndex
from lyrics_search.retrievers.dense import DenseRetriever

CHUNKS = [
    Chunk(chunk_id="s1:0", song_id="s1", section="verse", text="walking alone in the rain", position=0, split_by="none"),
    Chunk(chunk_id="s1:1", song_id="s1", section="chorus", text="thinking about you all night", position=1, split_by="none"),
    Chunk(chunk_id="s2:0", song_id="s2", section=None, text="dancing in the summer sun", position=0, split_by="none"),
]


def _build_retriever():
    embedder = TfidfEmbedder()
    embedder.fit([c.text for c in CHUNKS])
    vectors = embedder.encode([c.text for c in CHUNKS], is_query=False)
    index = NumpyIndex()
    index.build(vectors, [c.chunk_id for c in CHUNKS])
    chunk_lookup = {c.chunk_id: c for c in CHUNKS}
    return DenseRetriever(embedder, index, chunk_lookup)


def test_search_returns_ok_with_hits():
    retriever = _build_retriever()
    result = retriever.search("walking alone in the rain", top_k=3)
    assert result.status == "ok"
    assert result.hits[0].song_id == "s1"
    assert result.query_norm > 0


def test_search_aggregates_multiple_chunks_same_song_by_max():
    retriever = _build_retriever()
    result = retriever.search("walking rain thinking night", top_k=3)
    song_ids = [h.song_id for h in result.hits]
    assert song_ids.count("s1") == 1  # s1 has two matching chunks, must collapse to one Hit


def test_search_empty_query_returns_empty_query_status():
    retriever = _build_retriever()
    result = retriever.search("   ", top_k=3)
    assert result.status == "empty_query"
    assert result.hits == []
    assert result.query_norm == 0.0


def test_search_out_of_vocabulary_query_returns_oov_status():
    retriever = _build_retriever()
    result = retriever.search("zzzqqq xxxwww", top_k=3)
    assert result.status == "query_out_of_vocabulary"
    assert result.hits == []


def test_search_raises_on_chunk_id_not_in_lookup():
    embedder = TfidfEmbedder()
    embedder.fit([c.text for c in CHUNKS])
    vectors = embedder.encode([c.text for c in CHUNKS], is_query=False)
    index = NumpyIndex()
    index.build(vectors, [c.chunk_id for c in CHUNKS])
    incomplete_lookup = {"s1:0": CHUNKS[0]}  # missing s1:1, s2:0
    retriever = DenseRetriever(embedder, index, incomplete_lookup)
    with pytest.raises(KeyError, match="out of sync"):
        retriever.search("walking alone", top_k=3)
