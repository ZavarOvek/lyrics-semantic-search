"""SPEC-03 §3: LexicalRetriever tests -- BM25 over whole-song text_raw,
best_chunk resolved via per-song idf-weighted chunk scoring.

Uses a 4-song fixture (not 2) deliberately: with only 2 documents, BM25's
idf formula (log(N - freq + 0.5) - log(freq + 0.5)) gives an idf of
*exactly* 0 for any term appearing in precisely 1 of 2 docs -- a
degenerate small-corpus artifact of the formula itself, not a bug in
LexicalRetriever, but it makes a 2-doc fixture useless for testing
ranking/scoring behavior (every score comes out 0). Two filler songs
with disjoint vocabulary give real, non-zero idf weights instead.
"""
from __future__ import annotations

import pytest

from lyrics_search.contracts import Chunk, RawSong
from lyrics_search.retrievers.lexical import LexicalRetriever

SONGS = {
    "s1": RawSong(
        song_id="s1", artist="A", title="One",
        text_raw="walking alone in the rain thinking about you all night long",
        source="test",
    ),
    "s2": RawSong(
        song_id="s2", artist="B", title="Two",
        text_raw="dancing in the summer sun with my friends all day",
        source="test",
    ),
    "s3": RawSong(
        song_id="s3", artist="C", title="Three",
        text_raw="a completely different filler song about nothing at all today",
        source="test",
    ),
    "s4": RawSong(
        song_id="s4", artist="D", title="Four",
        text_raw="another filler song with unrelated words entirely different",
        source="test",
    ),
}

CHUNKS = {
    "s1:0": Chunk(chunk_id="s1:0", song_id="s1", section="verse", text="walking alone in the rain", position=0, split_by="none"),
    "s1:1": Chunk(chunk_id="s1:1", song_id="s1", section="chorus", text="thinking about you all night long", position=1, split_by="none"),
    "s2:0": Chunk(chunk_id="s2:0", song_id="s2", section=None, text="dancing in the summer sun with my friends", position=0, split_by="none"),
    "s3:0": Chunk(chunk_id="s3:0", song_id="s3", section=None, text="a completely different filler song about nothing at all today", position=0, split_by="none"),
    "s4:0": Chunk(chunk_id="s4:0", song_id="s4", section=None, text="another filler song with unrelated words entirely different", position=0, split_by="none"),
}


def _build_retriever():
    return LexicalRetriever(SONGS, CHUNKS)


def test_search_returns_ok_with_matching_song():
    retriever = _build_retriever()
    result = retriever.search("walking alone rain", top_k=2)
    assert result.status == "ok"
    assert result.hits[0].song_id == "s1"


def test_search_best_chunk_is_most_relevant_chunk_of_matched_song():
    retriever = _build_retriever()
    result = retriever.search("thinking about you night", top_k=2)
    hit = next(h for h in result.hits if h.song_id == "s1")
    assert hit.best_chunk.chunk_id == "s1:1"


def test_search_ranks_song_with_more_keyword_overlap_first():
    retriever = _build_retriever()
    result = retriever.search("dancing summer sun friends", top_k=2)
    assert result.hits[0].song_id == "s2"


def test_search_empty_query_returns_empty_query_status():
    retriever = _build_retriever()
    result = retriever.search("   ", top_k=2)
    assert result.status == "empty_query"
    assert result.hits == []


def test_search_out_of_vocabulary_query_returns_oov_status():
    retriever = _build_retriever()
    result = retriever.search("zzzqqq xxxwww", top_k=2)
    assert result.status == "query_out_of_vocabulary"
    assert result.hits == []
    assert result.query_norm == 0.0


def test_search_query_norm_is_matched_token_count():
    retriever = _build_retriever()
    result = retriever.search("walking zzzqqq alone", top_k=2)
    # "walking" and "alone" are in-vocabulary, "zzzqqq" is not
    assert result.query_norm == 2.0


def test_best_chunk_for_song_raises_on_missing_chunks():
    retriever = LexicalRetriever(SONGS, {})  # no chunks at all
    with pytest.raises(KeyError, match="out of sync"):
        retriever.search("walking alone rain", top_k=2)
