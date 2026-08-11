"""SPEC-03 §4: chunk -> song aggregation tests."""

from __future__ import annotations

from lyrics_search.contracts import Chunk
from lyrics_search.core.aggregate import aggregate_chunks_to_songs


def _chunk(song_id: str, position: int, text: str = "some lyrics") -> Chunk:
    return Chunk(
        chunk_id=f"{song_id}:{position}",
        song_id=song_id,
        section=None,
        text=text,
        position=position,
        split_by="none",
    )


def test_max_score_wins_not_mean():
    chunks = [
        (_chunk("song1", 0), 0.9),
        (_chunk("song1", 1), 0.1),
    ]
    hits = aggregate_chunks_to_songs(chunks)
    assert len(hits) == 1
    assert hits[0].score == 0.9
    assert hits[0].best_chunk.position == 0


def test_song_never_appears_twice():
    chunks = [
        (_chunk("song1", 0), 0.5),
        (_chunk("song1", 1), 0.7),
        (_chunk("song2", 0), 0.3),
    ]
    hits = aggregate_chunks_to_songs(chunks)
    song_ids = [h.song_id for h in hits]
    assert len(song_ids) == len(set(song_ids))
    assert set(song_ids) == {"song1", "song2"}


def test_best_chunk_always_belongs_to_same_song():
    chunks = [
        (_chunk("song1", 0), 0.5),
        (_chunk("song1", 1), 0.9),
        (_chunk("song2", 0), 0.3),
    ]
    hits = aggregate_chunks_to_songs(chunks)
    for hit in hits:
        assert hit.best_chunk.song_id == hit.song_id


def test_sorted_by_score_descending():
    chunks = [
        (_chunk("low", 0), 0.1),
        (_chunk("high", 0), 0.9),
        (_chunk("mid", 0), 0.5),
    ]
    hits = aggregate_chunks_to_songs(chunks)
    assert [h.song_id for h in hits] == ["high", "mid", "low"]


def test_one_chunk_per_song_degenerate_case():
    """Not hypothetical -- bge-m3's real max_seq_length is 8192 (SPEC-02-PATCH
    item 5), so 'whole song as one chunk' is a real future arm. Aggregation
    must not require special handling for it."""
    chunks = [
        (_chunk("song1", 0), 0.4),
        (_chunk("song2", 0), 0.8),
        (_chunk("song3", 0), 0.6),
    ]
    hits = aggregate_chunks_to_songs(chunks)
    assert len(hits) == 3
    assert [h.song_id for h in hits] == ["song2", "song3", "song1"]
    for hit in hits:
        assert hit.best_chunk.position == 0


def test_empty_input_returns_empty():
    assert aggregate_chunks_to_songs([]) == []


def test_tie_break_by_song_id():
    chunks = [
        (_chunk("zsong", 0), 0.5),
        (_chunk("asong", 0), 0.5),
    ]
    hits = aggregate_chunks_to_songs(chunks)
    assert [h.song_id for h in hits] == ["asong", "zsong"]
