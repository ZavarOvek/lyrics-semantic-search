"""SPEC-03 §4: aggregate per-chunk scores into per-song Hits.

Dense retrieval scores individual chunks; a search result is over songs.
Song relevance = MAXIMUM over its chunks, not mean -- a mean would punish a
long song for having a few irrelevant verses even when one chunk is a
strong match. The chunk that produced the max becomes `Hit.best_chunk`, the
fragment shown to the user as the reason for the match.

Pure function, no I/O. Must degenerate correctly when every song has
exactly one chunk (SPEC-02-PATCH found bge-m3's real max_seq_length is
8192, making "whole song as one vector" a real future arm to compare in
eval -- this function must not need special-casing to support that).
"""

from __future__ import annotations

from typing import Sequence

from lyrics_search.contracts import Chunk, Hit


def aggregate_chunks_to_songs(chunk_scores: Sequence[tuple[Chunk, float]]) -> list[Hit]:
    """Reduce (chunk, score) pairs to one Hit per song_id (max score wins).

    Returns Hits sorted by score descending; ties broken by song_id
    ascending for determinism.
    """
    best_for_song: dict[str, tuple[Chunk, float]] = {}
    for chunk, score in chunk_scores:
        current = best_for_song.get(chunk.song_id)
        if current is None or score > current[1]:
            best_for_song[chunk.song_id] = (chunk, score)

    hits = [
        Hit(song_id=song_id, score=score, best_chunk=chunk)
        for song_id, (chunk, score) in best_for_song.items()
    ]
    hits.sort(key=lambda h: (-h.score, h.song_id))
    return hits
