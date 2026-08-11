"""SPEC-02 §4.6-4.7: reject reasons and the two-level reject record.

Pure data definitions -- no I/O. Used by runners/preprocess.py to log
every song or chunk it drops, with a fixed reason rather than a free-form
string, so rejects.jsonl stays machine-checkable (the §4.7 invariant
checks rely on being able to count rejects by level).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RejectReason(str, Enum):
    # level="song"
    SONG_EMPTY_TEXT = "song_empty_text"
    SONG_TOO_SHORT = "song_too_short"
    SONG_INSTRUMENTAL_ONLY = "song_instrumental_only"
    # level="chunk"
    CHUNK_TOO_SHORT = "chunk_too_short"
    CHUNK_DUPLICATE_BLOCK = "chunk_duplicate_block"
    # SPEC-02-PATCH item 4: embedder-specific -- a chunk that encodes to a
    # (near-)zero vector under one embedder may be fine under another, so
    # this is logged per-embedder by runners/embed.py, not by preprocess.py.
    CHUNK_NO_SIGNAL = "chunk_no_signal"


@dataclass(frozen=True)
class Reject:
    level: str  # "song" | "chunk"
    reason: RejectReason
    song_id: str
    # SPEC-02-PATCH item 4: artist/title default to "" because runners/embed.py
    # only has chunk_id/song_id/text available (chunks.jsonl carries no
    # artist/title) when it logs a chunk_no_signal reject -- unlike
    # preprocess.py, which always has the full RawSong on hand.
    artist: str = ""
    title: str = ""
    position: int | None = None  # chunk-level only: pre-filter position
    text_preview: str | None = None  # chunk-level only: first ~80 chars
    chunk_id: str | None = None  # embed-level only: which chunk (SPEC-02-PATCH item 4)
    embedder_name: str | None = None  # embed-level only: which embedder flagged it
