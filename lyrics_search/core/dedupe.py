"""SPEC-02 §4.4: dedup of repeated blocks within a single song.

Operates on normalized text only (see core.text.normalize_for_dedupe);
callers keep the original, un-normalized text for the surviving chunk.
"""

from __future__ import annotations

from lyrics_search.core.text import hash_block


def dedupe_blocks(texts: list[str]) -> tuple[list[int], list[int]]:
    """Return (keep_indices, duplicate_indices) over `texts`, in original
    order, keeping the first occurrence of each normalized block.
    """
    seen: set[str] = set()
    keep: list[int] = []
    dup: list[int] = []
    for i, text in enumerate(texts):
        h = hash_block(text)
        if h in seen:
            dup.append(i)
        else:
            seen.add(h)
            keep.append(i)
    return keep, dup
