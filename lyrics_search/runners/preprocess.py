"""SPEC-02 §4: preprocess runner -- raw.jsonl -> chunks.jsonl + rejects.jsonl.

Imperative shell: all I/O lives here. The actual section/chunk detection is
core.sections.chunk_song (pure); this runner adds translation flagging,
song/chunk-level filtering, reject logging, and the §4.7 invariant checks
on top.

Assumption (not explicit in the spec's stated preprocess outputs, flagged
here and in the final report): RawSong.is_translation is set during this
stage but Chunk has no such field, so the only place it can be persisted
is raw.jsonl itself. This runner rewrites raw.jsonl in place with
is_translation populated before chunking.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, replace
from pathlib import Path

from lyrics_search.contracts import Chunk, RawSong
from lyrics_search.core.dedupe import dedupe_blocks
from lyrics_search.core.rejects import Reject, RejectReason
from lyrics_search.core.sections import CHUNKING_SECTIONS, CHUNKING_STRATEGIES, chunk_song_with

# SPEC-02 §4.5: translation detection via title/artist pattern matching,
# not language-ID -- see notes/phase0-findings.md (6/500 dev-corpus songs
# were explicit translations despite language metadata all saying "en").
# Genius.com credits translation pages to a pseudo-artist like "Genius
# English Translations" / "Genius Türkçe Çeviri" etc., and/or the title
# itself carries an explicit "(... Translation)" suffix -- both patterns
# are covered by one case-insensitive "translation(s)" substring check
# across both fields.
_TRANSLATION_RE = re.compile(r"(?i)\btranslations?\b")

MIN_SONG_WORDS = 10  # song-level: below this, treat as a stub/empty page
MIN_CHUNK_CHARS = 15  # chunk-level: matches notes/phase0-findings.md threshold
_INSTRUMENTAL_RE = re.compile(r"(?i)^\s*\[?\s*instrumental\s*\]?\s*$")


def detect_is_translation(song: RawSong) -> bool:
    return bool(_TRANSLATION_RE.search(song.artist) or _TRANSLATION_RE.search(song.title))


def _song_reject_reason(song: RawSong) -> RejectReason | None:
    text = song.text_raw.strip()
    if not text:
        return RejectReason.SONG_EMPTY_TEXT
    if _INSTRUMENTAL_RE.match(text):
        return RejectReason.SONG_INSTRUMENTAL_ONLY
    if len(text.split()) < MIN_SONG_WORDS:
        return RejectReason.SONG_TOO_SHORT
    return None


def run_preprocess(
    raw_path: Path | str,
    out_dir: Path | str,
    chunking: str = CHUNKING_SECTIONS,
) -> None:
    """EVAL-PREP §3: `chunking` selects the strategy; everything after the
    segments are produced -- dedupe, chunk-level filtering, reject
    logging, the §4.7 invariants -- is deliberately shared, so the two
    arms differ only where they actually differ.

    One real difference falls out of that and is left visible rather than
    papered over: `dedupe_blocks` is a *within-song* filter over a song's
    segments, so under `whole_song` (one segment per song) it can never
    fire. The chunked arm drops repeated choruses; the whole-song arm
    keeps them inside the single chunk. That is a property of the arms,
    not an inconsistency to fix.
    """
    if chunking not in CHUNKING_STRATEGIES:
        raise ValueError(
            f"unknown chunking strategy {chunking!r}; "
            f"expected one of {', '.join(CHUNKING_STRATEGIES)}"
        )
    raw_path = Path(raw_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = out_dir / "chunks.jsonl"
    rejects_path = out_dir / "rejects.jsonl"

    songs: list[RawSong] = []
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                songs.append(RawSong(**json.loads(line)))

    translation_count = 0
    for i, song in enumerate(songs):
        is_t = detect_is_translation(song)
        if is_t:
            translation_count += 1
        if is_t != song.is_translation:
            songs[i] = replace(song, is_translation=is_t)

    with open(raw_path, "w", encoding="utf-8") as f:
        for song in songs:
            f.write(json.dumps(asdict(song), ensure_ascii=False) + "\n")

    rejects: list[Reject] = []
    passed_song_ids: set[str] = set()
    split_by_counts: dict[str, int] = {}  # pre-filter, all segments
    split_by_force_counts: dict[
        tuple[str, bool], int
    ] = {}  # pre-filter, cross-tab (SPEC-02-PATCH item 6)
    force_split_events = (
        0  # count of warn() calls, i.e. oversized segments that triggered level-4 packing
    )
    pre_filter_chunk_total = 0
    surviving_chunks: list[Chunk] = []

    for song in songs:
        reason = _song_reject_reason(song)
        if reason is not None:
            rejects.append(
                Reject(
                    level="song",
                    reason=reason,
                    song_id=song.song_id,
                    artist=song.artist,
                    title=song.title,
                )
            )
            continue
        passed_song_ids.add(song.song_id)

        warns: list[int] = []
        # B023: `warns` is rebound each iteration and the callback is consumed
        # synchronously inside chunk_song_with, so it never outlives the loop body.
        segments = chunk_song_with(
            song.text_raw,
            chunking,
            warn=lambda wc: warns.append(wc),  # noqa: B023
        )
        force_split_events += len(warns)
        pre_filter_chunk_total += len(segments)

        _keep_idx, dup_idx = dedupe_blocks([seg.text for seg in segments])
        dup_idx_set = set(dup_idx)

        for position, seg in enumerate(segments):
            split_by_counts[seg.split_by] = split_by_counts.get(seg.split_by, 0) + 1
            key = (seg.split_by, seg.force_split)
            split_by_force_counts[key] = split_by_force_counts.get(key, 0) + 1

            if position in dup_idx_set:
                rejects.append(
                    Reject(
                        level="chunk",
                        reason=RejectReason.CHUNK_DUPLICATE_BLOCK,
                        song_id=song.song_id,
                        artist=song.artist,
                        title=song.title,
                        position=position,
                        text_preview=seg.text[:80],
                    )
                )
                continue
            if len(seg.text) < MIN_CHUNK_CHARS:
                rejects.append(
                    Reject(
                        level="chunk",
                        reason=RejectReason.CHUNK_TOO_SHORT,
                        song_id=song.song_id,
                        artist=song.artist,
                        title=song.title,
                        position=position,
                        text_preview=seg.text[:80],
                    )
                )
                continue

            surviving_chunks.append(
                Chunk(
                    chunk_id=f"{song.song_id}:{position}",
                    song_id=song.song_id,
                    section=seg.section,
                    text=seg.text,
                    position=position,
                    split_by=seg.split_by,
                    force_split=seg.force_split,
                )
            )

    with open(chunks_path, "w", encoding="utf-8") as f:
        for c in surviving_chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    with open(rejects_path, "w", encoding="utf-8") as f:
        for r in rejects:
            d = asdict(r)
            d["reason"] = r.reason.value
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    song_rejects = [r for r in rejects if r.level == "song"]
    chunk_rejects = [r for r in rejects if r.level == "chunk"]

    # SPEC-02 §4.7 invariants -- fail loudly (SPEC-00 §3.2) rather than
    # write out silently-inconsistent artifacts.
    assert len(passed_song_ids) + len(song_rejects) == len(songs), (
        f"song-count invariant violated: {len(passed_song_ids)} passed + "
        f"{len(song_rejects)} rejected != {len(songs)} total"
    )
    assert len(surviving_chunks) + len(chunk_rejects) == pre_filter_chunk_total, (
        f"chunk-count invariant violated: {len(surviving_chunks)} kept + "
        f"{len(chunk_rejects)} rejected != {pre_filter_chunk_total} generated"
    )

    print(f"Chunking strategy: {chunking}")
    print(
        f"Songs: {len(songs)} total, {len(passed_song_ids)} passed, "
        f"{len(song_rejects)} rejected at song level"
    )
    print(f"  is_translation: {translation_count}/{len(songs)}")
    print(
        f"Chunks: {pre_filter_chunk_total} generated, {len(surviving_chunks)} kept, "
        f"{len(chunk_rejects)} rejected at chunk level"
    )
    print(f"  split_by distribution: {split_by_counts}")
    # SPEC-02-PATCH item 5: this is an INFO-level count of the cascade
    # doing its job (level-4 length packing firing on top of a structural
    # split, or on a whole song with none), not a WARNING -- it cannot by
    # itself mean anything is silently truncated or broken. See
    # runners/embed.py for the *real* WARNING, checked against each
    # model's actual tokenizer max_length.
    print(f"  force_split events (level-4 length packing triggered): {force_split_events}")
    print(
        f"  split_by x force_split cross-tab (pre-filter): "
        f"{ {f'{k[0]}/force_split={k[1]}': v for k, v in sorted(split_by_force_counts.items())} }"
    )
    print(f"Wrote {chunks_path} and {rejects_path}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="SPEC-02 preprocess runner")
    parser.add_argument("--raw", default="data/dev/raw.jsonl")
    parser.add_argument("--out", default="data/dev/sections")
    parser.add_argument("--chunking", default=CHUNKING_SECTIONS, choices=list(CHUNKING_STRATEGIES))
    args = parser.parse_args()
    run_preprocess(args.raw, args.out, args.chunking)


if __name__ == "__main__":
    main()
