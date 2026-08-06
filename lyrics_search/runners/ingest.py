"""SPEC-02 §3: ingest runner -- the imperative shell around a Source.

Pulls RawSong records from any Source implementation and writes them to
<out_dir>/raw.jsonl, one JSON object per line, in the order the Source
yields them. Byte-identical output across repeated runs is primarily the
Source's responsibility (see hf_dataset.py's sort-by-id); this runner adds
one more determinism/safety guarantee on top: two songs from the same
source resolving to the same song_id are not both silently written, since
Chunk.chunk_id = f"{song_id}:{position}" assumes song_id uniquely
identifies one RawSong's text downstream. First occurrence wins; the rest
are counted and reported, not silently dropped (SPEC-00 §3.2).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from lyrics_search.contracts import Source


def run_ingest(source: Source, out_dir: Path | str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw.jsonl"

    written = 0
    skipped_dupe = 0
    seen_ids: set[str] = set()
    with open(raw_path, "w", encoding="utf-8") as f:
        for song in source.iter_songs():
            if song.song_id in seen_ids:
                skipped_dupe += 1
                continue
            seen_ids.add(song.song_id)
            f.write(json.dumps(asdict(song), ensure_ascii=False) + "\n")
            written += 1

    print(f"Ingested {written} songs -> {raw_path}")
    if skipped_dupe:
        print(
            f"WARNING: skipped {skipped_dupe} song(s) with a song_id "
            "already seen from this source (duplicate artist+title, kept "
            "first occurrence)."
        )
    return raw_path


def main() -> None:
    import argparse

    from lyrics_search.sources.hf_dataset import HFDatasetSource

    parser = argparse.ArgumentParser(description="SPEC-02 ingest runner")
    parser.add_argument("--out", default="data/dev", help="output corpus dir")
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--full", action="store_true", help="ingest full dataset, ignore --sample-size"
    )
    args = parser.parse_args()

    source = HFDatasetSource(
        sample_size=None if args.full else args.sample_size,
        seed=args.seed,
    )
    run_ingest(source, args.out)


if __name__ == "__main__":
    main()
