"""SUBAGENTS.md: pick the pilot fragments the query generator will work on.

Writes one record per song to `data/eval/pilot/fragments.jsonl`. Nothing is
printed but counts -- the fragments are copyrighted lyrics and must not
reach a terminal transcript.

Fragment choice follows EVAL-AUTO §2: a random chunk that passed the
filters, never automatically the first. Songs open with a chorus often
enough that taking position 0 would make the sample look homogeneous and
would over-represent the most repeated, least distinctive text in the
corpus.

Selection is seeded so the pilot can be repeated against a reworded
prompt and the difference read as the prompt's doing rather than the
sample's.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, ".")  # run from the repo root, as every other script here does

from lyrics_search.paths import corpus_dir, raw_path, work_dir

MIN_FRAGMENT_CHARS = 80


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--corpus", default="full")
    parser.add_argument("--chunking", default="sections")
    parser.add_argument("--songs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--out", default="data/eval/pilot/fragments.jsonl")
    args = parser.parse_args()

    root = Path(args.data_root)
    genre_file = corpus_dir(root, args.corpus) / "genre.jsonl"
    chunks_file = work_dir(root, args.corpus, args.chunking) / "chunks.jsonl"
    raw_file = raw_path(root, args.corpus)

    for required in (genre_file, chunks_file, raw_file):
        if not required.exists():
            raise FileNotFoundError(f"missing corpus artifact: {required}")

    out_path = Path(args.out)
    if out_path.exists():
        raise FileExistsError(
            f"{out_path} already exists. Pass --out elsewhere, or delete it "
            f"deliberately -- overwriting a pilot batch silently would make "
            f"the measured numbers unattributable to a prompt version."
        )

    genre = {r["song_id"]: r["genre"] for r in _read_jsonl(genre_file)}
    rng = random.Random(args.seed)
    chosen = set(rng.sample(sorted(genre), args.songs))

    by_song: dict[str, list[dict]] = {}
    for chunk in _read_jsonl(chunks_file):
        if chunk["song_id"] in chosen and len(chunk["text"]) >= MIN_FRAGMENT_CHARS:
            by_song.setdefault(chunk["song_id"], []).append(chunk)

    meta = {r["song_id"]: r for r in _read_jsonl(raw_file) if r["song_id"] in chosen}

    records = []
    for song_id in sorted(chosen):
        candidates = by_song.get(song_id)
        if not candidates:
            # Every chunk was shorter than the floor. Recorded, not
            # silently skipped -- a sample smaller than requested has to
            # be visible in the numbers.
            records.append({"song_id": song_id, "skipped": "no_chunk_over_min_chars"})
            continue
        chunk = rng.choice(candidates)
        song = meta[song_id]
        records.append(
            {
                "song_id": song_id,
                "chunk_id": chunk["chunk_id"],
                "position": chunk["position"],
                "section": chunk["section"],
                "split_by": chunk["split_by"],
                "force_split": chunk["force_split"],
                "genre": genre[song_id],
                "is_translation": song["is_translation"],
                "artist": song["artist"],
                "title": song["title"],
                "text": chunk["text"],
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    usable = [r for r in records if "skipped" not in r]
    print(f"songs requested : {args.songs}")
    print(f"fragments written: {len(usable)}")
    print(f"skipped         : {len(records) - len(usable)}")
    print(f"seed            : {args.seed}")
    print(f"out             : {out_path}")


if __name__ == "__main__":
    main()
