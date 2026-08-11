"""One-off: build data/<corpus>/genre.jsonl, the genre lookup for EVAL-PREP §6's genre slice.

    .venv/Scripts/python.exe scripts/build_genre_lookup.py --corpus full
    .venv/Scripts/python.exe scripts/build_genre_lookup.py --corpus dev

Provenance, which the report must state and this script cannot infer for
the reader: the value is the source dataset's own `tag` column
(mrYou/lyrics-dataset), not a classification made here. Phase 0 found it
takes exactly four values -- rock, country, rap, pop, near-balanced -- so
it is a dataset field, flat and closed, not something derived from the
lyrics. That makes it good to slice on and bad to call "genre" without
saying where it came from.

Why a separate artifact rather than a field on RawSong: adding one would
change no vector. Chunking reads `text_raw`, so re-ingesting with a new
metadata field produces byte-identical embeddings -- but the stage cache
keys on config, not on code, so the change would need `--force`, and
`--force` cascades through preprocess, embed and index for every
embedder. Tens of minutes of recompute for an unchanged result, plus a
field that only eval reads living in the structure the whole pipeline
passes through. Joining against the source at eval time instead would
mean pulling the dataset and recomputing every song_id on each run. A
lookup table written once is the cheap option, and it lands in data/,
outside git, like every other real-data artifact.

song_id is computed with the same `make_song_id` the ingest path uses, so
the ids line up by construction rather than by coincidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")  # run from the repo root, as every other script here does

from lyrics_search.core.text import make_song_id
from lyrics_search.genre import genre_path
from lyrics_search.paths import raw_path
from lyrics_search.sources.hf_dataset import DEFAULT_FILENAME, DEFAULT_REPO_ID

SOURCE_COLUMN = "tag"


def _corpus_song_ids(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"missing {path} -- run `build` for this corpus first.")
    ids = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["song_id"])
    return ids


def build_genre_lookup(corpus: str, data_root: Path | str = "data") -> Path:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    wanted = _corpus_song_ids(raw_path(data_root, corpus))

    path = hf_hub_download(repo_id=DEFAULT_REPO_ID, filename=DEFAULT_FILENAME, repo_type="dataset")
    df = pd.read_parquet(path).sort_values("id").reset_index(drop=True)

    genre: dict[str, str] = {}
    conflicts: list[tuple[str, str, str]] = []
    for row in df.itertuples():
        song_id = make_song_id(str(row.artist), str(row.title))
        if song_id not in wanted:
            continue
        tag = str(getattr(row, SOURCE_COLUMN))
        if song_id in genre:
            # Two source rows normalize to the same song_id (the corpus has
            # near-duplicate artist/title pairs -- see
            # scripts/dedup_check_full.py). Keep the first in the dataset's
            # own `id` order, which is the same order ingest uses, and count
            # the disagreements rather than picking silently.
            if genre[song_id] != tag:
                conflicts.append((song_id, genre[song_id], tag))
            continue
        genre[song_id] = tag

    out_path = genre_path(data_root, corpus)
    with open(out_path, "w", encoding="utf-8") as f:
        for song_id in sorted(genre):
            f.write(json.dumps({"song_id": song_id, "genre": genre[song_id]}) + "\n")

    counts: dict[str, int] = {}
    for tag in genre.values():
        counts[tag] = counts.get(tag, 0) + 1

    missing = len(wanted) - len(genre)
    print(f"Wrote {out_path}")
    print(
        f"  source column: {SOURCE_COLUMN!r} from {DEFAULT_REPO_ID} (dataset field, not inferred)"
    )
    print(f"  corpus songs: {len(wanted)}, covered: {len(genre)}, uncovered: {missing}")
    print(f"  distribution: {dict(sorted(counts.items(), key=lambda kv: -kv[1]))}")
    if conflicts:
        print(
            f"  song_id collisions with disagreeing tags: {len(conflicts)} "
            f"(first kept); e.g. {conflicts[:3]}"
        )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", required=True, choices=["dev", "full"])
    parser.add_argument("--data-root", default="data")
    args = parser.parse_args()
    build_genre_lookup(args.corpus, args.data_root)


if __name__ == "__main__":
    main()
