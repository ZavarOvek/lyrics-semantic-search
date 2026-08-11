"""EVAL-PREP §6: reading the genre lookup that scripts/build_genre_lookup.py writes.

`data/<corpus>/genre.jsonl`, one `{"song_id": ..., "genre": ...}` per
line. Deliberately not part of the pipeline: nothing in ingest,
preprocess, embed, index or search reads it, and no vector depends on it.
It is analysis metadata, loaded by the eval runner and nowhere else.

The value is the source dataset's own `tag` column, not a classification
made in this project -- see the builder script's docstring for why that
distinction matters when reporting a genre breakdown.

The reader lives here rather than in scripts/ so `lyrics_search` never
imports from a one-off script; the script imports this instead.
"""

from __future__ import annotations

import json
from pathlib import Path

from lyrics_search.paths import corpus_dir

GENRE_FILENAME = "genre.jsonl"


def genre_path(data_root: Path | str, corpus: str) -> Path:
    return corpus_dir(data_root, corpus) / GENRE_FILENAME


def load_genre_lookup(data_root: Path | str, corpus: str) -> dict[str, str]:
    path = genre_path(data_root, corpus)
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path} -- run `python scripts/build_genre_lookup.py --corpus {corpus}` first."
        )
    lookup: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                lookup[record["song_id"]] = record["genre"]
    return lookup
