"""SPEC-02 §3: local JSONL ingestion source.

Reads RawSong-shaped records from a local .jsonl file (one JSON object per
line, keys matching RawSong fields with a couple of tolerant fallbacks).
Used for golden test fixtures (tests/golden/*.jsonl) and for re-ingesting a
previously produced raw.jsonl without re-hitting the network/HF Hub.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from lyrics_search.contracts import RawSong
from lyrics_search.core.text import make_song_id


@dataclass
class JsonlFileSource:
    path: Path | str

    def iter_songs(self) -> Iterator[RawSong]:
        path = Path(self.path)
        with open(path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                try:
                    artist = rec["artist"]
                    title = rec["title"]
                except KeyError as e:
                    raise ValueError(
                        f"{path}:{lineno}: record missing required field {e}"
                    ) from e
                text_raw = rec.get("text_raw", rec.get("lyrics", ""))
                yield RawSong(
                    song_id=rec.get("song_id") or make_song_id(artist, title),
                    artist=artist,
                    title=title,
                    text_raw=text_raw,
                    source=rec.get("source", f"jsonl:{path.name}"),
                    is_translation=rec.get("is_translation", False),
                )
