"""SPEC-02 §3: HuggingFace dataset ingestion source.

Ports the loading logic from phase0.py's build_corpus() (mrYou/lyrics-dataset,
pulled via hf_hub_download + parquet -- brunokreiner/genius-lyrics, the
original SPEC-01 first choice, returned 401 on file download despite public
metadata; not retried, see notes/phase0-findings.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from lyrics_search.contracts import RawSong
from lyrics_search.core.text import make_song_id

DEFAULT_REPO_ID = "mrYou/lyrics-dataset"
DEFAULT_FILENAME = "data/train-00000-of-00001.parquet"


@dataclass
class HFDatasetSource:
    """Source protocol implementation over a HF Hub parquet dataset.

    sample_size=None pulls the full dataset (~30k songs); an int takes a
    seeded random sample (used for the dev corpus).

    Determinism: after sampling, rows are re-sorted by the dataset's own
    `id` column before yielding. That column is stable across runs (it
    comes from the source parquet file itself), so raw.jsonl is
    byte-identical run-to-run regardless of how `df.sample` internally
    orders its output -- sorting removes that as a variable rather than
    relying on it.
    """

    repo_id: str = DEFAULT_REPO_ID
    filename: str = DEFAULT_FILENAME
    repo_type: str = "dataset"
    sample_size: int | None = None
    seed: int = 42

    def iter_songs(self) -> Iterator[RawSong]:
        import pandas as pd
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=self.repo_id, filename=self.filename, repo_type=self.repo_type
        )
        df = pd.read_parquet(path)
        if self.sample_size is not None:
            df = df.sample(n=self.sample_size, random_state=self.seed)
        df = df.sort_values("id").reset_index(drop=True)

        for row in df.itertuples():
            artist = str(row.artist)
            title = str(row.title)
            lyrics = "" if row.lyrics is None else str(row.lyrics)
            yield RawSong(
                song_id=make_song_id(artist, title),
                artist=artist,
                title=title,
                text_raw=lyrics,
                source=f"hf:{self.repo_id}",
            )
