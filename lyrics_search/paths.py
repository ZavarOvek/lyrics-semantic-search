"""On-disk layout of the data root (EVAL-PREP §3).

    data/
      <corpus>/                 dev | full | demo
        raw.jsonl               ingest output -- shared by every chunking arm
        raw.jsonl.meta.json
        <chunking>/             sections | whole_song
          chunks.jsonl
          rejects.jsonl
          embeddings/<embedder>/
          indexes/<embedder>/<index>/

`raw.jsonl` sits at the corpus root because ingest is the one stage that
does not depend on the chunking strategy: both arms read the same songs,
and duplicating them per arm would double the largest text artifact for
nothing. Everything downstream of chunking is per-arm, symmetrically --
including the default `sections`, which is named in the tree like any
other arm rather than being the unlabelled one at the corpus root.

This module is deliberately dependency-free (Path only) so build, search,
eval and one-off scripts can all agree on the layout without importing a
runner.
"""

from __future__ import annotations

from pathlib import Path


def corpus_dir(data_root: Path | str, corpus: str) -> Path:
    return Path(data_root) / corpus


def raw_path(data_root: Path | str, corpus: str) -> Path:
    return corpus_dir(data_root, corpus) / "raw.jsonl"


def work_dir(data_root: Path | str, corpus: str, chunking: str) -> Path:
    """Where every artifact downstream of chunking lives."""
    return corpus_dir(data_root, corpus) / chunking
