"""One-off: convert the fastText .vec text file to gensim's native binary format.

    .venv/Scripts/python.exe scripts/convert_fasttext_vectors.py

PRE-EVAL-OPT §1. `cc.en.300.vec` is a 4.5 GB text file holding ~2M word
vectors parsed line by line. Loading the 500,000-word slice this project
uses costs ~178s, against 2.9-18.7s for every other arm's load. Both eval
tracks push thousands of queries through these models, so that cost is
paid on every run.

gensim's native format keeps the vector matrix in a plain `.npy` sidecar,
which `KeyedVectors.load(mmap='r')` maps from disk instead of reading and
parsing.

The vocabulary limit is baked in at conversion time, and that is the point
rather than a shortcut. `KeyedVectors.load()` takes no `limit` parameter,
so the artifact must already contain exactly the vocabulary the pipeline
uses today. Converting all ~2M vectors would pull words into scope that
are currently out of it, changing `_embed_one`'s word-average for any
chunk containing them -- and so changing embeddings that PRE-EVAL-OPT
explicitly does not rebuild. The limit therefore goes into the filename,
so an artifact built with a different one cannot be mistaken for this one.

Deterministic: `.vec` is ordered by descending corpus frequency and
`load_word2vec_format(limit=N)` takes the first N lines, so the same N
words come out on every run.

Does not delete the source `.vec` -- that decision belongs to the owner,
after the timings in the report.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")  # run from the repo root, as every other script here does

DEFAULT_VEC = "data/models/fasttext/cc.en.300.vec"
DEFAULT_LIMIT = 500_000  # the FastTextAvgEmbedder default, and what both configs use


def binary_path(vec_path: Path | str, limit: int) -> Path:
    """Artifact path for `vec_path` at `limit`, e.g. cc.en.300.limit500000.kv.

    Pure. The limit is in the name because the binary cannot be re-limited
    at load time, so it is the artifact's identity, not a load option.
    """
    vec_path = Path(vec_path)
    return vec_path.with_suffix(f".limit{limit}.kv")


def _dir_size(path: Path) -> int:
    """Total bytes of the artifact and any sidecars gensim wrote beside it."""
    return sum(p.stat().st_size for p in path.parent.glob(path.name + "*"))


def convert(
    vec_path: Path | str = DEFAULT_VEC,
    limit: int = DEFAULT_LIMIT,
    out_path: Path | str | None = None,
    overwrite: bool = False,
) -> Path:
    from gensim.models import KeyedVectors

    vec_path = Path(vec_path)
    if not vec_path.exists():
        raise FileNotFoundError(f"missing {vec_path}")
    out_path = Path(out_path) if out_path else binary_path(vec_path, limit)
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} already exists -- pass --overwrite to rebuild it.")

    print(f"[convert] reading {vec_path} (limit={limit}) ...", file=sys.stderr)
    t0 = time.time()
    kv = KeyedVectors.load_word2vec_format(str(vec_path), limit=limit)
    text_load_s = time.time() - t0
    print(
        f"[convert] parsed {len(kv.index_to_key)} words, dim={kv.vector_size}, "
        f"dtype={kv.vectors.dtype} in {text_load_s:.1f}s",
        file=sys.stderr,
    )

    if len(kv.index_to_key) != limit:
        raise ValueError(
            f"expected {limit} words from {vec_path}, got {len(kv.index_to_key)} "
            f"-- the source file is shorter than the requested limit."
        )

    kv.save(str(out_path))
    print(f"[convert] wrote {out_path} ({_dir_size(out_path) / 2**20:.1f} MiB)", file=sys.stderr)

    t0 = time.time()
    reloaded = KeyedVectors.load(str(out_path), mmap="r")
    binary_load_s = time.time() - t0
    print(f"[convert] reload with mmap='r': {binary_load_s:.1f}s", file=sys.stderr)

    if reloaded.index_to_key != kv.index_to_key:
        raise ValueError(
            f"{out_path}: vocabulary differs after the save/load round trip "
            f"-- the artifact does not reproduce the source ordering."
        )

    print(
        f"[convert] text {text_load_s:.1f}s -> binary {binary_load_s:.1f}s "
        f"({text_load_s / binary_load_s:.0f}x). Source .vec left in place.",
        file=sys.stderr,
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vec", default=DEFAULT_VEC)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--out", default=None, help="default: <vec stem>.limit<N>.kv")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    convert(args.vec, args.limit, args.out, args.overwrite)


if __name__ == "__main__":
    main()
