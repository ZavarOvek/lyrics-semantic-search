"""SPEC-03 §1: query-time state loading.

The rule this module exists to enforce: the online (search) branch NEVER
fits or builds anything itself -- it only loads state a prior `build` run
already produced, and fails loudly (with the exact missing-artifact path)
on any gap, rather than silently degrading (e.g. falling back to an
unfit/untrained embedder, or building an index on the fly). This mirrors
SPEC-00 §3.2's fail-loudly rule, applied specifically to the online/offline
split.

Two extra checks beyond "does the file exist": for embedders with a
persisted fit (tfidf, fasttext-avg), the loaded fitted-state file's SHA1 is
re-hashed and compared against the hash embed.py recorded at build time
(catching silent post-build corruption/modification), and the embedder's
`dim` after `load_fit()` is compared against the `dim` embed.py recorded
(catching a fitted-state file that doesn't match the vectors it was
supposedly used to build).

`warmup()` is a mandatory one-shot throwaway encode() call. bge-m3/e5-base
already self-warm inside their own `__init__` (a cudnn kernel warm-up, see
their modules); this warmup is a separate, from-*loaded*-state check that
the object returned by `load_fitted_embedder()` responds correctly, and is
mandatory for every embedder including tfidf (which has no cudnn cost but
still benefits from catching a broken fit at load time instead of on the
first real user query).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from lyrics_search.contracts import Chunk, RawSong


def _hash_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_fitted_embedder(embedder, embeddings_dir: Path | str):
    """Load `embedder`'s persisted fit from `embeddings_dir/<embedder.name>/`,
    as written by runners/embed.py. No-op (returned unchanged) for
    embedders with no `load_fit` method (bge-m3, e5-base -- nothing to
    fit). Raises FileNotFoundError/ValueError on any missing artifact,
    hash mismatch, or dim mismatch -- never silently fits from scratch.
    """
    embeddings_dir = Path(embeddings_dir) / embedder.name
    meta_path = embeddings_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{embedder.name}: missing {meta_path} -- run `build` before `search`."
        )
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    if not hasattr(embedder, "load_fit"):
        return embedder

    fitted_state_file = meta.get("fitted_state_file")
    fitted_state_sha1 = meta.get("fitted_state_sha1")
    if not fitted_state_file:
        raise ValueError(
            f"{embedder.name}: {meta_path} has no fitted_state_file recorded "
            f"-- was it built by a pre-SPEC-02-PATCH embed runner?"
        )
    fitted_state_path = embeddings_dir / fitted_state_file
    if not fitted_state_path.exists():
        raise FileNotFoundError(
            f"{embedder.name}: missing fitted state file {fitted_state_path} "
            f"referenced by {meta_path} -- run `build` before `search`."
        )

    actual_hash = _hash_file(fitted_state_path)
    if fitted_state_sha1 is not None and actual_hash != fitted_state_sha1:
        raise ValueError(
            f"{embedder.name}: fitted state hash mismatch for {fitted_state_path} "
            f"({meta_path} says {fitted_state_sha1}, file is actually {actual_hash}) "
            f"-- it was modified or corrupted after being built."
        )

    embedder.load_fit(fitted_state_path)

    expected_dim = meta.get("dim")
    if expected_dim is not None and embedder.dim != expected_dim:
        raise ValueError(
            f"{embedder.name}: dim mismatch after load_fit() -- {meta_path} says "
            f"dim={expected_dim}, loaded embedder reports dim={embedder.dim}."
        )
    return embedder


def warmup(embedder) -> None:
    """Mandatory one-shot throwaway encode() call (SPEC-03 §1) -- surfaces
    a broken load at startup instead of on the first real user query."""
    embedder.encode(["warmup"], is_query=True)


def load_chunk_lookup(chunks_path: Path | str) -> dict[str, Chunk]:
    chunks_path = Path(chunks_path)
    if not chunks_path.exists():
        raise FileNotFoundError(f"missing {chunks_path} -- run `build` before `search`.")
    lookup: dict[str, Chunk] = {}
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lookup_rec = json.loads(line)
                chunk = Chunk(**lookup_rec)
                lookup[chunk.chunk_id] = chunk
    return lookup


def load_song_corpus(raw_path: Path | str) -> dict[str, RawSong]:
    raw_path = Path(raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"missing {raw_path} -- run `build` before `search`.")
    corpus: dict[str, RawSong] = {}
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                song = RawSong(**json.loads(line))
                corpus[song.song_id] = song
    return corpus


def load_index(index, index_dir: Path | str):
    """Load a previously-built index's on-disk state. `index` is an
    already-constructed index instance (e.g. from registry.resolve_index);
    this only calls its .load(), after confirming the directory itself
    exists so the failure message names the whole missing build output
    rather than an obscure missing single file inside a directory that
    was never even created.
    """
    index_dir = Path(index_dir)
    if not index_dir.exists():
        raise FileNotFoundError(
            f"{index.name}: missing index directory {index_dir} -- run `build` before `search`."
        )
    index.load(index_dir)
    return index
