"""SPEC-02 §5: embed runner -- chunks.jsonl -> per-embedder vectors + meta.json.

Imperative shell: loads chunk texts once, runs each requested embedder
over them, and writes normalized vectors plus a reproducibility meta.json
to data/<corpus>/embeddings/<embedder_name>/.

EVAL-PREP §1: an embedder may return a dense or a sparse matrix
(contracts.Embedder). Dense output is written as float16 `vectors.npy` as
it always was; sparse output as float32 `vectors.npz`. Everything between
encoding and writing -- norms, the no-signal drop, row selection -- goes
through core/vectors.py so this runner carries no `issparse` branch of its
own beyond the dtype choice at write time.

TfidfEmbedder and FastTextAvgEmbedder need `fit(corpus_texts)` before
`encode()` (see their docstrings); this runner calls fit() on the full
chunk-text list being embedded when the embedder exposes that method,
before encoding, and persists the fitted state alongside the vectors
(SPEC-02-PATCH item 3) so a later process can `encode()` a query with the
*same* fitted vocabulary/IDF without re-fitting (fit-at-query-time would
silently give a different vector space per process/corpus).

SPEC-02-PATCH item 4: chunks whose embedding comes out as (near-)zero
vectors carry no retrievable signal and are excluded from the vectors file
and chunk_ids.json, logged instead as chunk-level `chunk_no_signal` rejects
next to the embeddings (this is embedder-specific -- a chunk with no
signal for tfidf may be fine for bge-m3 -- so it is not folded into
preprocess's rejects.jsonl, which is embedder-agnostic).

SPEC-02-PATCH item 5: for embedders with a real tokenizer/max_length
(bge-m3, e5-base), this runner double-checks every chunk's *actual*
tokenized length against the model's real max_length and prints a real
WARNING only if the length-based cascade in core/sections.py somehow let
something through that a tokenizer would truncate. This is the only
WARNING this runner or preprocess.py emit; force_split counts are plain
information, not a warning (see runners/preprocess.py).
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from lyrics_search.core.rejects import Reject, RejectReason
from lyrics_search.core.vectors import is_sparse, row_norms, select_rows
from lyrics_search.vector_store import save_matrix

ZERO_NORM_EPS = 1e-6


def _load_chunk_texts(chunks_path: Path) -> tuple[list[str], list[str]]:
    ids, texts = [], []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ids.append(rec["chunk_id"])
            texts.append(rec["text"])
    return ids, texts


def _hash_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run_embed(
    embedder: Any,
    chunks_path: Path | str,
    out_dir: Path | str,
    *,
    cache_key: str | None = None,
    input_hash: str | None = None,
    stage_config: dict | None = None,
) -> Path:
    chunks_path = Path(chunks_path)
    out_dir = Path(out_dir) / embedder.name
    out_dir.mkdir(parents=True, exist_ok=True)

    chunk_ids, texts = _load_chunk_texts(chunks_path)
    if not texts:
        raise ValueError(f"No chunks found in {chunks_path}")

    # SPEC-02-PATCH item 3: fit (if the embedder needs it) once here, then
    # persist the fitted state to disk so a separate query-time process can
    # load_fit() the *same* state instead of silently re-fitting against
    # whatever corpus happens to be at hand.
    fitted_state_path: Path | None = None
    fitted_state_hash: str | None = None
    if hasattr(embedder, "fit"):
        embedder.fit(texts)
        fitted_state_path = out_dir / "fitted_state.joblib"
        embedder.save_fit(fitted_state_path)
        fitted_state_hash = _hash_file(fitted_state_path)

    uses_cuda = getattr(embedder, "device", None) == "cuda"

    t0 = time.time()
    peak_vram_mb = None
    torch = None
    if uses_cuda:
        import torch

        torch.cuda.reset_peak_memory_stats()

    vectors = embedder.encode(texts, is_query=False)

    if uses_cuda:
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2
    elapsed_s = time.time() - t0

    dim = vectors.shape[1]
    # EVAL-PREP §1: `vectors` may be dense or sparse (contracts.Embedder).
    # core/vectors.py carries the branch; everything downstream of these two
    # calls works on a plain dense 1-D array of norms either way.
    norms = row_norms(vectors)

    # SPEC-02-PATCH item 4: a (near-)zero vector carries no retrievable
    # signal for this embedder -- exclude it from vectors.npy/chunk_ids.json
    # rather than let it sit in the index as a spurious nearest-neighbour
    # magnet, and log it as a chunk_no_signal reject instead.
    zero_mask = norms <= ZERO_NORM_EPS
    dropped_no_signal = int(zero_mask.sum())
    if dropped_no_signal:
        rejects_path = out_dir / "rejects.jsonl"
        with open(rejects_path, "w", encoding="utf-8") as f:
            for cid in (chunk_ids[i] for i in np.nonzero(zero_mask)[0]):
                song_id = cid.rsplit(":", 1)[0]
                rec = Reject(
                    level="chunk",
                    reason=RejectReason.CHUNK_NO_SIGNAL,
                    song_id=song_id,
                    chunk_id=cid,
                    embedder_name=embedder.name,
                )
                d = {
                    "level": rec.level,
                    "reason": rec.reason.value,
                    "song_id": rec.song_id,
                    "chunk_id": rec.chunk_id,
                    "embedder_name": rec.embedder_name,
                }
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    keep_mask = ~zero_mask
    vectors = select_rows(vectors, keep_mask)
    chunk_ids = [cid for cid, keep in zip(chunk_ids, keep_mask, strict=True) if keep]
    kept_norms = norms[keep_mask]
    norm_ok = bool(np.allclose(kept_norms, 1.0, atol=1e-2)) if len(kept_norms) else True

    # SPEC-02-PATCH item 5: for embedders exposing a real tokenizer and
    # max_seq_length (bge-m3, e5-base), double-check every *surviving*
    # chunk's actual tokenized length against that real limit. The level-4
    # length-packing cascade in core/sections.py is designed to prevent
    # this from ever firing; a genuine WARNING here would mean the cascade
    # let something through that the real tokenizer would still truncate.
    tokenizer_max_length = getattr(embedder, "max_seq_length", None)
    token_max_observed = None
    token_overflow_count = 0
    if tokenizer_max_length is not None and hasattr(embedder, "token_lengths"):
        surviving_texts = [t for t, keep in zip(texts, keep_mask, strict=True) if keep]
        lengths = embedder.token_lengths(surviving_texts, is_query=False)
        token_max_observed = max(lengths) if lengths else 0
        overflow_idx = [i for i, n in enumerate(lengths) if n > tokenizer_max_length]
        token_overflow_count = len(overflow_idx)
        if token_overflow_count:
            sample = [chunk_ids[i] for i in overflow_idx[:5]]
            print(
                f"WARNING [{embedder.name}]: {token_overflow_count} chunk(s) exceed "
                f"the real tokenizer max_length={tokenizer_max_length} "
                f"(max observed={token_max_observed}); sample chunk_ids: {sample}"
            )

    ids_path = out_dir / "chunk_ids.json"
    meta_path = out_dir / "meta.json"

    # EVAL-PREP §1: dense matrices keep the float16 halving they always
    # had. Sparse ones stay float32: a CSR entry costs 4 bytes of value
    # plus 4 bytes of int32 column index, so float16 would shave 25% off an
    # artifact already measured in tens of MiB, and scipy's sparse matmul
    # does not support float16 anyway -- it would upcast on every query.
    # The file extension records which of the two was written (see
    # vector_store), so load_matrix() needs no flag from this meta.json.
    if is_sparse(vectors):
        dtype_saved = "float32"
    else:
        vectors = vectors.astype(np.float16)
        dtype_saved = "float16"
    vectors_path = save_matrix(out_dir, vectors)

    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(chunk_ids, f, ensure_ascii=False)

    meta = {
        "embedder_name": embedder.name,
        "dim": dim,
        "count": len(chunk_ids),
        "dropped_no_signal": dropped_no_signal,
        "chunks_source": str(chunks_path),
        "elapsed_s": round(elapsed_s, 2),
        "peak_vram_mb": round(peak_vram_mb, 1) if peak_vram_mb is not None else None,
        "norm_check_passed": norm_ok,
        "norm_min": float(kept_norms.min()) if len(kept_norms) else None,
        "norm_max": float(kept_norms.max()) if len(kept_norms) else None,
        "dtype_saved": dtype_saved,
        # EVAL-PREP §1: sparsity is reported, not just implied by the file
        # name -- it is the whole justification for the representation, and
        # item 2 asks for on-disk size per embedder alongside it.
        "sparse": is_sparse(vectors),
        "nnz": int(vectors.nnz) if is_sparse(vectors) else None,
        "vectors_file": vectors_path.name,
        "vectors_bytes": vectors_path.stat().st_size,
        # SPEC-02-PATCH item 3: presence/hash/dim of the persisted fitted
        # state, so a query-time process can verify it's loading the exact
        # state this index was built with, not silently re-fitting.
        "fitted_state_file": fitted_state_path.name if fitted_state_path else None,
        "fitted_state_sha1": fitted_state_hash,
        # SPEC-02-PATCH item 5: real-tokenizer overflow check results.
        "tokenizer_max_length": tokenizer_max_length,
        "token_max_observed": token_max_observed,
        "token_overflow_count": token_overflow_count,
    }
    # SPEC-03 §6: stage-cache fields, only present when the caller (build.py)
    # supplies them -- a bare/direct run_embed() call (e.g. existing tests,
    # or the CLI below) still works exactly as before and simply produces a
    # meta.json with no cache_key, which is_dir_stage_fresh() correctly
    # reads as "not fresh" rather than raising.
    if cache_key is not None:
        meta["cache_key"] = cache_key
    if input_hash is not None:
        meta["input_hash"] = input_hash
    if stage_config is not None:
        meta["stage_config"] = stage_config
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    size_mib = meta["vectors_bytes"] / 1024**2
    print(
        f"[{embedder.name}] {len(chunk_ids)} chunks -> dim={dim}, "
        f"{elapsed_s:.1f}s, norm_ok={norm_ok}, dropped_no_signal={dropped_no_signal}, "
        f"{meta['vectors_file']} {size_mib:.1f}MiB"
        + (f", nnz={meta['nnz']}" if meta["nnz"] is not None else "")
        + (f", peak_vram={peak_vram_mb:.0f}MiB" if peak_vram_mb is not None else "")
    )
    return out_dir


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="SPEC-02 embed runner")
    parser.add_argument("--chunks", default="data/dev/sections/chunks.jsonl")
    parser.add_argument("--out", default="data/dev/sections/embeddings")
    parser.add_argument(
        "--embedders",
        default="bge-m3,e5-base,tfidf",
        help="comma-separated: bge-m3,e5-base,tfidf,fasttext-avg",
    )
    parser.add_argument(
        "--fasttext-vectors",
        default=None,
        help="path to a local cc.en.300.vec file (required for fasttext-avg)",
    )
    args = parser.parse_args()

    requested = [e.strip() for e in args.embedders.split(",") if e.strip()]
    for name in requested:
        if name == "bge-m3":
            from lyrics_search.embedders.bge_m3 import BGEM3Embedder

            embedder = BGEM3Embedder()
        elif name == "e5-base":
            from lyrics_search.embedders.e5 import E5Embedder

            embedder = E5Embedder()
        elif name == "tfidf":
            from lyrics_search.embedders.tfidf import TfidfEmbedder

            embedder = TfidfEmbedder()
        elif name == "fasttext-avg":
            if not args.fasttext_vectors:
                raise SystemExit("--fasttext-vectors is required for fasttext-avg")
            from lyrics_search.embedders.fasttext_avg import FastTextAvgEmbedder

            embedder = FastTextAvgEmbedder(args.fasttext_vectors)
        else:
            raise SystemExit(f"Unknown embedder: {name}")

        run_embed(embedder, args.chunks, args.out)

        # Free GPU memory between embedders rather than holding multiple
        # loaded models resident at once (matters more on the full 30k
        # corpus run than here, but cheap to do unconditionally).
        if getattr(embedder, "device", None) == "cuda":
            del embedder
            import torch

            torch.cuda.empty_cache()
        else:
            del embedder


if __name__ == "__main__":
    main()
