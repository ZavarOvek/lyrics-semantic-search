"""SPEC-02 §6: brute-force NumpyIndex (dot product over normalized vectors).

Baseline exact-search index -- a direct port of phase0.py's search()
dot-product approach. O(n) per query; fine at dev-corpus/full-corpus
(~2951 / ~180k chunks) scale on CPU, not meant to scale past that -- see
FaissIndex for the approximate-search alternative.

SPEC-03 §6: NumpyIndex is deliberately exempt from the stage-cache
mechanism -- build() is ~0s (SPEC-02 measured it as "just stores the
array"), so there is no expensive work worth persisting. save()/load()
exist anyway for interface symmetry with FaissIndex and so the online
branch can always load *some* on-disk index without special-casing which
index type is configured, but callers using NumpyIndex are free to just
call build() fresh every time instead.

EVAL-PREP §1: NumpyIndex is the index that dispatches on the *type* of the
matrix it is given -- dense `np.ndarray` or `scipy.sparse` (see
contracts.Embedder for why both exist). It is also therefore the only
index a sparse embedder can use, since FaissIndex takes dense input only.
The scoring maths does not change: rows are L2-normalised in both
representations, so cosine similarity is a dot product either way. The
branch lives in core/vectors.py (arithmetic) and vector_store.py (file
format), not inline here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from lyrics_search.core.vectors import dot_scores, is_sparse
from lyrics_search.vector_store import load_matrix, save_matrix


class NumpyIndex:
    name = "numpy"

    def __init__(self) -> None:
        self._vectors: Any | None = None
        self._chunk_ids: list[str] = []

    def build(self, vectors: Any, chunk_ids: list[str]) -> None:
        if vectors.shape[0] != len(chunk_ids):
            raise ValueError(
                f"vectors/chunk_ids length mismatch: {vectors.shape[0]} != {len(chunk_ids)}"
            )
        if is_sparse(vectors):
            # tocsr() is a no-op when it already is CSR; it matters when the
            # matrix arrives as COO/CSC, where row slicing and the matvec
            # below would otherwise be needlessly slow.
            self._vectors = vectors.tocsr().astype(np.float32)
        else:
            self._vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self._chunk_ids = list(chunk_ids)

    def search(self, query_vector: Any, top_k: int) -> list[tuple[str, float]]:
        if self._vectors is None:
            raise RuntimeError("NumpyIndex.build() must be called before search()")
        scores = dot_scores(self._vectors, query_vector)
        top_idx = np.argsort(-scores)[:top_k]
        return [(self._chunk_ids[i], float(scores[i])) for i in top_idx]

    def save(self, dir_path: Path | str) -> None:
        if self._vectors is None:
            raise RuntimeError("Cannot save(): build() was never called.")
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        save_matrix(dir_path, self._vectors)
        with open(dir_path / "chunk_ids.json", "w", encoding="utf-8") as f:
            json.dump(self._chunk_ids, f)

    def load(self, dir_path: Path | str) -> None:
        dir_path = Path(dir_path)
        self._vectors = load_matrix(dir_path)
        with open(dir_path / "chunk_ids.json", encoding="utf-8") as f:
            self._chunk_ids = json.load(f)
