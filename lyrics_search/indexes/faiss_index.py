"""SPEC-02 §6: FaissIndex -- approximate nearest-neighbor search via faiss HNSW.

Uses IndexHNSWFlat with inner-product metric: every embedder in this
project L2-normalizes its output vectors, so inner product is equivalent
to cosine similarity on the unit sphere. Approximate, so results can
differ from NumpyIndex's exact brute-force search -- see the dev-corpus
top-10 comparison in notes/ for how much they actually diverge in
practice.

SPEC-03 §6/§7: unlike NumpyIndex, FaissIndex's build() is genuinely
expensive (~187s one-time HNSW graph construction at full-corpus scale,
SPEC-02 measurement) -- exactly the kind of stage worth caching so `search`
never has to pay it again. save()/load() persist the index plus a
chunk_ids.json sidecar (faiss has no concept of our string chunk ids, only
integer positions). `efSearch` is a runtime HNSW search-quality parameter,
not part of the serialized graph itself, so load() re-applies it from this
instance's own construction-time setting after reading the index back.

Deliberately uses faiss.serialize_index()/deserialize_index() (in-memory
bytes, written to disk via Python's own open()) rather than
faiss.write_index()/read_index() (which fopen()s the path directly in
C++). Found by testing during this phase: faiss's C++ file I/O fails to
open any path containing non-ASCII characters on Windows -- and this
project was developed under a checkout whose root directory contains
Cyrillic characters, so write_index()/read_index() would have failed on
every real save/load there, not just in a contrived test. Any user whose
checkout path is non-Latin hits the same wall. serialize_index() avoids
touching the filesystem from C++ at all, sidestepping the issue entirely.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lyrics_search.core.vectors import is_sparse


class FaissIndex:
    name = "faiss-hnsw"

    def __init__(self, m: int = 32, ef_construction: int = 200, ef_search: int = 64):
        self._m = m
        self._ef_construction = ef_construction
        self._ef_search = ef_search
        self._index = None
        self._chunk_ids: list[str] = []

    def build(self, vectors: np.ndarray, chunk_ids: list[str]) -> None:
        import faiss

        # EVAL-PREP §1: faiss takes dense float32 only. Densifying a sparse
        # matrix here would silently turn a 39 MiB artifact into a 51.8 GiB
        # allocation, so refuse instead -- a sparse embedder runs with the
        # `numpy` index, which is a property of the method rather than a
        # gap to be papered over.
        if is_sparse(vectors):
            raise TypeError(
                f"{self.name}: cannot index a sparse matrix "
                f"({type(vectors).__name__}). faiss requires dense float32 "
                f"input, and densifying is not a safe default at corpus "
                f"scale. Use index: numpy for sparse embedders (tfidf)."
            )
        if vectors.shape[0] != len(chunk_ids):
            raise ValueError(
                f"vectors/chunk_ids length mismatch: {vectors.shape[0]} != {len(chunk_ids)}"
            )
        dim = vectors.shape[1]
        index = faiss.IndexHNSWFlat(dim, self._m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = self._ef_construction
        index.hnsw.efSearch = self._ef_search
        index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        self._index = index
        self._chunk_ids = list(chunk_ids)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        if self._index is None:
            raise RuntimeError("FaissIndex.build() must be called before search()")
        q = np.ascontiguousarray(query_vector, dtype=np.float32).reshape(1, -1)
        scores, idx = self._index.search(q, top_k)
        return [
            (self._chunk_ids[i], float(s)) for s, i in zip(scores[0], idx[0]) if i != -1
        ]

    def save(self, dir_path: Path | str) -> None:
        import faiss

        if self._index is None:
            raise RuntimeError("Cannot save(): build() was never called.")
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        raw_bytes = faiss.serialize_index(self._index).tobytes()
        with open(dir_path / "index.faiss", "wb") as f:
            f.write(raw_bytes)
        with open(dir_path / "chunk_ids.json", "w", encoding="utf-8") as f:
            json.dump(self._chunk_ids, f)

    def load(self, dir_path: Path | str) -> None:
        import faiss

        dir_path = Path(dir_path)
        with open(dir_path / "index.faiss", "rb") as f:
            raw_bytes = f.read()
        raw = np.frombuffer(raw_bytes, dtype=np.uint8)
        self._index = faiss.deserialize_index(raw)
        # efSearch is a runtime HNSW param, not part of the serialized graph
        # -- re-apply this instance's own construction-time setting.
        self._index.hnsw.efSearch = self._ef_search
        with open(dir_path / "chunk_ids.json", encoding="utf-8") as f:
            self._chunk_ids = json.load(f)
