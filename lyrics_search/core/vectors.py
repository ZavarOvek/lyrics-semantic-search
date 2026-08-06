"""EVAL-PREP §1: dense/sparse polymorphism for embedding matrices.

`Embedder.encode()` may return either a dense `np.ndarray` or a
`scipy.sparse` matrix (see contracts.Embedder). Only `tfidf` returns
sparse today, but the choice is a property of the representation, not of
that one embedder: TF-IDF is ~0.14% dense on the full corpus, where a
dense float32 matrix would be 13.5 GiB against 39 MiB sparse.

Every operation the pipeline performs on an embedding matrix -- row
norms, row selection, dot-product scoring -- has a different call for the
two representations. Rather than scatter `if issparse(...)` through the
embed runner, the index and the retriever, the branch lives here once, in
pure functions with no I/O.

Cosine similarity itself is unchanged either way: vectors stay
L2-normalised, so scoring is a dot product in both representations.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse


def is_sparse(matrix: Any) -> bool:
    return sparse.issparse(matrix)


def row_norms(matrix: Any) -> np.ndarray:
    """L2 norm of every row, as a dense 1-D array in both representations."""
    if is_sparse(matrix):
        # sparse.linalg.norm(axis=1) allocates a dense result of length
        # n_rows only -- not the full matrix -- so this stays cheap.
        return sparse.linalg.norm(matrix, axis=1)
    return np.linalg.norm(matrix, axis=1)


def select_rows(matrix: Any, mask: np.ndarray) -> Any:
    """Boolean row selection, preserving the representation."""
    return matrix[mask]


def as_dense_vector(vector: Any) -> np.ndarray:
    """Collapse one encoded row to a dense 1-D float32 array.

    Queries are a single row, so densifying is bounded by the vocabulary
    size (~300 KiB at tfidf's full-corpus dim), not by corpus size. Doing
    it once here lets scoring stay a plain matrix @ vector product for
    both representations.
    """
    if is_sparse(vector):
        return np.asarray(vector.todense(), dtype=np.float32).ravel()
    return np.asarray(vector, dtype=np.float32).ravel()


def vector_norm(vector: Any) -> float:
    """L2 norm of a single encoded row, in either representation."""
    if is_sparse(vector):
        return float(sparse.linalg.norm(vector))
    return float(np.linalg.norm(vector))


def dot_scores(matrix: Any, query_vector: Any) -> np.ndarray:
    """Score every row of `matrix` against `query_vector`.

    Returns a dense 1-D array of length n_rows in both representations:
    the score vector is one float per chunk, which is small regardless of
    how the matrix itself is stored.
    """
    query = as_dense_vector(query_vector)
    scores = matrix @ query
    return np.asarray(scores, dtype=np.float32).ravel()
