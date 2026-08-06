"""EVAL-PREP §1: on-disk format for an embedding matrix, dense or sparse.

The I/O counterpart to `core/vectors.py` (which holds the pure dense/sparse
branch). `core/` stays free of I/O, so the file-format branch lives here.

The representation is recorded by the *file extension*, not by a flag in
meta.json: a dense matrix goes to `vectors.npy`, a sparse one to
`vectors.npz`. That way `load_matrix()` can pick the right reader by
looking at the directory, and an artifact built by an older run (always
`vectors.npy`) still loads without a meta.json migration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from lyrics_search.core.vectors import is_sparse

DENSE_NAME = "vectors.npy"
SPARSE_NAME = "vectors.npz"


def save_matrix(dir_path: Path | str, matrix: Any) -> Path:
    """Write `matrix` into `dir_path`, choosing the format from its type."""
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    if is_sparse(matrix):
        path = dir_path / SPARSE_NAME
        sparse.save_npz(path, matrix.tocsr())
    else:
        path = dir_path / DENSE_NAME
        np.save(path, matrix)
    return path


def load_matrix(dir_path: Path | str) -> Any:
    """Read whichever of the two files is present, preserving representation."""
    dir_path = Path(dir_path)
    sparse_path = dir_path / SPARSE_NAME
    if sparse_path.exists():
        return sparse.load_npz(sparse_path)
    dense_path = dir_path / DENSE_NAME
    if dense_path.exists():
        return np.load(dense_path)
    raise FileNotFoundError(
        f"no embedding matrix in {dir_path} -- expected {DENSE_NAME} or "
        f"{SPARSE_NAME}; run `build` before `search`."
    )
