"""SPEC-03 §6/§7: save()/load() round-trip tests for both index types.

Verifies a freshly-constructed index that only ever calls load() (never
build()) reproduces the same search results as the original built index --
the exact property runners/search.py depends on for §1's "query-time
branch never fits/builds anything, only loads persisted state" rule.

EVAL-PREP §1: the same round-trip is checked for a sparse matrix, which
travels as vectors.npz rather than vectors.npy, plus the two consequences
of NumpyIndex being the only index that takes sparse input: identical
scores from either representation of the same vectors, and FaissIndex
refusing sparse loudly instead of densifying.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from lyrics_search.indexes.numpy_index import NumpyIndex

VECTORS = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.7071, 0.7071, 0.0],
    ],
    dtype=np.float32,
)
SPARSE_VECTORS = sparse.csr_matrix(VECTORS)
CHUNK_IDS = ["c0", "c1", "c2", "c3"]
QUERY = np.array([1.0, 0.0, 0.0], dtype=np.float32)
SPARSE_QUERY = sparse.csr_matrix(QUERY.reshape(1, -1))[0]


def test_numpy_index_save_load_roundtrip(tmp_path):
    original = NumpyIndex()
    original.build(VECTORS, CHUNK_IDS)
    expected = original.search(QUERY, top_k=4)

    original.save(tmp_path)

    loaded = NumpyIndex()
    loaded.load(tmp_path)
    actual = loaded.search(QUERY, top_k=4)

    assert actual == expected


def test_numpy_index_sparse_save_load_roundtrip(tmp_path):
    """EVAL-PREP §1: a sparse index survives save()/load() as sparse, and
    the reloaded instance answers identically."""
    original = NumpyIndex()
    original.build(SPARSE_VECTORS, CHUNK_IDS)
    expected = original.search(QUERY, top_k=4)

    original.save(tmp_path)
    # The representation is recorded by the file name, not a meta flag --
    # so assert the sparse artifact is what actually landed on disk.
    assert (tmp_path / "vectors.npz").exists()
    assert not (tmp_path / "vectors.npy").exists()

    loaded = NumpyIndex()
    loaded.load(tmp_path)
    assert sparse.issparse(loaded._vectors)
    assert loaded.search(QUERY, top_k=4) == expected


def test_numpy_index_sparse_and_dense_score_identically():
    """The representation must not change the answer: the same vectors
    stored either way give the same chunk order and the same scores."""
    dense_index = NumpyIndex()
    dense_index.build(VECTORS, CHUNK_IDS)
    sparse_index = NumpyIndex()
    sparse_index.build(SPARSE_VECTORS, CHUNK_IDS)

    for query in (QUERY, SPARSE_QUERY):
        dense_hits = dense_index.search(query, top_k=4)
        sparse_hits = sparse_index.search(query, top_k=4)
        assert [cid for cid, _ in sparse_hits] == [cid for cid, _ in dense_hits]
        for (_, a), (_, b) in zip(sparse_hits, dense_hits):
            assert a == pytest.approx(b, abs=1e-6)


def test_numpy_index_search_before_build_or_load_raises():
    fresh = NumpyIndex()
    with pytest.raises(RuntimeError):
        fresh.search(QUERY, top_k=4)


def test_numpy_index_save_before_build_raises(tmp_path):
    fresh = NumpyIndex()
    with pytest.raises(RuntimeError):
        fresh.save(tmp_path)


def test_faiss_index_save_load_roundtrip(tmp_path):
    pytest.importorskip("faiss")
    from lyrics_search.indexes.faiss_index import FaissIndex

    original = FaissIndex(m=8, ef_construction=40, ef_search=16)
    original.build(VECTORS, CHUNK_IDS)
    expected = original.search(QUERY, top_k=4)

    original.save(tmp_path)

    # Fresh instance, same construction-time config (as a real caller would
    # build from the experiment config), never calls build() -- only load().
    loaded = FaissIndex(m=8, ef_construction=40, ef_search=16)
    loaded.load(tmp_path)
    actual = loaded.search(QUERY, top_k=4)

    assert [chunk_id for chunk_id, _ in actual] == [chunk_id for chunk_id, _ in expected]
    for (_, score_a), (_, score_b) in zip(actual, expected):
        assert score_a == pytest.approx(score_b, abs=1e-4)


def test_faiss_index_rejects_sparse_input():
    """EVAL-PREP §1: faiss takes dense float32 only. It must say so, not
    densify -- at full-corpus scale that is a 51.8 GiB allocation. The
    message has to name the alternative, per SPEC-00 §3.2 (fail loudly)."""
    pytest.importorskip("faiss")
    from lyrics_search.indexes.faiss_index import FaissIndex

    with pytest.raises(TypeError, match="numpy"):
        FaissIndex(m=8, ef_construction=40, ef_search=16).build(SPARSE_VECTORS, CHUNK_IDS)


def test_faiss_index_search_before_build_or_load_raises():
    pytest.importorskip("faiss")
    from lyrics_search.indexes.faiss_index import FaissIndex

    fresh = FaissIndex()
    with pytest.raises(RuntimeError):
        fresh.search(QUERY, top_k=4)
