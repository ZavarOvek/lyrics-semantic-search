"""EVAL-PREP §1: tfidf's sparse return, the uncapped vocabulary, and the
dense/sparse helpers that make both safe for the rest of the pipeline.

The end-to-end path (embed -> vectors.npz -> index -> search) is already
exercised by test_end_to_end_search.py, which runs on tfidf+numpy. What is
checked here is the behaviour that file would still pass without: that the
matrix really is sparse rather than a dense array that happens to work,
that the cap is genuinely gone, and that every helper the pipeline routes
through gives the same answer for both representations.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from lyrics_search.core.vectors import (
    as_dense_vector,
    dot_scores,
    is_sparse,
    row_norms,
    select_rows,
    vector_norm,
)
from lyrics_search.embedders.tfidf import TfidfEmbedder
from lyrics_search.vector_store import load_matrix, save_matrix

CORPUS = [
    "sunshine morning light over the quiet valley",
    "ocean waves crashing on an empty shore",
    "mountain climbing to the summit before dawn",
    "walking alone in the rain thinking it over",
]


def _fitted() -> TfidfEmbedder:
    embedder = TfidfEmbedder()
    embedder.fit(CORPUS)
    return embedder


def test_tfidf_encode_returns_sparse_csr():
    matrix = _fitted().encode(CORPUS, is_query=False)
    assert sparse.issparse(matrix)
    assert matrix.format == "csr"
    assert matrix.dtype == np.float32
    assert matrix.shape == (len(CORPUS), _fitted().dim)


def test_tfidf_rows_stay_unit_norm():
    """Cosine similarity is a dot product only because rows are
    L2-normalised; that must survive the change of representation."""
    matrix = _fitted().encode(CORPUS, is_query=False)
    assert np.allclose(row_norms(matrix), 1.0, atol=1e-5)


def test_tfidf_vocabulary_is_uncapped_by_default():
    """EVAL-PREP §1 removed max_features. `dim` must therefore be the full
    vocabulary of the fitted corpus, not a truncation of it."""
    embedder = _fitted()
    distinct_terms = {w for line in CORPUS for w in line.split() if len(w) > 1}
    assert embedder.dim == len(distinct_terms)
    assert embedder._vectorizer.max_features is None


def test_tfidf_cap_still_available_when_asked_for():
    """The parameter stays, so an older index can be reproduced on
    purpose; nothing in the pipeline passes it."""
    embedder = TfidfEmbedder(max_features=5)
    embedder.fit(CORPUS)
    assert embedder.dim == 5


def test_tfidf_encode_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        TfidfEmbedder().encode(CORPUS, is_query=False)


def test_row_norms_agree_across_representations():
    dense = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    assert np.allclose(row_norms(sparse.csr_matrix(dense)), row_norms(dense))


def test_select_rows_agrees_across_representations():
    dense = np.arange(12, dtype=np.float32).reshape(4, 3)
    mask = np.array([True, False, True, False])
    picked = select_rows(sparse.csr_matrix(dense), mask)
    assert is_sparse(picked)
    assert np.array_equal(picked.toarray(), select_rows(dense, mask))


def test_dot_scores_agree_across_representations():
    dense = np.array([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]], dtype=np.float32)
    query = np.array([1.0, 0.0], dtype=np.float32)
    expected = dot_scores(dense, query)

    csr = sparse.csr_matrix(dense)
    for q in (query, csr[0]):
        scores = dot_scores(csr, q)
        # Always a plain dense 1-D array, whatever went in -- argsort in
        # NumpyIndex.search depends on that.
        assert isinstance(scores, np.ndarray) and scores.ndim == 1
        assert np.allclose(scores, expected)


def test_vector_norm_and_as_dense_vector_handle_a_sparse_row():
    row = sparse.csr_matrix(np.array([[3.0, 4.0, 0.0]], dtype=np.float32))[0]
    assert vector_norm(row) == pytest.approx(5.0)
    dense = as_dense_vector(row)
    assert dense.ndim == 1
    assert np.allclose(dense, [3.0, 4.0, 0.0])


def test_zero_row_norm_is_zero_for_a_sparse_matrix():
    """runners/embed.py drops no-signal chunks by comparing row norms to
    ZERO_NORM_EPS; an all-zero sparse row must report 0.0, not raise."""
    dense = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    assert row_norms(sparse.csr_matrix(dense))[1] == pytest.approx(0.0)


@pytest.mark.parametrize("make_sparse", [False, True])
def test_vector_store_roundtrip_preserves_representation(tmp_path, make_sparse):
    dense = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    matrix = sparse.csr_matrix(dense) if make_sparse else dense

    path = save_matrix(tmp_path, matrix)
    assert path.name == ("vectors.npz" if make_sparse else "vectors.npy")

    loaded = load_matrix(tmp_path)
    assert is_sparse(loaded) is make_sparse
    assert np.allclose(loaded.toarray() if make_sparse else loaded, dense)


def test_vector_store_names_the_missing_artifact(tmp_path):
    with pytest.raises(FileNotFoundError, match="vectors.np"):
        load_matrix(tmp_path)
