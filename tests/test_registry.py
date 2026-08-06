"""SPEC-03 §5 / SPEC-03-PATCH §1: registry tests -- name -> factory
resolution.

Deliberately does NOT instantiate bge-m3/e5-base/fasttext-avg (would
trigger a heavy model/vector-file load); only checks they are *registered*
by name (and, for fasttext-avg, that its required params are declared).
tfidf and the index factories are cheap enough to actually
resolve/instantiate.
"""
from __future__ import annotations

import pytest

from lyrics_search import registry


def test_tfidf_embedder_resolves():
    embedder = registry.resolve_embedder("tfidf")
    assert embedder.name == "tfidf"


def test_numpy_index_resolves():
    index = registry.resolve_index("numpy")
    assert index.name == "numpy"


def test_faiss_index_resolves():
    index = registry.resolve_index("faiss")
    assert index.name == "faiss-hnsw"


def test_bge_m3_and_e5_base_are_registered_without_instantiating():
    names = registry.registered_embedder_names()
    assert "bge-m3" in names
    assert "e5-base" in names


def test_fasttext_avg_is_registered():
    assert "fasttext-avg" in registry.registered_embedder_names()


def test_fasttext_avg_declares_vectors_path_required():
    assert registry.required_embedder_params("fasttext-avg") == ("vectors_path",)


def test_bge_m3_has_no_required_params():
    assert registry.required_embedder_params("bge-m3") == ()


def test_tfidf_resolves_with_params():
    embedder = registry.resolve_embedder("tfidf", {"max_features": 123})
    assert embedder._vectorizer.max_features == 123


def test_registered_embedder_names_sorted():
    names = registry.registered_embedder_names()
    assert names == sorted(names)


def test_registered_index_names_sorted():
    names = registry.registered_index_names()
    assert names == sorted(names)


def test_resolve_embedder_unknown_name_raises_key_error():
    with pytest.raises(KeyError):
        registry.resolve_embedder("does-not-exist")


def test_resolve_index_unknown_name_raises_key_error():
    with pytest.raises(KeyError):
        registry.resolve_index("does-not-exist")


def test_resolve_embedder_error_lists_registered_names():
    with pytest.raises(KeyError) as excinfo:
        registry.resolve_embedder("does-not-exist")
    assert "tfidf" in str(excinfo.value)
