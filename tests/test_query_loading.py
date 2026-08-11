"""SPEC-03 §1: query-time state loading tests -- must never fit/build,
must fail loudly (with exact missing-artifact paths) on any gap, must
verify fitted_state_sha1 and dim."""

from __future__ import annotations

import hashlib
import json
import re

import numpy as np
import pytest

from lyrics_search.contracts import Chunk, RawSong
from lyrics_search.embedders.tfidf import TfidfEmbedder
from lyrics_search.indexes.numpy_index import NumpyIndex
from lyrics_search.retrievers.loading import (
    load_chunk_lookup,
    load_fitted_embedder,
    load_index,
    load_song_corpus,
    warmup,
)


def _build_fitted_tfidf(embeddings_dir):
    """Simulate what runners/embed.py does for a tfidf embedder: fit,
    save_fit, and write a meta.json with fitted_state_file/sha1/dim."""
    out_dir = embeddings_dir / "tfidf"
    out_dir.mkdir(parents=True)
    embedder = TfidfEmbedder()
    embedder.fit(["walking alone in the rain", "thinking about you"])
    fitted_state_path = out_dir / "fitted_state.joblib"
    embedder.save_fit(fitted_state_path)
    sha1 = hashlib.sha1(fitted_state_path.read_bytes()).hexdigest()
    meta = {
        "embedder_name": "tfidf",
        "dim": embedder.dim,
        "fitted_state_file": "fitted_state.joblib",
        "fitted_state_sha1": sha1,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return embedder.dim


# --- load_fitted_embedder -----------------------------------------------


def test_load_fitted_embedder_round_trip(tmp_path):
    expected_dim = _build_fitted_tfidf(tmp_path)
    fresh = TfidfEmbedder()
    loaded = load_fitted_embedder(fresh, tmp_path)
    assert loaded.dim == expected_dim
    vec = loaded.encode(["walking alone"], is_query=True)
    assert vec.shape == (1, expected_dim)


def test_load_fitted_embedder_missing_meta_raises(tmp_path):
    fresh = TfidfEmbedder()
    with pytest.raises(FileNotFoundError, match=re.escape("meta.json")):
        load_fitted_embedder(fresh, tmp_path)


def test_load_fitted_embedder_missing_fitted_state_file_raises(tmp_path):
    _build_fitted_tfidf(tmp_path)
    (tmp_path / "tfidf" / "fitted_state.joblib").unlink()
    fresh = TfidfEmbedder()
    with pytest.raises(FileNotFoundError, match=re.escape("fitted_state.joblib")):
        load_fitted_embedder(fresh, tmp_path)


def test_load_fitted_embedder_hash_mismatch_raises(tmp_path):
    _build_fitted_tfidf(tmp_path)
    # corrupt the fitted-state file after its hash was recorded
    p = tmp_path / "tfidf" / "fitted_state.joblib"
    p.write_bytes(p.read_bytes() + b"corruption")
    fresh = TfidfEmbedder()
    with pytest.raises(ValueError, match="hash mismatch"):
        load_fitted_embedder(fresh, tmp_path)


def test_load_fitted_embedder_dim_mismatch_raises(tmp_path):
    _build_fitted_tfidf(tmp_path)
    meta_path = tmp_path / "tfidf" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["dim"] = meta["dim"] + 999
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    fresh = TfidfEmbedder()
    with pytest.raises(ValueError, match="dim mismatch"):
        load_fitted_embedder(fresh, tmp_path)


def test_load_fitted_embedder_no_fit_method_is_noop(tmp_path):
    """bge-m3/e5-base have no load_fit -- but embed.py still writes a
    meta.json for them (just with fitted_state_file=None), so loading
    still requires that meta.json to exist; it's only the load_fit()
    step itself that's skipped."""

    class FakeNeuralEmbedder:
        name = "fake-neural"
        dim = 4

        def encode(self, texts, *, is_query):
            return np.zeros((len(texts), self.dim), dtype=np.float32)

    out_dir = tmp_path / "fake-neural"
    out_dir.mkdir()
    meta = {"embedder_name": "fake-neural", "dim": 4, "fitted_state_file": None}
    (out_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    embedder = FakeNeuralEmbedder()
    result = load_fitted_embedder(embedder, tmp_path)
    assert result is embedder


# --- warmup ---------------------------------------------------------------


def test_warmup_calls_encode_with_is_query_true():
    calls = []

    class FakeEmbedder:
        def encode(self, texts, *, is_query):
            calls.append((tuple(texts), is_query))
            return np.zeros((len(texts), 4), dtype=np.float32)

    warmup(FakeEmbedder())
    assert calls == [(("warmup",), True)]


# --- load_chunk_lookup ------------------------------------------------------


def test_load_chunk_lookup_round_trip(tmp_path):
    chunks_path = tmp_path / "chunks.jsonl"
    chunk = Chunk(
        chunk_id="abc:0",
        song_id="abc",
        section=None,
        text="hello",
        position=0,
        split_by="none",
    )
    from dataclasses import asdict

    chunks_path.write_text(json.dumps(asdict(chunk)) + "\n", encoding="utf-8")
    lookup = load_chunk_lookup(chunks_path)
    assert lookup == {"abc:0": chunk}


def test_load_chunk_lookup_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_chunk_lookup(tmp_path / "nope.jsonl")


# --- load_song_corpus -------------------------------------------------------


def test_load_song_corpus_round_trip(tmp_path):
    raw_path = tmp_path / "raw.jsonl"
    song = RawSong(
        song_id="abc",
        artist="A",
        title="T",
        text_raw="lyrics here",
        source="test",
    )
    from dataclasses import asdict

    raw_path.write_text(json.dumps(asdict(song)) + "\n", encoding="utf-8")
    corpus = load_song_corpus(raw_path)
    assert corpus == {"abc": song}


def test_load_song_corpus_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_song_corpus(tmp_path / "nope.jsonl")


# --- load_index --------------------------------------------------------------


def test_load_index_round_trip(tmp_path):
    build_index = NumpyIndex()
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    build_index.build(vectors, ["c1", "c2"])
    index_dir = tmp_path / "numpy"
    build_index.save(index_dir)

    fresh = NumpyIndex()
    loaded = load_index(fresh, index_dir)
    assert loaded is fresh
    results = loaded.search(np.array([1.0, 0.0], dtype=np.float32), top_k=1)
    assert results[0][0] == "c1"


def test_load_index_missing_dir_raises(tmp_path):
    fresh = NumpyIndex()
    with pytest.raises(FileNotFoundError, match="index directory"):
        load_index(fresh, tmp_path / "does-not-exist")
