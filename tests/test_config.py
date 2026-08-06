"""SPEC-03 §5: config tests -- YAML -> validated ExperimentConfig."""
from __future__ import annotations

import pytest

from lyrics_search.config import ExperimentConfig, RetrievalConfig, load_config


def test_retrieval_config_defaults():
    r = RetrievalConfig()
    assert r.mode == "dense"
    assert r.top_k == 50
    assert r.return_n == 10


def test_experiment_config_defaults():
    c = ExperimentConfig()
    assert c.corpus == "dev"
    assert c.embedder.name == "bge-m3"
    assert c.embedder.params == {}
    assert c.index == "numpy"
    assert c.retrieval.mode == "dense"


def test_experiment_config_rejects_unknown_field():
    with pytest.raises(Exception):
        ExperimentConfig(bogus_field=1)


def test_experiment_config_rejects_invalid_mode():
    with pytest.raises(Exception):
        ExperimentConfig(retrieval={"mode": "not-a-real-mode"})


def test_load_config_valid_yaml(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "corpus: dev\nembedder: tfidf\nindex: numpy\nretrieval:\n  mode: dense\n",
        encoding="utf-8",
    )
    config = load_config(p)
    assert config.embedder.name == "tfidf"
    assert config.index == "numpy"


def test_load_config_unknown_embedder_raises(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("embedder: not-a-real-embedder\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown embedder"):
        load_config(p)


def test_load_config_unknown_index_raises(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("index: not-a-real-index\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown index"):
        load_config(p)


def test_load_config_empty_file_uses_defaults(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("", encoding="utf-8")
    config = load_config(p)
    assert config.embedder.name == "bge-m3"


def test_load_config_fasttext_avg_without_vectors_path_raises(tmp_path):
    """SPEC-03-PATCH §1: fasttext-avg IS registered, but its required
    `vectors_path` param must be present in the config -- a config that
    omits it must fail loudly at load time, not deep inside a build/search
    run (and never falls back to some other embedder)."""
    p = tmp_path / "cfg.yaml"
    p.write_text("embedder: fasttext-avg\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required params"):
        load_config(p)


def test_load_config_fasttext_avg_with_vectors_path_loads(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "embedder:\n  name: fasttext-avg\n  params:\n    vectors_path: some/path.vec\n",
        encoding="utf-8",
    )
    config = load_config(p)
    assert config.embedder.name == "fasttext-avg"
    assert config.embedder.params == {"vectors_path": "some/path.vec"}


def test_load_config_embedder_plain_string_shorthand_still_works(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("embedder: bge-m3\n", encoding="utf-8")
    config = load_config(p)
    assert config.embedder.name == "bge-m3"
    assert config.embedder.params == {}


def test_experiment_config_accepts_demo_corpus():
    """SPEC-04 §3: `demo` is a valid corpus alongside `dev`/`full`."""
    c = ExperimentConfig(corpus="demo")
    assert c.corpus == "demo"


def test_experiment_config_rejects_unknown_corpus():
    with pytest.raises(Exception):
        ExperimentConfig(corpus="not-a-real-corpus")


def test_load_config_demo_yaml():
    """configs/demo.yaml itself must load cleanly with the settings the
    demo pipeline (scripts/generate_demo_corpus.py) actually builds
    against -- catches the config and the script drifting apart."""
    config = load_config("configs/demo.yaml")
    assert config.corpus == "demo"
    assert config.embedder.name == "tfidf"
    assert config.index == "numpy"
    assert config.retrieval.mode == "hybrid"
