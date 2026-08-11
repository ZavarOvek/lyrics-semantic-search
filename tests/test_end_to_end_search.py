"""SPEC-03 §7 e2e: build (ingest->preprocess->embed->index) then search,
against a small local golden fixture via JsonlFileSource -- never touches
the network/HF Hub or the real data/dev corpus. Uses `tfidf`+`numpy` (fast,
CPU-only, no model download) as the embedder/index under test; the
retriever logic itself is already covered in isolation by
test_dense_retriever.py/test_lexical_retriever.py/test_hybrid_retriever.py,
so this file is about the *pipeline wiring* (build writes what search
reads, cache-skip on rebuild, --force bypasses it), not retrieval quality.
"""

from __future__ import annotations

from pathlib import Path

from lyrics_search.config import ExperimentConfig, RetrievalConfig
from lyrics_search.runners.build import run_build
from lyrics_search.runners.search import run_search
from lyrics_search.sources.jsonl_file import JsonlFileSource

GOLDEN = Path(__file__).parent / "golden" / "spec03_mini_corpus.jsonl"


def _config(mode: str = "hybrid") -> ExperimentConfig:
    return ExperimentConfig(
        corpus="dev",
        embedder="tfidf",
        index="numpy",
        retrieval=RetrievalConfig(mode=mode, top_k=10, return_n=5),
    )


def test_build_then_search_finds_matching_song(tmp_path):
    config = _config("hybrid")
    run_build(config, data_root=tmp_path, source=JsonlFileSource(GOLDEN))
    result, _timing = run_search(config, "sunshine morning", data_root=tmp_path)
    assert result.status == "ok"
    assert (
        result.hits[0].best_chunk.text.lower().startswith("sunshine sunshine")
        or "sunshine" in result.hits[0].best_chunk.text.lower()
    )


def test_build_then_search_dense_mode(tmp_path):
    config = _config("dense")
    run_build(config, data_root=tmp_path, source=JsonlFileSource(GOLDEN))
    result, _timing = run_search(config, "ocean waves", data_root=tmp_path)
    assert result.status == "ok"
    assert result.branch_statuses is None
    assert len(result.hits) > 0


def test_build_then_search_lexical_mode(tmp_path):
    config = _config("lexical")
    run_build(config, data_root=tmp_path, source=JsonlFileSource(GOLDEN))
    result, _timing = run_search(config, "mountain climbing summit", data_root=tmp_path)
    assert result.status == "ok"
    assert len(result.hits) > 0


def test_rebuild_skips_all_fresh_stages(tmp_path, capsys):
    config = _config("dense")
    run_build(config, data_root=tmp_path, source=JsonlFileSource(GOLDEN))
    capsys.readouterr()  # discard first build's output
    run_build(config, data_root=tmp_path, source=JsonlFileSource(GOLDEN))
    out = capsys.readouterr().out
    assert "[ingest] skip" in out
    assert "[preprocess:sections] skip" in out
    assert "[embed:tfidf] skip" in out
    assert "[index:numpy] skip" in out


def test_force_bypasses_cache_and_rebuilds(tmp_path, capsys):
    config = _config("dense")
    run_build(config, data_root=tmp_path, source=JsonlFileSource(GOLDEN))
    capsys.readouterr()
    run_build(config, data_root=tmp_path, force=True, source=JsonlFileSource(GOLDEN))
    out = capsys.readouterr().out
    assert "skip" not in out


def test_empty_query_end_to_end(tmp_path):
    config = _config("hybrid")
    run_build(config, data_root=tmp_path, source=JsonlFileSource(GOLDEN))
    result, _timing = run_search(config, "   ", data_root=tmp_path)
    assert result.status == "empty_query"
    assert result.hits == []


def test_return_n_trims_hits(tmp_path):
    config = ExperimentConfig(
        corpus="dev",
        embedder="tfidf",
        index="numpy",
        retrieval=RetrievalConfig(mode="dense", top_k=10, return_n=1),
    )
    run_build(config, data_root=tmp_path, source=JsonlFileSource(GOLDEN))
    result, _timing = run_search(config, "ocean waves sea", data_root=tmp_path)
    assert len(result.hits) <= 1


def test_search_repeat_reports_warm_timing(tmp_path):
    config = _config("dense")
    run_build(config, data_root=tmp_path, source=JsonlFileSource(GOLDEN))
    _result, timing = run_search(config, "rain thunder", data_root=tmp_path, repeat=3)
    assert timing["load_s"] >= 0
    assert timing["cold_query_s"] >= 0
    assert timing["warm_query_s_avg"] is not None
    assert timing["warm_query_s_avg"] >= 0


def test_build_then_search_demo_corpus(tmp_path):
    """SPEC-04 §3: `corpus: demo` (config.py's Literal extension) resolves
    and builds/searches through the exact same pipeline as `dev`/`full` --
    only the on-disk directory name differs (<data_root>/demo)."""
    config = ExperimentConfig(
        corpus="demo",
        embedder="tfidf",
        index="numpy",
        retrieval=RetrievalConfig(mode="hybrid", top_k=10, return_n=5),
    )
    work_dir = run_build(config, data_root=tmp_path, source=JsonlFileSource(GOLDEN))
    assert work_dir == tmp_path / "demo" / "sections"
    assert (tmp_path / "demo" / "raw.jsonl").exists()  # shared across chunking arms
    result, _timing = run_search(config, "mountain climbing summit", data_root=tmp_path)
    assert result.status == "ok"
    assert len(result.hits) > 0
