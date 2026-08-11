"""SPEC-03 §7: `search` -- the online-only CLI entrypoint.

Strictly query-time: every piece of state (embedder fit, index, chunk/song
lookups) is *loaded* via retrievers/loading.py, never fit or built here
(SPEC-03 §1). `--repeat N` re-runs the same query N times after the initial
load+warmup, printing cold (load+first query) vs warm (subsequent queries
only) latency -- SPEC-03 §7's <100ms warm target is measured this way.
"""

from __future__ import annotations

import time
from pathlib import Path

from lyrics_search.config import ExperimentConfig
from lyrics_search.contracts import SearchResult
from lyrics_search.paths import raw_path as _raw_path
from lyrics_search.paths import work_dir as _work_dir
from lyrics_search.retrievers.dense import DenseRetriever
from lyrics_search.retrievers.hybrid import HybridRetriever
from lyrics_search.retrievers.lexical import LexicalRetriever
from lyrics_search.retrievers.loading import (
    load_chunk_lookup,
    load_fitted_embedder,
    load_index,
    load_song_corpus,
    warmup,
)


def build_retriever(config: ExperimentConfig, data_root: Path | str = "data"):
    """Load all on-disk state named by `config` and construct the
    retriever for `config.retrieval.mode`. Returns (retriever, load_s)."""
    from lyrics_search import registry

    t0 = time.time()
    # raw.jsonl is shared across chunking arms; everything else is per-arm
    # (see paths.py).
    work_dir = _work_dir(data_root, config.corpus, config.chunking)
    chunks_path = work_dir / "chunks.jsonl"
    raw_path = _raw_path(data_root, config.corpus)
    embeddings_dir = work_dir / "embeddings"
    index_dir = work_dir / "indexes" / config.embedder.name / config.index

    dense_retriever = None
    lexical_retriever = None

    if config.retrieval.mode in ("dense", "hybrid"):
        embedder = registry.resolve_embedder(config.embedder.name, config.embedder.params)
        load_fitted_embedder(embedder, embeddings_dir)
        warmup(embedder)
        index = registry.resolve_index(config.index)
        load_index(index, index_dir)
        chunk_lookup = load_chunk_lookup(chunks_path)
        dense_retriever = DenseRetriever(embedder, index, chunk_lookup)

    if config.retrieval.mode in ("lexical", "hybrid"):
        song_corpus = load_song_corpus(raw_path)
        chunk_lookup = load_chunk_lookup(chunks_path)
        lexical_retriever = LexicalRetriever(song_corpus, chunk_lookup)

    if config.retrieval.mode == "dense":
        retriever = dense_retriever
    elif config.retrieval.mode == "lexical":
        retriever = lexical_retriever
    else:
        retriever = HybridRetriever(
            dense_retriever, lexical_retriever, rrf_k=config.retrieval.rrf_k
        )

    load_s = time.time() - t0
    return retriever, load_s


def run_search(
    config: ExperimentConfig,
    query: str,
    data_root: Path | str = "data",
    repeat: int = 1,
) -> tuple[SearchResult, dict]:
    """Load state, run `query` once (cold) then `repeat - 1` more times
    (warm), trim to config.retrieval.return_n. Returns (trimmed_result,
    timing_dict) where timing_dict has "load_s", "cold_query_s",
    "warm_query_s_avg" (None if repeat < 2)."""
    retriever, load_s = build_retriever(config, data_root=data_root)

    t0 = time.time()
    result = retriever.search(query, config.retrieval.top_k)
    cold_query_s = time.time() - t0

    warm_query_s_avg = None
    if repeat > 1:
        t0 = time.time()
        for _ in range(repeat - 1):
            retriever.search(query, config.retrieval.top_k)
        warm_query_s_avg = (time.time() - t0) / (repeat - 1)

    trimmed = SearchResult(
        hits=result.hits[: config.retrieval.return_n],
        status=result.status,
        query_norm=result.query_norm,
        branch_statuses=result.branch_statuses,
    )
    timing = {"load_s": load_s, "cold_query_s": cold_query_s, "warm_query_s_avg": warm_query_s_avg}
    return trimmed, timing


def _print_result(result: SearchResult, timing: dict) -> None:
    print(f"status={result.status} query_norm={result.query_norm:.4f}")
    if result.branch_statuses:
        print(f"branch_statuses={result.branch_statuses}")
    for rank, hit in enumerate(result.hits, start=1):
        preview = hit.best_chunk.text[:80].replace("\n", " ")
        print(f'  {rank}. {hit.song_id}  score={hit.score:.4f}  "{preview}"')
    print(
        f"load={timing['load_s'] * 1000:.1f}ms cold_query={timing['cold_query_s'] * 1000:.1f}ms"
        + (
            f" warm_query_avg={timing['warm_query_s_avg'] * 1000:.1f}ms"
            if timing["warm_query_s_avg"] is not None
            else ""
        )
    )


def main() -> None:
    import argparse

    from lyrics_search.config import load_config

    parser = argparse.ArgumentParser(description="SPEC-03 search runner")
    parser.add_argument("--config", required=True, help="path to an ExperimentConfig YAML")
    parser.add_argument("query", help="search query text")
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--repeat", type=int, default=1, help="repeat query N times to measure warm latency"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    result, timing = run_search(config, args.query, data_root=args.data_root, repeat=args.repeat)
    _print_result(result, timing)


if __name__ == "__main__":
    main()
