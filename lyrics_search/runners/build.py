"""SPEC-03 §5/§6/§7: `build` -- the full offline pipeline (ingest ->
preprocess -> embed -> index) for one ExperimentConfig, with stage-level
caching.

Each of the 4 stages is skipped (not rebuilt) when its on-disk artifact is
already "fresh" for the *current* stage_config -- see cache.py for the
exact freshness/atomicity contract this module relies on. `--force` (or
`force=True`) bypasses every freshness check and rebuilds all 4 stages
unconditionally, the only way to recover from a stage's own *code* having
changed without any config field changing (cache.py's docstring, and
SPEC-03 §6, are explicit that a config hash alone cannot detect that).

Two different artifact shapes need two different stage helpers:
  - flat-file stages (ingest -> raw.jsonl, preprocess -> chunks.jsonl):
    `_run_flat_stage()` -- freshness via is_flat_stage_fresh(), meta
    written as a sidecar *after* the data file (write_sidecar_meta()).
  - directory-artifact stages (embed, index): freshness via
    is_dir_stage_fresh(), built at a `.building_tmp` sibling then swapped
    into place in one atomic step via atomic_replace_dir() -- the stage's
    own meta.json travels inside that same swap, so it can never end up
    visible without the data it describes being complete too.

input_hash chaining (SPEC-03 §6): each stage's input_hash is the
*previous* stage's own cache_key, not a content-hash of its input file --
ingest's input_hash is the fixed ROOT_INPUT_HASH sentinel (no upstream
stage). This means re-running an earlier stage invalidates every stage
after it even when the later stage's own stage_config is unchanged.

Layout (EVAL-PREP §3): ingest is chunking-independent, so `raw.jsonl`
stays at the corpus root and both arms share it. Everything downstream of
chunking lives under `<corpus>/<chunking>/` -- chunks.jsonl, rejects.jsonl,
embeddings/, indexes/. The alternative, suffixing only the non-default
strategy's paths, needs no migration but leaves the default arm unlabelled
in the tree; for a project whose point is comparing arms, an unnamed
default is the worse failure. This also follows SPEC-04 §0.1's precedent
of keying by every dimension that varies, so one arm cannot evict another.
"""

from __future__ import annotations

import json
from pathlib import Path

from lyrics_search.cache import (
    ROOT_INPUT_HASH,
    atomic_replace_dir,
    compute_cache_key,
    is_dir_stage_fresh,
    is_flat_stage_fresh,
    write_sidecar_meta,
)
from lyrics_search.config import ExperimentConfig
from lyrics_search.paths import corpus_dir as _corpus_dir
from lyrics_search.paths import raw_path as _raw_path
from lyrics_search.paths import work_dir as _work_dir
from lyrics_search.runners.embed import run_embed
from lyrics_search.runners.ingest import run_ingest
from lyrics_search.runners.preprocess import MIN_CHUNK_CHARS, MIN_SONG_WORDS, run_preprocess
from lyrics_search.vector_store import load_matrix


def _ingest_stage_config(config: ExperimentConfig) -> dict:
    if config.corpus == "full":
        return {"corpus": "full"}
    if config.corpus == "demo":
        # SPEC-04 §3: the demo corpus is a fixed, fully-synthetic set (not a
        # sample of anything), always ingested via an explicit `source=`
        # (scripts/generate_demo_corpus.py), never HFDatasetSource.
        return {"corpus": "demo"}
    return {"corpus": "dev", "sample_size": 500, "seed": 42}


def _run_ingest_stage(config: ExperimentConfig, corpus_dir: Path, force: bool, source=None) -> str:
    raw_path = corpus_dir / "raw.jsonl"
    stage_config = _ingest_stage_config(config)
    cache_key = compute_cache_key(stage_config, ROOT_INPUT_HASH)

    if not force and is_flat_stage_fresh(raw_path, cache_key):
        print(f"[ingest] skip -- fresh (cache_key={cache_key[:12]})")
        return cache_key

    if source is None:
        from lyrics_search.sources.hf_dataset import HFDatasetSource

        source = HFDatasetSource(
            sample_size=stage_config.get("sample_size"),
            seed=stage_config.get("seed", 42),
        )
    run_ingest(source, corpus_dir)
    write_sidecar_meta(
        raw_path,
        cache_key=cache_key,
        input_hash=ROOT_INPUT_HASH,
        stage_config=stage_config,
    )
    return cache_key


def _run_preprocess_stage(
    raw_path: Path, work_dir: Path, chunking: str, ingest_cache_key: str, force: bool
) -> str:
    chunks_path = work_dir / "chunks.jsonl"
    # `chunking` is part of the stage config, not just of the path: two
    # arms with the same min_* thresholds are still different preprocess
    # runs, and the cache must say so.
    stage_config = {
        "chunking": chunking,
        "min_song_words": MIN_SONG_WORDS,
        "min_chunk_chars": MIN_CHUNK_CHARS,
    }
    cache_key = compute_cache_key(stage_config, ingest_cache_key)

    if not force and is_flat_stage_fresh(chunks_path, cache_key):
        print(f"[preprocess:{chunking}] skip -- fresh (cache_key={cache_key[:12]})")
        return cache_key

    run_preprocess(raw_path, work_dir, chunking)
    write_sidecar_meta(
        chunks_path,
        cache_key=cache_key,
        input_hash=ingest_cache_key,
        stage_config=stage_config,
    )
    return cache_key


def _run_embed_stage(
    config: ExperimentConfig, work_dir: Path, preprocess_cache_key: str, force: bool
) -> str:
    embedder_name = config.embedder.name
    chunks_path = work_dir / "chunks.jsonl"
    embeddings_dir = work_dir / "embeddings"
    final_dir = embeddings_dir / embedder_name
    # params flow into the cache key too: a params-only change (e.g. a
    # different vectors_path) must invalidate this stage same as a
    # different embedder name would.
    stage_config: dict = {"embedder": embedder_name, "params": config.embedder.params}
    cache_key = compute_cache_key(stage_config, preprocess_cache_key)

    if not force and is_dir_stage_fresh(final_dir, cache_key):
        print(f"[embed:{embedder_name}] skip -- fresh (cache_key={cache_key[:12]})")
        return cache_key

    from lyrics_search import registry

    embedder = registry.resolve_embedder(embedder_name, config.embedder.params)
    tmp_root = embeddings_dir / f"{embedder_name}.building_tmp"
    run_embed(
        embedder,
        chunks_path,
        tmp_root,
        cache_key=cache_key,
        input_hash=preprocess_cache_key,
        stage_config=stage_config,
    )
    # run_embed() writes into tmp_root/<embedder.name> (== final_dir's own
    # name) -- swap just that inner directory into final_dir's place, then
    # drop the now-empty tmp_root wrapper.
    atomic_replace_dir(tmp_root / embedder_name, final_dir)
    try:
        tmp_root.rmdir()
    except OSError:
        pass
    return cache_key


def _run_index_stage(
    config: ExperimentConfig, work_dir: Path, embed_cache_key: str, force: bool
) -> str:
    embedder_name = config.embedder.name
    embeddings_dir = work_dir / "embeddings" / embedder_name
    # SPEC-04 §0.1: keyed by *both* embedder and index type, not index type
    # alone -- the eval phase's whole point is several embedders on the same
    # corpus+index-type, and a shared slot meant every switch evicted the
    # previous embedder's index (see notes/reports/spec03-patch-report.md's
    # "shared index slot" observation, which this fixes).
    index_dir = work_dir / "indexes" / embedder_name / config.index
    stage_config: dict = {"index": config.index, "embedder": embedder_name}
    cache_key = compute_cache_key(stage_config, embed_cache_key)

    if not force and is_dir_stage_fresh(index_dir, cache_key):
        print(f"[index:{config.index}] skip -- fresh (cache_key={cache_key[:12]})")
        return cache_key

    from lyrics_search import registry

    index = registry.resolve_index(config.index)

    # EVAL-PREP §1: load_matrix() reads whichever representation the embed
    # stage wrote (vectors.npy dense / vectors.npz sparse) and hands the
    # index the matrix unchanged; the index decides whether it can take it.
    # A sparse matrix paired with `index: faiss-hnsw` fails here, loudly,
    # rather than being densified into a 51.8 GiB allocation.
    vectors = load_matrix(embeddings_dir)
    with open(embeddings_dir / "chunk_ids.json", encoding="utf-8") as f:
        chunk_ids = json.load(f)
    index.build(vectors, chunk_ids)

    tmp_root = work_dir / "indexes" / embedder_name / f"{config.index}.building_tmp"
    index.save(tmp_root)
    with open(tmp_root / "meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {"cache_key": cache_key, "input_hash": embed_cache_key, "stage_config": stage_config},
            f,
            ensure_ascii=False,
            indent=2,
        )
    atomic_replace_dir(tmp_root, index_dir)
    print(f"[index:{config.index}] built (cache_key={cache_key[:12]})")
    return cache_key


def run_build(
    config: ExperimentConfig,
    data_root: Path | str = "data",
    force: bool = False,
    source=None,
) -> Path:
    """Run ingest -> preprocess -> embed -> index for `config`, skipping any
    stage that's already fresh (unless `force`). Returns the chunking work
    dir (`<data_root>/<config.corpus>/<config.chunking>`) holding
    everything `search` needs except raw.jsonl.

    `source` overrides the ingest stage's default HFDatasetSource -- used
    by tests to ingest from a local JsonlFileSource fixture without a
    network/HF Hub call; production/CLI use leaves it None.
    """
    corpus_dir = _corpus_dir(data_root, config.corpus)
    work_dir = _work_dir(data_root, config.corpus, config.chunking)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    ingest_key = _run_ingest_stage(config, corpus_dir, force, source=source)
    preprocess_key = _run_preprocess_stage(
        _raw_path(data_root, config.corpus), work_dir, config.chunking, ingest_key, force
    )
    embed_key = _run_embed_stage(config, work_dir, preprocess_key, force)
    _run_index_stage(config, work_dir, embed_key, force)

    print(f"Build complete: {work_dir}")
    return work_dir


def main() -> None:
    import argparse

    from lyrics_search.config import load_config

    parser = argparse.ArgumentParser(description="SPEC-03 build runner")
    parser.add_argument("--config", required=True, help="path to an ExperimentConfig YAML")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--force", action="store_true", help="bypass every stage's cache")
    args = parser.parse_args()

    config = load_config(args.config)
    run_build(config, data_root=args.data_root, force=args.force)


if __name__ == "__main__":
    main()
