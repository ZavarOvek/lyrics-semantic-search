"""SPEC-03 §6: stage-cache tests -- key computation, chaining, freshness
checks, and atomic write helpers (including the directory-swap path,
whose necessity was discovered empirically: Windows os.replace() cannot
replace a non-empty directory)."""

from __future__ import annotations

import json
import os

import pytest

from lyrics_search.cache import (
    ROOT_INPUT_HASH,
    atomic_replace_dir,
    atomic_write_bytes,
    canonical_json,
    compute_cache_key,
    is_dir_stage_fresh,
    is_flat_stage_fresh,
    read_cache_key,
    sidecar_meta_path,
    write_sidecar_meta,
)


# --- canonical_json / compute_cache_key -------------------------------------


def test_canonical_json_is_key_order_independent():
    a = canonical_json({"x": 1, "y": 2})
    b = canonical_json({"y": 2, "x": 1})
    assert a == b


def test_compute_cache_key_deterministic():
    key1 = compute_cache_key({"a": 1}, "root")
    key2 = compute_cache_key({"a": 1}, "root")
    assert key1 == key2


def test_compute_cache_key_changes_with_config():
    key1 = compute_cache_key({"a": 1}, "root")
    key2 = compute_cache_key({"a": 2}, "root")
    assert key1 != key2


def test_compute_cache_key_chained_input_hash_changes_downstream_key():
    """The core §6 requirement: re-running an upstream stage invalidates a
    downstream stage's cache even if the downstream stage's own config is
    byte-identical, because input_hash is the upstream cache_key, not a
    content-hash of a file."""
    same_config = {"embedder": "bge-m3"}
    upstream_key_v1 = compute_cache_key({"seed": 42}, ROOT_INPUT_HASH)
    upstream_key_v2 = compute_cache_key({"seed": 43}, ROOT_INPUT_HASH)
    assert upstream_key_v1 != upstream_key_v2

    downstream_key_v1 = compute_cache_key(same_config, upstream_key_v1)
    downstream_key_v2 = compute_cache_key(same_config, upstream_key_v2)
    assert downstream_key_v1 != downstream_key_v2


def test_root_input_hash_is_stable_sentinel():
    assert ROOT_INPUT_HASH == "root"


# --- sidecar meta / flat-stage freshness ------------------------------------


def test_sidecar_meta_path_naming():
    assert sidecar_meta_path("data/dev/raw.jsonl").name == "raw.jsonl.meta.json"


def test_flat_stage_not_fresh_when_artifact_missing(tmp_path):
    artifact = tmp_path / "raw.jsonl"
    assert is_flat_stage_fresh(artifact, "somekey") is False


def test_flat_stage_not_fresh_when_meta_missing(tmp_path):
    artifact = tmp_path / "raw.jsonl"
    artifact.write_text("data")
    assert is_flat_stage_fresh(artifact, "somekey") is False


def test_flat_stage_fresh_after_write_sidecar_meta(tmp_path):
    artifact = tmp_path / "raw.jsonl"
    artifact.write_text("data")
    key = compute_cache_key({"seed": 1}, ROOT_INPUT_HASH)
    write_sidecar_meta(
        artifact, cache_key=key, input_hash=ROOT_INPUT_HASH, stage_config={"seed": 1}
    )
    assert is_flat_stage_fresh(artifact, key) is True


def test_flat_stage_not_fresh_on_key_mismatch(tmp_path):
    artifact = tmp_path / "raw.jsonl"
    artifact.write_text("data")
    key = compute_cache_key({"seed": 1}, ROOT_INPUT_HASH)
    write_sidecar_meta(
        artifact, cache_key=key, input_hash=ROOT_INPUT_HASH, stage_config={"seed": 1}
    )
    other_key = compute_cache_key({"seed": 2}, ROOT_INPUT_HASH)
    assert is_flat_stage_fresh(artifact, other_key) is False


def test_read_cache_key_missing_file_returns_none(tmp_path):
    assert read_cache_key(tmp_path / "nope.meta.json") is None


def test_read_cache_key_malformed_json_returns_none(tmp_path):
    p = tmp_path / "bad.meta.json"
    p.write_text("{not valid json")
    assert read_cache_key(p) is None


def test_write_sidecar_meta_round_trip_fields(tmp_path):
    artifact = tmp_path / "chunks.jsonl"
    artifact.write_text("data")
    key = compute_cache_key({"x": 1}, "upstream-key")
    write_sidecar_meta(artifact, cache_key=key, input_hash="upstream-key", stage_config={"x": 1})
    with open(sidecar_meta_path(artifact), encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["cache_key"] == key
    assert meta["input_hash"] == "upstream-key"
    assert meta["stage_config"] == {"x": 1}


# --- directory-stage freshness ----------------------------------------------


def test_dir_stage_not_fresh_when_dir_missing(tmp_path):
    assert is_dir_stage_fresh(tmp_path / "bge-m3", "somekey") is False


def test_dir_stage_fresh_after_meta_written(tmp_path):
    d = tmp_path / "bge-m3"
    d.mkdir()
    key = compute_cache_key({"embedder": "bge-m3"}, "upstream")
    meta = {"cache_key": key, "input_hash": "upstream", "stage_config": {"embedder": "bge-m3"}}
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    assert is_dir_stage_fresh(d, key) is True


def test_dir_stage_not_fresh_on_pre_spec03_meta_without_cache_key(tmp_path):
    """A meta.json written by the pre-SPEC-03 embed runner has no
    cache_key field at all -- must read as "not fresh", not crash."""
    d = tmp_path / "bge-m3"
    d.mkdir()
    (d / "meta.json").write_text(json.dumps({"embedder_name": "bge-m3"}), encoding="utf-8")
    assert is_dir_stage_fresh(d, "anykey") is False


# --- atomic_write_bytes ------------------------------------------------------


def test_atomic_write_bytes_writes_content(tmp_path):
    p = tmp_path / "out.bin"
    atomic_write_bytes(p, b"hello")
    assert p.read_bytes() == b"hello"


def test_atomic_write_bytes_leaves_no_tmp_file(tmp_path):
    p = tmp_path / "out.bin"
    atomic_write_bytes(p, b"hello")
    remaining = list(tmp_path.iterdir())
    assert remaining == [p]


def test_atomic_write_bytes_overwrites_existing(tmp_path):
    p = tmp_path / "out.bin"
    atomic_write_bytes(p, b"first")
    atomic_write_bytes(p, b"second")
    assert p.read_bytes() == b"second"


# --- atomic_replace_dir -------------------------------------------------------


def test_atomic_replace_dir_first_build_no_existing_target(tmp_path):
    final_dir = tmp_path / "target"
    tmp_dir = tmp_path / "target.tmp"
    tmp_dir.mkdir()
    (tmp_dir / "f.txt").write_text("first")

    atomic_replace_dir(tmp_dir, final_dir)

    assert final_dir.exists()
    assert (final_dir / "f.txt").read_text() == "first"
    assert not tmp_dir.exists()


def test_atomic_replace_dir_swaps_existing_target(tmp_path):
    final_dir = tmp_path / "target"
    final_dir.mkdir()
    (final_dir / "f.txt").write_text("old")

    tmp_dir = tmp_path / "target.tmp"
    tmp_dir.mkdir()
    (tmp_dir / "f.txt").write_text("new")

    atomic_replace_dir(tmp_dir, final_dir)

    assert (final_dir / "f.txt").read_text() == "new"
    assert not tmp_dir.exists()
    # no stray .old-* directory left behind
    leftovers = [p for p in tmp_path.iterdir() if p.name != "target"]
    assert leftovers == []


def test_atomic_replace_dir_result_readable_as_dir_stage(tmp_path):
    """End-to-end: build in a temp dir with a meta.json inside, swap into
    place, then confirm is_dir_stage_fresh() reads it correctly -- this is
    the exact sequence runners/build.py's embed/index stages will use."""
    final_dir = tmp_path / "bge-m3"
    tmp_dir = tmp_path / "bge-m3.tmp"
    tmp_dir.mkdir()
    key = compute_cache_key({"embedder": "bge-m3"}, "upstream-key")
    meta = {"cache_key": key, "input_hash": "upstream-key", "stage_config": {"embedder": "bge-m3"}}
    (tmp_dir / "vectors.npy").write_bytes(b"fake-vectors")
    (tmp_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    atomic_replace_dir(tmp_dir, final_dir)

    assert is_dir_stage_fresh(final_dir, key) is True
    assert (final_dir / "vectors.npy").read_bytes() == b"fake-vectors"
