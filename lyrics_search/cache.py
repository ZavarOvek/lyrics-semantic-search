"""SPEC-03 §6: stage-level caching for the offline pipeline.

Every stage (ingest, preprocess, embed, index) is identified by a

    cache_key = sha1(canonical_json(stage_config) + "|" + input_hash)

`input_hash` is deliberately *not* a content-hash of the stage's actual
input file(s) -- hashing `vectors.npy` at full-corpus scale would itself
cost real time, defeating the point of caching. It is instead the
*previous* stage's own `cache_key`, chained: ingest has no upstream stage,
so its `input_hash` is the fixed sentinel `ROOT_INPUT_HASH`; preprocess's
`input_hash` is ingest's `cache_key`; embed's is preprocess's `cache_key`;
index's is embed's `cache_key`. This means re-running an upstream stage
automatically invalidates every downstream stage's cache even when the
downstream stage's own config is byte-identical -- a stale `chunks.jsonl`
can never look "fresh" to `embed` just because `embed`'s own config didn't
change.

Every stage records its own `cache_key`/`input_hash`/`stage_config` in a
sidecar meta file next to its artifact(s):
  - flat-file stages (ingest -> raw.jsonl, preprocess -> chunks.jsonl):
    `<artifact>.meta.json` (e.g. `raw.jsonl.meta.json`).
  - directory-artifact stages (embed, index): the fields are merged into
    the `meta.json` that already lives inside the stage's own output
    directory.

Atomicity: an interrupted run must never be mistaken for a valid,
complete cache entry.
  - flat-file stages: the data file is written via `atomic_write_bytes()`
    (temp path + `os.replace()`) and the sidecar meta file is written
    (also atomically) strictly *after* the data file is already in place
    -- so if the meta file exists at all, the data it describes is
    guaranteed complete.
  - directory-artifact stages: the whole output directory (data files +
    meta.json together) is built at a temp path, then `atomic_replace_dir()`
    swaps it into place in one logical step -- meta.json necessarily
    becomes visible atomically together with the data it describes, a
    strictly stronger guarantee than "meta written last".

`--force` (handled by callers, not this module) bypasses the freshness
check entirely so a stage always reruns -- needed for the one case a
config hash structurally cannot detect: the stage's own *code* changed
without any config field changing.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

ROOT_INPUT_HASH = "root"  # sentinel: ingest has no upstream stage to chain from


def canonical_json(obj: Any) -> str:
    """Deterministic JSON serialization for hashing: sorted keys, no
    whitespace variance, so semantically-identical configs (regardless of
    dict insertion order) always hash identically."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def compute_cache_key(stage_config: dict, input_hash: str) -> str:
    """cache_key = sha1(canonical_json(stage_config) + "|" + input_hash)."""
    payload = canonical_json(stage_config) + "|" + input_hash
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def sidecar_meta_path(artifact_path: Path | str) -> Path:
    artifact_path = Path(artifact_path)
    return artifact_path.with_name(artifact_path.name + ".meta.json")


def read_cache_key(meta_path: Path | str) -> str | None:
    """Read `cache_key` from a sidecar or directory meta.json file.

    Returns None if the file doesn't exist, isn't valid JSON, or doesn't
    carry a cache_key field (e.g. a pre-SPEC-03 embed meta.json written
    before stage-caching existed) -- all of which mean "not fresh, must
    rebuild", never an exception.
    """
    meta_path = Path(meta_path)
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    return data.get("cache_key")


def is_flat_stage_fresh(artifact_path: Path | str, expected_cache_key: str) -> bool:
    """Freshness check for a flat-file stage (ingest -> raw.jsonl,
    preprocess -> chunks.jsonl). Fresh iff the artifact file exists AND its
    sidecar meta file exists AND its recorded cache_key matches."""
    artifact_path = Path(artifact_path)
    if not artifact_path.exists():
        return False
    return read_cache_key(sidecar_meta_path(artifact_path)) == expected_cache_key


def is_dir_stage_fresh(dir_path: Path | str, expected_cache_key: str) -> bool:
    """Freshness check for a directory-artifact stage (embed, index). Fresh
    iff the directory exists AND its meta.json exists AND its recorded
    cache_key matches."""
    dir_path = Path(dir_path)
    if not dir_path.exists():
        return False
    return read_cache_key(dir_path / "meta.json") == expected_cache_key


def write_sidecar_meta(
    artifact_path: Path | str,
    *,
    cache_key: str,
    input_hash: str,
    stage_config: dict,
    extra: dict | None = None,
) -> None:
    """Atomically write `<artifact>.meta.json`. Caller must have already
    written the artifact's own data file (e.g. via atomic_write_bytes())
    before calling this -- meta existing is the freshness signal that the
    data is complete, so it must genuinely come last."""
    meta: dict[str, Any] = {
        "cache_key": cache_key,
        "input_hash": input_hash,
        "stage_config": stage_config,
    }
    if extra:
        meta.update(extra)
    atomic_write_bytes(
        sidecar_meta_path(artifact_path),
        json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"),
    )


def atomic_write_bytes(path: Path | str, data: bytes) -> None:
    """Write bytes via temp-file-then-os.replace(), so a concurrent reader
    (or a crash mid-write) never observes a partially-written file."""
    path = Path(path)
    tmp_path = path.with_name(path.name + f".tmp-{uuid.uuid4().hex[:8]}")
    with open(tmp_path, "wb") as f:
        f.write(data)
    os.replace(tmp_path, path)


def atomic_replace_dir(tmp_dir: Path | str, final_dir: Path | str) -> None:
    """Atomically (best-effort) replace final_dir's contents with tmp_dir's.

    Windows' os.replace() cannot directly replace a non-empty directory
    (verified empirically during SPEC-03: raises PermissionError), unlike
    POSIX rename(2). Uses a swap instead: rename any existing final_dir out
    of the way first, rename tmp_dir into final_dir's place, then delete
    the old one. There is a narrow window between the two renames where
    final_dir doesn't exist on disk; a process interrupted in that window
    leaves final_dir looking identical to "never built" to
    is_dir_stage_fresh() -- the safe direction to fail in, since it forces
    a rebuild rather than risking a reader seeing a half-swapped directory.
    """
    tmp_dir = Path(tmp_dir)
    final_dir = Path(final_dir)
    old_dir = None
    if final_dir.exists():
        old_dir = final_dir.with_name(final_dir.name + f".old-{uuid.uuid4().hex[:8]}")
        os.replace(final_dir, old_dir)
    os.replace(tmp_dir, final_dir)
    if old_dir is not None:
        shutil.rmtree(old_dir, ignore_errors=True)
