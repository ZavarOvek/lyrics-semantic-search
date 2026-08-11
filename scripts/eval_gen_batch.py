"""EVAL-AUTO §2: print one batch of fragments for a generation worker.

The generation loop calls the `eval-query-generator` subagent once per
song. This script is how the caller gets its work: it prints the
fragments of one batch, and nothing else.

Only `chunk_id` and the fragment text are printed. The title and artist
are deliberately withheld -- see `eval_build_sample.py` for why -- and
the stratum, genre and structural fields are withheld simply because the
generator has no use for them and any of them could steer the wording.

Replies are collected into `raw/*.jsonl` files, one line per song. The
grouping is an artefact of how the replies get written and carries no
meaning: a chunk_id is "done" if it appears in *any* file in the
directory, so the batch boundaries used during generation need not match
the ones used to check it. `--check` reports which chunk_ids of a batch
are still missing, which is what makes an interrupted run resumable at
the level of a single song rather than a whole batch.

The dispatcher cannot be a subagent: subagents have no `Task` tool and
so cannot invoke the generator. Every call is issued from the main loop.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")  # run from the repo root, as every other script here does

RAW_DIR = Path("data/eval/auto/raw")


def done_ids(raw_dir: Path = RAW_DIR) -> set[str]:
    """Every chunk_id that already has a reply, from any file in `raw_dir`.

    Reading the id out of the record rather than off a filename keeps the
    grouping of replies into files a free choice: files can be merged,
    split or renamed without changing what counts as done.
    """
    seen: set[str] = set()
    for path in sorted(raw_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                seen.add(json.loads(line)["chunk_id"])
    return seen


def load_batch(sample_path: Path, batch: int, size: int) -> list[dict]:
    rows = [
        json.loads(line) for line in sample_path.read_text(encoding="utf-8").splitlines() if line
    ]
    usable = [r for r in rows if "skipped" not in r]
    start = batch * size
    if start >= len(usable):
        raise ValueError(f"batch {batch} is past the end ({len(usable)} usable songs)")
    return usable[start : start + size]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", default="data/eval/auto/sample.jsonl")
    parser.add_argument("--raw-dir", default=str(RAW_DIR))
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--size", type=int, default=25)
    parser.add_argument(
        "--check", action="store_true", help="report progress instead of printing work"
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows = load_batch(Path(args.sample), args.batch, args.size)
    done = done_ids(raw_dir)

    if args.check:
        have = [r for r in rows if r["chunk_id"] in done]
        print(f"batch {args.batch}: {len(have)}/{len(rows)} done")
        for row in rows:
            if row["chunk_id"] not in done:
                print(f"MISSING {row['chunk_id']}")
        return

    for row in rows:
        if row["chunk_id"] in done:
            continue
        print(f"### {row['chunk_id']}")
        print(row["text"])
        print()


if __name__ == "__main__":
    main()
