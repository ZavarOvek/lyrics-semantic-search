"""EVAL-THEMATIC §3: build the judgement pool from every arm's top-5.

    ./.venv/Scripts/python.exe scripts/thematic_pool.py \
        --themes data/eval/thematic/themes.jsonl \
        --out-dir data/eval/thematic

## Why the arm list is imported and not repeated here

`ARMS` comes from `scripts/eval_matrix.py`. The pool must be formed by
exactly the arms that later get measured: an arm absent from pooling has
its unique finds left unjudged, and under the "unjudged = non-relevant"
assumption those count as misses. That is §3's whole point, and a second
hand-maintained copy of the list is precisely how the two would drift
apart without anyone noticing.

The spec names 12 runs (4 embedders x 3 modes); nine are run. The four
nominal lexical cells return bit-identical rankings -- `build_retriever`
resolves no embedder in lexical mode and BM25 scores `text_raw` from
`raw.jsonl`, which every arm shares -- so pooling all four would add the
same five song_ids four times. The union is identical either way. This is
the same decision already recorded for the automatic track.

## Why provenance is written to a separate file

Three artifacts come out of this:

- `pool.jsonl` -- what the labelling interface reads. Theme, and the
  candidates already in the order they are to be shown. Nothing about
  which arm found a song or at what rank.
- `pool_provenance.jsonl` -- arm and rank per (theme, song). Read only by
  the metrics step, for §7's per-arm pool coverage.
- `runs.jsonl` -- each arm's full top-10 per theme. Metrics run at 10
  while the pool is built at 5, so the top-10 has to be kept; the gap
  between them is exactly what pool coverage and bpref exist to report.

§8 forbids showing the assessor the arm or the position. Keeping that data
in a file the interface never opens makes the leak impossible rather than
merely disallowed -- the interface has no field to render even if someone
later adds a line of display code.

## Display order

§5 follows TREC 2017: a warm-up in the natural order of results, then
blind. Warm-up themes order candidates by best rank across arms, so the
assessor calibrates on plausible material first. Every other theme is
shuffled by a generator seeded from the theme_id -- deterministic, so a
re-run shows the same order and a half-finished pass stays consistent,
but carrying no information about rank.

The ordering is resolved *here*, not in the interface, for the same reason
provenance is: the interface receives a list and cannot reconstruct what
produced it.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")  # run from the repo root, as every other script here does
sys.path.insert(0, "scripts")

from eval_matrix import ARMS  # the arm list, imported rather than repeated

from lyrics_search.config import load_config
from lyrics_search.runners.search import build_retriever

POOL_DEPTH = 5  # §2: top-5 from each run
RUN_DEPTH = 10  # §7: metrics are computed at 10

# §8 forbids showing the assessor the arm name or the position. That holds
# right now because the pool record below is a fixed dict literal -- but a
# later edit could widen it without anyone noticing, and the failure would be
# silent and invisible in the labelled data. So the field set is stated once
# and checked before the file is written.
POOL_FIELDS = frozenset({"theme_id", "text", "tier", "warmup", "candidates"})


def load_themes(path: Path | str, *, limit: int | None = None) -> list[dict]:
    """Themes from a JSONL file, skipping `_meta`, anything marked `keep:
    false`, and anything the second filtering pass left as `unsure`.

    `keep: null` alone is ambiguous -- it means both "never reviewed" and
    "reviewed twice and still undecided" (`thematic_filter.py`'s second pass
    leaves `mark: unsure` rather than forcing a verdict on a theme nobody
    could settle). Those must not resolve the same way here, so `mark`
    disambiguates:

      * no `mark` at all -> never reviewed -> kept. A half-finished filter
        must not silently shrink the pool; §9 filtering is the human's step,
        not this script's.
      * `mark: unsure` -> reviewed, left undecided -> excluded. By the time
        this runs against the real pool, §9 filtering is finished, and an
        explicit "not sure" is a verdict, not an omission -- pooling it
        would grade the system against a theme nobody signed off as
        answerable.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"missing themes file {path}")
    themes = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "_meta" in record:
                continue
            if record.get("keep") is False:
                continue
            if record.get("mark") == "unsure":
                continue
            for field in ("theme_id", "text", "tier"):
                if not record.get(field):
                    raise SystemExit(f"theme record missing {field!r}: {record}")
            themes.append(record)
    if not themes:
        raise SystemExit(f"{path} yielded no themes")
    if limit is not None:
        themes = themes[:limit]
    return themes


def display_order(
    theme_id: str, candidates: list[str], best_rank: dict[str, int], *, warmup: bool
) -> list[str]:
    """Order candidates for display. Warm-up: by best rank across arms.
    Otherwise: deterministically shuffled, seeded from the theme_id."""
    if warmup:
        return sorted(candidates, key=lambda s: (best_rank[s], s))
    shuffled = sorted(candidates)  # sort first so the shuffle input is order-independent
    random.Random(theme_id).shuffle(shuffled)
    return shuffled


def build_pool(
    themes_path: Path | str,
    out_dir: Path | str,
    *,
    data_root: str = "data",
    warmup_themes: int = 4,
    limit: int | None = None,
) -> None:
    themes = load_themes(themes_path, limit=limit)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # (theme_id, song_id) -> list of {arm, rank}
    provenance: dict[tuple[str, str], list[dict]] = {}
    runs: list[dict] = []

    for i, (arm, config_path) in enumerate(ARMS, 1):
        config = load_config(config_path)
        print(f"[{i}/{len(ARMS)}] {arm}: loading ...", flush=True)
        retriever, load_s = build_retriever(config, data_root=data_root)
        print(f"      loaded in {load_s:.1f}s, {len(themes)} themes", flush=True)
        t0 = time.time()
        for theme in themes:
            result = retriever.search(theme["text"], config.retrieval.top_k)
            ranked = [hit.song_id for hit in result.hits[:RUN_DEPTH]]
            runs.append(
                {
                    "arm": arm,
                    "theme_id": theme["theme_id"],
                    "status": result.status,
                    "ranked": ranked,
                }
            )
            for rank, song_id in enumerate(ranked[:POOL_DEPTH], start=1):
                provenance.setdefault((theme["theme_id"], song_id), []).append(
                    {"arm": arm, "rank": rank}
                )
        print(f"      searched in {time.time() - t0:.1f}s", flush=True)

    pool_records = []
    for n, theme in enumerate(themes):
        theme_id = theme["theme_id"]
        candidates = [s for (t, s) in provenance if t == theme_id]
        best_rank = {
            s: min(entry["rank"] for entry in provenance[(theme_id, s)]) for s in candidates
        }
        warmup = n < warmup_themes
        pool_records.append(
            {
                "theme_id": theme_id,
                "text": theme["text"],
                "tier": theme["tier"],
                "warmup": warmup,
                "candidates": display_order(theme_id, candidates, best_rank, warmup=warmup),
            }
        )

    for record in pool_records:
        extra = set(record) - POOL_FIELDS
        if extra:
            raise SystemExit(
                f"pool record for {record['theme_id']} carries {sorted(extra)}, which the "
                f"labelling interface must never see (EVAL-THEMATIC §8). Put it in "
                f"pool_provenance.jsonl instead."
            )

    write_jsonl(
        out_dir / "pool.jsonl",
        {
            "source": "EVAL-THEMATIC §3",
            "pool_depth": POOL_DEPTH,
            "arms_pooled": [a for a, _ in ARMS],
            "themes": len(themes),
            "warmup_themes": warmup_themes,
            "judgements": sum(len(r["candidates"]) for r in pool_records),
            "note": "candidates are already in display order; arm and rank are deliberately absent",
        },
        pool_records,
    )
    write_jsonl(
        out_dir / "pool_provenance.jsonl",
        {"source": "EVAL-THEMATIC §7", "note": "metrics only -- never read by the labelling UI"},
        [
            {"theme_id": t, "song_id": s, "found_by": entries}
            for (t, s), entries in sorted(provenance.items())
        ],
    )
    write_jsonl(
        out_dir / "runs.jsonl",
        {"source": "EVAL-THEMATIC §7", "run_depth": RUN_DEPTH, "arms": [a for a, _ in ARMS]},
        runs,
    )

    sizes = [len(r["candidates"]) for r in pool_records]
    empty = [r["theme_id"] for r in pool_records if not r["candidates"]]
    print(
        f"\npool: {len(themes)} themes, {sum(sizes)} judgements, "
        f"{min(sizes)}-{max(sizes)} per theme (mean {sum(sizes) / len(sizes):.1f})"
    )
    if empty:
        print(f"themes with an empty pool: {empty}")
    bad = [r for r in runs if r["status"] != "ok"]
    if bad:
        by_arm: dict[str, int] = {}
        for r in bad:
            by_arm[r["arm"]] = by_arm.get(r["arm"], 0) + 1
        print(f"non-ok searches (recorded, not dropped): {by_arm}")


def write_jsonl(path: Path, meta: dict, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"_meta": meta}, ensure_ascii=False) + "\n")
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {path} ({len(records)} records)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--themes", default="data/eval/thematic/themes.jsonl")
    parser.add_argument("--out-dir", default="data/eval/thematic")
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--warmup", type=int, default=4, help="themes shown in natural order first (§5)"
    )
    parser.add_argument("--limit", type=int, default=None, help="first N themes only, for smoke")
    args = parser.parse_args()
    build_pool(
        args.themes,
        args.out_dir,
        data_root=args.data_root,
        warmup_themes=args.warmup,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
