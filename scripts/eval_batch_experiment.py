"""Does batching several fragments into one generator call damage the queries?

Mass generation needs ~500 generator calls. Subagents cannot invoke
subagents (`notes/decisions/subagents-cannot-nest.md`), so every call is
issued from the main loop and the only lever on cost is how many
fragments one call covers. Batching is therefore worth having -- but it
introduces a failure that single calls cannot produce at all, and that
the normal gate cannot see.

## The failure batching introduces

The generator sees several fragments at once and may carry vocabulary,
imagery or subject matter from one into the queries written for another.
`eval_gate_queries.py` scores each query against *its own* fragment, so a
query for fragment 4 that borrowed a distinctive word from fragment 3
passes the gate cleanly. It still corrupts the benchmark: that word is
indexed with song 3, so the query for song 4 now retrieves song 3.

This script scores every query against the *other* fragments in its
group, which is the only way that failure becomes visible.

## Why the baseline is not zero

The tempting comparison is "cross-fragment overlap should be zero,
because in single-call mode the model never saw the other fragments".
That is wrong, and using it would condemn batching on a number that
means nothing.

Two unrelated pieces of English share rare-ish terms by chance. A
paraphrase about a woman leaving at night and an unrelated fragment about
a train at night can share a moderately uncommon word with no causal
connection whatsoever. The chance rate is small but not zero, and it
depends on the vocabulary of the particular songs drawn.

So the control -- the same 57 songs generated one call at a time -- is
scored with the *same grouping applied after the fact*. The control's
cross-fragment number is the chance rate for exactly these songs and
exactly these groups. Contamination is the excess of the batched arm over
that, not the raw value.

## What is reported

Per arm, per query type:

1. **cross-fragment leak** -- the metric above, against the chance
   baseline. This one can kill batching on its own.
2. **own-fragment overlap** -- `ratio` and `max_shared_idf`, which must
   match the control within noise, otherwise batching changed how
   closely the model tracks the source.
3. **gate failure rate** -- against the control's rate.
4. **positional effect** -- failures and leak by index within the group.
   Models often treat the first and last item of a list differently, and
   if failures concentrate by position the group is too large.
5. **output length and pair Jaccard** -- batching tends to shorten
   replies and to pull a song's two queries toward each other, both of
   which reduce what the query set can distinguish.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, ".")  # run from the repo root, as every other script here does

import joblib

from lyrics_search.core.lexical_leak import idf_weighted_overlap
from lyrics_search.paths import work_dir

QUERY_TYPES = ("paraphrase", "description")


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _idf_from_df(df: float, n_docs: int) -> float:
    return math.log((1 + n_docs) / (1 + df)) + 1


def _df_from_idf(idf: float, n_docs: int) -> float:
    return (1 + n_docs) / math.exp(idf - 1) - 1


def load_arm(raw_dir: Path) -> dict[str, dict]:
    """First reply per chunk_id, so a rerun cannot double-count a song."""
    replies: dict[str, dict] = {}
    for path in sorted(raw_dir.glob("*.jsonl")):
        for record in _read_jsonl(path):
            replies.setdefault(record["chunk_id"], record)
    return replies


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--corpus", default="full")
    parser.add_argument("--chunking", default="sections")
    parser.add_argument("--sample", default="data/eval/auto/sample.jsonl")
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="NAME=DIR",
        help="an arm to score, e.g. control=data/eval/auto/raw",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        required=True,
        help="fragments per generator call; applied to every arm, including "
        "the control, so the chance baseline is computed on the same groups",
    )
    parser.add_argument("--gate-df", type=int, default=100)
    parser.add_argument("--out", default="data/eval/auto/batch_experiment.jsonl")
    args = parser.parse_args()

    root = Path(args.data_root)
    state_path = (
        work_dir(root, args.corpus, args.chunking) / "embeddings" / "tfidf" / "fitted_state.joblib"
    )
    vectorizer = joblib.load(state_path)
    analyze = vectorizer.build_analyzer()
    idf_values = vectorizer.idf_
    idf = {term: float(idf_values[i]) for term, i in vectorizer.vocabulary_.items()}
    n_docs = round(math.exp(max(idf.values()) - 1) * 2 - 1)
    gate_idf = _idf_from_df(args.gate_df, n_docs)

    fragments = {r["chunk_id"]: r for r in _read_jsonl(Path(args.sample)) if "chunk_id" in r}

    arms = {}
    for spec in args.arm:
        name, _, directory = spec.partition("=")
        arms[name] = load_arm(Path(directory))

    # Every arm is scored on the songs all arms have, so the numbers are
    # paired. Comparing an arm on 57 songs against one on 40 would confound
    # the batching effect with which songs happened to be included.
    common = sorted(set.intersection(*(set(a) for a in arms.values())))
    if not common:
        raise SystemExit("the arms share no chunk_ids")

    groups = [common[i : i + args.group_size] for i in range(0, len(common), args.group_size)]
    group_of = {cid: gi for gi, group in enumerate(groups) for cid in group}
    position_of = {cid: i for group in groups for i, cid in enumerate(group)}

    terms_of = {cid: analyze(fragments[cid]["text"]) for cid in common}

    rows = []
    for arm_name, replies in arms.items():
        for cid in common:
            group = groups[group_of[cid]]
            others = [o for o in group if o != cid]
            reply = replies[cid]
            for field in QUERY_TYPES:
                text = reply[field]
                query_terms = analyze(text)
                own = idf_weighted_overlap(query_terms, terms_of[cid], idf)
                cross = 0.0
                cross_terms: list[str] = []
                for other in others:
                    score = idf_weighted_overlap(query_terms, terms_of[other], idf)
                    if score.max_shared_idf > cross:
                        cross = score.max_shared_idf
                        cross_terms = [
                            t for t in score.shared if idf.get(t, 0.0) >= score.max_shared_idf
                        ]
                rows.append(
                    {
                        "arm": arm_name,
                        "chunk_id": cid,
                        "query_type": field,
                        "group": group_of[cid],
                        "position": position_of[cid],
                        "n_words": len(text.split()),
                        "ratio": own.ratio,
                        "own_max_idf": own.max_shared_idf,
                        "cross_max_idf": cross,
                        "cross_terms": cross_terms,
                        "passes": own.max_shared_idf < gate_idf,
                        "text": text,
                    }
                )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_arm: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_arm[row["arm"]].append(row)

    print(f"songs compared : {len(common)}  (paired across {len(arms)} arms)")
    print(f"group size     : {args.group_size}  -> {len(groups)} groups")
    print(f"gate           : shared term in <= {args.gate_df} chunks (idf >= {gate_idf:.3f})")

    print("\n=== 1. cross-fragment leak (the number batching can fail on) ===")
    print("the control's value is the chance rate for these songs, not zero\n")
    print(f"{'arm':<10} {'type':<12} {'mean':>7} {'max':>7} {'df@max':>8} {'>=gate':>7}")
    for arm_name, arm_rows in by_arm.items():
        for field in QUERY_TYPES:
            sel = [r for r in arm_rows if r["query_type"] == field]
            worst = max(r["cross_max_idf"] for r in sel)
            over = sum(1 for r in sel if r["cross_max_idf"] >= gate_idf)
            print(
                f"{arm_name:<10} {field:<12} "
                f"{statistics.mean(r['cross_max_idf'] for r in sel):>7.3f} "
                f"{worst:>7.3f} {_df_from_idf(worst, n_docs):>8.0f} {over:>7}"
            )

    print("\n=== 2. own-fragment overlap (must match the control) ===")
    print(f"{'arm':<10} {'type':<12} {'ratio':>7} {'own_idf':>9} {'worst':>7}")
    for arm_name, arm_rows in by_arm.items():
        for field in QUERY_TYPES:
            sel = [r for r in arm_rows if r["query_type"] == field]
            print(
                f"{arm_name:<10} {field:<12} "
                f"{statistics.mean(r['ratio'] for r in sel):>7.3f} "
                f"{statistics.mean(r['own_max_idf'] for r in sel):>9.3f} "
                f"{max(r['own_max_idf'] for r in sel):>7.3f}"
            )

    print("\n=== 3. gate failure rate ===")
    for arm_name, arm_rows in by_arm.items():
        for field in QUERY_TYPES:
            sel = [r for r in arm_rows if r["query_type"] == field]
            fails = [r for r in sel if not r["passes"]]
            print(f"{arm_name:<10} {field:<12} {len(fails):>3} / {len(sel)}")

    print("\n=== 4. positional effect within the group ===")
    print(f"{'arm':<10} {'pos':>4} {'n':>4} {'fails':>6} {'cross mean':>11}")
    for arm_name, arm_rows in by_arm.items():
        for pos in range(args.group_size):
            sel = [r for r in arm_rows if r["position"] == pos]
            if not sel:
                continue
            print(
                f"{arm_name:<10} {pos:>4} {len(sel):>4} "
                f"{sum(1 for r in sel if not r['passes']):>6} "
                f"{statistics.mean(r['cross_max_idf'] for r in sel):>11.3f}"
            )

    print("\n=== 5. output length and pair Jaccard ===")
    print(f"{'arm':<10} {'para words':>11} {'desc words':>11} {'pair jaccard':>13}")
    for arm_name, arm_rows in by_arm.items():
        para = [r for r in arm_rows if r["query_type"] == "paraphrase"]
        desc = {r["chunk_id"]: r for r in arm_rows if r["query_type"] == "description"}
        pairs = [
            jaccard(set(analyze(r["text"])), set(analyze(desc[r["chunk_id"]]["text"])))
            for r in para
            if r["chunk_id"] in desc
        ]
        print(
            f"{arm_name:<10} "
            f"{statistics.mean(r['n_words'] for r in para):>11.1f} "
            f"{statistics.mean(r['n_words'] for r in desc.values()):>11.1f} "
            f"{statistics.mean(pairs):>13.3f}"
        )

    print(f"\nrows: {out_path}")


if __name__ == "__main__":
    main()
