"""SUBAGENTS.md: score the pilot generations against their source fragments.

Reads `data/eval/pilot/generated.jsonl`, parses each reply, and measures
what the pilot exists to measure: how much of the source fragment's rare
vocabulary survived into the query. Per-query detail goes to
`data/eval/pilot/leak.jsonl`; the terminal gets aggregates only, because
the queries are derived from copyrighted lyrics.

Two things are checked that the leak metric alone would miss:

  - the reply's *shape*. A markdown fence is stripped rather than
    rejected: unwrapping a fence is deterministic, not a guess at intent,
    and the contents are then either valid JSON or they are not. But
    strict and fence-recovered replies are still counted separately,
    because the fence rate is a property of the prompt worth watching --
    a tolerant parser that reported one number would hide it. Anything
    beyond a fence stays an error, since recovering it would mean
    guessing.
  - title and artist leakage, separately from fragment leakage. A song
    title is often absent from the fragment that was shown to the
    generator yet present in the indexed text, so a title word can hand
    BM25 an anchor without ever registering as fragment overlap.

The gate is expressed as a document frequency, not as a raw IDF number.
"a term occurring in at most K chunks" is a claim about the corpus that
can be checked; "IDF above 8.5" is a magic constant.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, ".")  # run from the repo root, as every other script here does

import joblib

from lyrics_search.core.lexical_leak import idf_weighted_overlap
from lyrics_search.paths import work_dir

FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)
REQUIRED_FIELDS = ("paraphrase", "description")


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _parse_reply(raw: str) -> tuple[dict | None, str]:
    """Return the parsed object and how strictly it complied.

    `strict` means the contract was met as written: the first character
    is `{`, the last is `}`, and the whole reply is one JSON object with
    both fields non-empty. `fenced` means the same object was recovered
    after stripping a markdown code fence -- accepted, but counted, so
    the prompt's fence rate stays visible.
    """
    stripped = raw.strip()
    mode = "strict"
    body = stripped
    if not (stripped.startswith("{") and stripped.endswith("}")):
        match = FENCE.match(stripped)
        if not match:
            return None, "unparseable"
        body = match.group(1)
        mode = "fenced"

    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        return None, "unparseable"
    if not isinstance(obj, dict):
        return None, "unparseable"
    for field in REQUIRED_FIELDS:
        value = obj.get(field)
        if not isinstance(value, str) or not value.strip():
            return None, "missing_field"
    return obj, mode


def _df_from_idf(idf: float, n_docs: int) -> float:
    """Invert sklearn's smoothed IDF back to a document count.

    `idf = ln((1 + n) / (1 + df)) + 1`, so the gate can be stated in
    documents rather than in an opaque log-scale threshold.
    """
    return (1 + n_docs) / math.exp(idf - 1) - 1


def _idf_from_df(df: float, n_docs: int) -> float:
    return math.log((1 + n_docs) / (1 + df)) + 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--corpus", default="full")
    parser.add_argument("--chunking", default="sections")
    parser.add_argument("--fragments", default="data/eval/pilot/fragments.jsonl")
    parser.add_argument("--generated", default="data/eval/pilot/generated.jsonl")
    parser.add_argument("--out", default="data/eval/pilot/leak.jsonl")
    parser.add_argument(
        "--gate-df",
        type=int,
        default=100,
        help="a shared term occurring in at most this many chunks fails the query",
    )
    args = parser.parse_args()

    root = Path(args.data_root)
    state_path = (
        work_dir(root, args.corpus, args.chunking) / "embeddings" / "tfidf" / "fitted_state.joblib"
    )
    if not state_path.exists():
        raise FileNotFoundError(f"missing fitted tfidf state: {state_path}")

    vectorizer = joblib.load(state_path)
    analyze = vectorizer.build_analyzer()
    idf_values = vectorizer.idf_
    idf = {term: float(idf_values[i]) for term, i in vectorizer.vocabulary_.items()}
    # sklearn keeps no document count, but it is recoverable: the rarest
    # possible term has df == 1, so n = exp(max_idf - 1) * 2 - 1.
    n_docs = round(math.exp(max(idf.values()) - 1) * 2 - 1)
    gate_idf = _idf_from_df(args.gate_df, n_docs)

    fragments = {r["chunk_id"]: r for r in _read_jsonl(Path(args.fragments)) if "chunk_id" in r}

    records = []
    for row in _read_jsonl(Path(args.generated)):
        fragment = fragments[row["chunk_id"]]
        source_terms = analyze(fragment["text"])
        meta_terms = set(analyze(fragment["title"])) | set(analyze(fragment["artist"]))

        obj, mode = _parse_reply(row["raw_reply"])
        record = {
            "chunk_id": row["chunk_id"],
            "song_id": row["song_id"],
            "genre": fragment["genre"],
            "is_translation": fragment["is_translation"],
            "prompt_version": row["prompt_version"],
            "parse": mode,
        }
        if obj is None:
            records.append(record)
            continue

        for field in REQUIRED_FIELDS:
            query_terms = analyze(obj[field])
            score = idf_weighted_overlap(query_terms, source_terms, idf)
            meta_hits = sorted({t for t in query_terms if t in meta_terms and t in idf})
            record[field] = {
                "text": obj[field],
                "n_terms": len(query_terms),
                "ratio": score.ratio,
                "max_shared_idf": score.max_shared_idf,
                "max_shared_df": _df_from_idf(score.max_shared_idf, n_docs)
                if score.max_shared_idf
                else None,
                "shared": list(score.shared),
                "unknown": list(score.unknown),
                "meta_shared": meta_hits,
                "meta_max_idf": max((idf[t] for t in meta_hits), default=0.0),
                "fails_gate": score.max_shared_idf >= gate_idf,
            }
        para = set(analyze(obj["paraphrase"]))
        desc = set(analyze(obj["description"]))
        record["pair_jaccard"] = len(para & desc) / len(para | desc) if (para | desc) else 0.0
        records.append(record)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    parsed = [r for r in records if "paraphrase" in r]
    print(f"corpus chunks (recovered): {n_docs}")
    print(f"gate: shared term in <= {args.gate_df} chunks (idf >= {gate_idf:.3f})")
    print(f"replies              : {len(records)}")
    for mode in ("strict", "fenced", "missing_field", "unparseable"):
        print(f"  parse {mode:<14}: {sum(1 for r in records if r['parse'] == mode)}")

    for field in REQUIRED_FIELDS:
        scores = [r[field] for r in parsed]
        ratios = [s["ratio"] for s in scores]
        maxidf = [s["max_shared_idf"] for s in scores]
        print(f"\n{field}:")
        print(
            f"  terms  mean/min/max : {statistics.mean(s['n_terms'] for s in scores):.1f} / "
            f"{min(s['n_terms'] for s in scores)} / {max(s['n_terms'] for s in scores)}"
        )
        print(f"  idf ratio mean/max  : {statistics.mean(ratios):.3f} / {max(ratios):.3f}")
        print(f"  max_shared_idf mean : {statistics.mean(maxidf):.3f}")
        print(
            f"  max_shared_idf worst: {max(maxidf):.3f} "
            f"(df ~ {_df_from_idf(max(maxidf), n_docs):.0f} chunks)"
        )
        print(
            f"  fails gate          : {sum(1 for s in scores if s['fails_gate'])} / {len(scores)}"
        )
        # Counted at the gate, not by any shared term: a query and a title
        # both containing "the" is not leakage, and reporting it as such
        # would bury the one case that is.
        print(
            f"  title/artist leak   : "
            f"{sum(1 for s in scores if s['meta_max_idf'] >= gate_idf)} / {len(scores)} "
            f"(rarest shared with title/artist: "
            f"{max(s['meta_max_idf'] for s in scores):.2f})"
        )

    jac = [r["pair_jaccard"] for r in parsed]
    print(f"\npair jaccard mean/max : {statistics.mean(jac):.3f} / {max(jac):.3f}")
    print(f"out: {out_path}")


if __name__ == "__main__":
    main()
