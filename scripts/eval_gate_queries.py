"""EVAL-AUTO §3-4: score the generated queries, gate them, assemble the set.

Reads every reply in `data/eval/auto/raw/*.jsonl`, scores each query
against the fragment it was generated from, drops the ones that leak, and
writes the survivors to `data/eval/auto/queries.jsonl`.

## The gate

A query fails if it shares a term with its source fragment that occurs in
at most `--gate-df` chunks of the corpus. The threshold is a document
count and not an IDF number on purpose: "a term occurring in at most 100
of 181 471 chunks" is a checkable claim about the corpus, while "IDF above
8.5" is a magic constant.

Why a mechanical gate rather than a better prompt is argued in
`notes/decisions/corpus-relative-rarity.md`. In short: the generator is
asked not to reuse distinctive words, and judges that against English at
large, but the quantity that matters is rarity *in this corpus*. `amid`
is an ordinary preposition occurring in nine chunks here. No prompt closes
that gap, because the model has no access to the corpus term
distribution.

## What failing costs

A failed query is regenerated, up to `--max-attempts` times. The unit
that leaves is the *query*, not the song: a leaking description says
nothing about whether the paraphrase leaks, and dropping a clean
paraphrase because its sibling failed would discard evidence for no
reason. Song-level loss is reported separately anyway, because a song
that loses its paraphrase has lost the known-item probe that the fragment
was drawn for.

## The two numbers that can invalidate the sample

Reported separately, because they answer different questions:

  - **final attrition** -- how many queries left the set after all
    attempts were spent.
  - **whether the same songs fail repeatedly**. Attempts are not
    independent. A song built around distinctive vocabulary is hard to
    paraphrase cleanly on attempt one and hard for the same reason on
    attempts two and three. If repeat failures concentrate, the loss is
    not random attrition but systematic removal of the songs where dense
    retrieval should have the largest advantage over lexical -- which
    biases the benchmark toward the lexical baseline.

Per-attempt failure rate is not a substitute for either.

## The title and artist gate is a control, not a filter

The generator is never shown the title or artist, so a query sharing a
rare term with them should be impossible rather than merely rare. The
check is kept precisely to prove the channel is closed; it is reported
and never acted on. If it ever fires, the harness is leaking, and no
amount of regeneration is the right response to that.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")  # run from the repo root, as every other script here does

import joblib

from lyrics_search.core.lexical_leak import idf_weighted_overlap
from lyrics_search.paths import raw_path, work_dir

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


def load_replies(raw_dir: Path) -> dict[str, list[dict]]:
    """All replies per chunk_id, ordered by attempt.

    Attempts accumulate rather than overwrite: the history is the input to
    the repeat-failure question, so a regeneration that replaced its
    predecessor would destroy the number it exists to produce.
    """
    by_chunk: dict[str, list[dict]] = {}
    for path in sorted(raw_dir.glob("*.jsonl")):
        for record in _read_jsonl(path):
            by_chunk.setdefault(record["chunk_id"], []).append(record)
    for replies in by_chunk.values():
        replies.sort(key=lambda r: r.get("attempt", 1))
    return by_chunk


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--corpus", default="full")
    parser.add_argument("--chunking", default="sections")
    parser.add_argument("--sample", default="data/eval/auto/sample.jsonl")
    parser.add_argument("--raw-dir", default="data/eval/auto/raw")
    parser.add_argument("--scored", default="data/eval/auto/scored.jsonl")
    parser.add_argument("--retry", default="data/eval/auto/retry.txt")
    parser.add_argument("--out", default="data/eval/auto/queries.jsonl")
    parser.add_argument("--gate-df", type=int, default=100)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument(
        "--partial",
        action="store_true",
        help="score what exists instead of refusing an incomplete run",
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

    fragments = {r["chunk_id"]: r for r in _read_jsonl(Path(args.sample)) if "chunk_id" in r}
    songs = {r["song_id"]: r for r in _read_jsonl(raw_path(root, args.corpus))}
    replies = load_replies(Path(args.raw_dir))

    missing = [cid for cid in fragments if cid not in replies]
    if missing and not args.partial:
        raise SystemExit(
            f"{len(missing)} of {len(fragments)} fragments have no reply. "
            f"Assembling now would produce a set that is smaller than the "
            f"design without saying so. Finish generation, or pass --partial "
            f"to score what exists."
        )

    scored: list[dict] = []
    for chunk_id, attempts in sorted(replies.items()):
        fragment = fragments[chunk_id]
        source_terms = analyze(fragment["text"])
        song = songs[fragment["song_id"]]
        meta_terms = set(analyze(song["title"])) | set(analyze(song["artist"]))

        for reply in attempts:
            attempt = reply.get("attempt", 1)
            for field in QUERY_TYPES:
                text = reply[field]
                query_terms = analyze(text)
                score = idf_weighted_overlap(query_terms, source_terms, idf)
                meta_hits = sorted({t for t in query_terms if t in meta_terms and t in idf})
                scored.append(
                    {
                        "chunk_id": chunk_id,
                        "song_id": fragment["song_id"],
                        "stratum": fragment["stratum"],
                        "genre": fragment["genre"],
                        "is_translation": fragment["is_translation"],
                        "query_type": field,
                        "attempt": attempt,
                        "text": text,
                        "n_terms": len(query_terms),
                        "n_words": len(text.split()),
                        "ratio": score.ratio,
                        "max_shared_idf": score.max_shared_idf,
                        "max_shared_df": _df_from_idf(score.max_shared_idf, n_docs)
                        if score.max_shared_idf
                        else None,
                        "shared": list(score.shared),
                        "meta_max_idf": max((idf[t] for t in meta_hits), default=0.0),
                        "passes": score.max_shared_idf < gate_idf,
                    }
                )

    scored_path = Path(args.scored)
    scored_path.parent.mkdir(parents=True, exist_ok=True)
    with scored_path.open("w", encoding="utf-8") as handle:
        for record in scored:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    # The accepted query per (chunk, type) is the first attempt that
    # passes. Taking the last would silently prefer a later regeneration
    # over an earlier clean one and make the attempt count meaningless.
    accepted: dict[tuple[str, str], dict] = {}
    attempts_used: dict[tuple[str, str], int] = {}
    for record in scored:
        key = (record["chunk_id"], record["query_type"])
        attempts_used[key] = max(attempts_used.get(key, 0), record["attempt"])
        if record["passes"] and key not in accepted:
            accepted[key] = record

    exhausted = [
        key
        for key in attempts_used
        if key not in accepted and attempts_used[key] >= args.max_attempts
    ]
    pending = [
        key
        for key in attempts_used
        if key not in accepted and attempts_used[key] < args.max_attempts
    ]

    retry_path = Path(args.retry)
    retry_ids = sorted({cid for cid, _ in pending})
    retry_path.write_text("\n".join(retry_ids) + ("\n" if retry_ids else ""), encoding="utf-8")

    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as handle:
        meta = {
            "_meta": {
                "source": "EVAL-AUTO, generated by eval-query-generator",
                "sample": args.sample,
                "gate_df": args.gate_df,
                "gate_idf": round(gate_idf, 4),
                "corpus_chunks": n_docs,
                "max_attempts": args.max_attempts,
                "songs": len({c for c, _ in accepted}),
                "queries": len(accepted),
                "partial": bool(missing),
            }
        }
        handle.write(json.dumps(meta, ensure_ascii=False) + "\n")
        for (_chunk_id, field), record in sorted(accepted.items()):
            handle.write(
                json.dumps(
                    {
                        "query": record["text"],
                        "relevant_song_ids": [record["song_id"]],
                        "query_type": field,
                        "stratum": record["stratum"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"corpus chunks (recovered): {n_docs}")
    print(f"gate: shared term in <= {args.gate_df} chunks (idf >= {gate_idf:.3f})")
    print(f"fragments in sample      : {len(fragments)}")
    print(f"fragments with a reply   : {len(replies)}  (missing {len(missing)})")
    print(f"scored query-attempts    : {len(scored)}")

    for field in QUERY_TYPES:
        first = [r for r in scored if r["query_type"] == field and r["attempt"] == 1]
        if not first:
            continue
        fails = [r for r in first if not r["passes"]]
        print(f"\n{field} (attempt 1, n={len(first)}):")
        print(
            f"  words  mean/min/max : {statistics.mean(r['n_words'] for r in first):.1f} / "
            f"{min(r['n_words'] for r in first)} / {max(r['n_words'] for r in first)}"
        )
        print(
            f"  idf ratio mean/max  : {statistics.mean(r['ratio'] for r in first):.3f} / "
            f"{max(r['ratio'] for r in first):.3f}"
        )
        worst = max(r["max_shared_idf"] for r in first)
        print(
            f"  max_shared_idf worst: {worst:.3f} (df ~ {_df_from_idf(worst, n_docs):.0f} chunks)"
        )
        print(f"  fails gate          : {len(fails)} / {len(first)}")
        # A control, not a filter: the generator never saw the title or
        # artist, so anything above the gate here means the harness leaked.
        leaks = [r for r in first if r["meta_max_idf"] >= gate_idf]
        print(f"  title/artist control: {len(leaks)} / {len(first)} above the gate")

    # Degeneracy: two queries that are near-copies of each other measure
    # one thing twice. Checked on the accepted set, since that is what ships.
    texts = [r["text"] for r in accepted.values()]
    duplicates = [t for t, c in Counter(texts).items() if c > 1]
    print(f"\naccepted queries         : {len(accepted)}")
    print(f"  exact duplicate texts  : {len(duplicates)}")
    print(f"  by type                : {dict(Counter(f for _, f in accepted))}")
    print(f"  by stratum             : {dict(Counter(r['stratum'] for r in accepted.values()))}")

    print(f"\nstill retryable          : {len(pending)} queries ({len(retry_ids)} songs)")
    print(f"dropped after {args.max_attempts} attempts: {len(exhausted)} queries")
    if exhausted:
        by_type = Counter(f for _, f in exhausted)
        lost_songs = {cid for cid, f in exhausted if f == "paraphrase"}
        print(f"  by type                : {dict(by_type)}")
        print(f"  songs losing paraphrase: {len(lost_songs)}")

    # The correlated-attrition question. A song whose first attempt failed
    # and whose later attempts also failed is the case that matters; a
    # count of first-attempt failures alone cannot distinguish it.
    retried = {key for key in attempts_used if attempts_used[key] > 1}
    if retried:
        repeat = [key for key in retried if key not in accepted]
        print(f"\nqueries regenerated      : {len(retried)}")
        print(f"  still failing after    : {len(repeat)}")
        print(
            f"  repeat-failure rate    : {len(repeat) / len(retried):.1%} "
            f"(vs. the attempt-1 rate above -- if this is higher, failures "
            f"are correlated and the loss is not random)"
        )
    else:
        print("\nno regenerations yet: the repeat-failure number is not yet defined")

    print(f"\nscored: {scored_path}")
    print(f"retry : {retry_path}")
    print(f"out   : {out_path}")


if __name__ == "__main__":
    main()
