"""EVAL-PREP §6: `eval` -- config + eval set in, metrics table out.

    python -m lyrics_search.runners.eval --config configs/full_dense_faiss.yaml \
        --eval-set data/eval/queries.jsonl

Online only, like `search`: everything is loaded, nothing is fit or built
(SPEC-03 §1). One query at a time through the same `build_retriever()`
path the CLI uses, so what is measured is the system as it actually runs.

Slicing (the valuable part). Slices are cut on properties of a query's
*relevant* songs -- the ground truth -- not of what came back. Slicing on
the results would make the slice depend on the retriever being measured,
and two retrievers would then be scored on different partitions of the
same eval set.

A query can have several relevant songs that disagree on a property. Such
a query is put in a `mixed` bucket rather than assigned to either side or
dropped: assigning it would corrupt both cells, and dropping it would
make the slice's `n` values silently fail to add up to the total.

Four dimensions:
  - `split_by` x `force_split` -- does retrieval degrade on songs whose
    lyrics carried no structural markup? A song's `split_by` is the value
    shared by all its chunks, or `mixed`; its `force_split` is True if any
    chunk needed level-4 length packing. Under the `whole_song` chunking
    arm this dimension degenerates to a single `none/False` cell by
    construction -- correctly so, and visibly.
  - `is_translation` -- from raw.jsonl, set during preprocess.
  - `genre` -- from data/<corpus>/genre.jsonl. Skipped with a printed
    note when that file is absent, since it is optional analysis metadata
    built by a separate one-off script, not a pipeline artifact.
  - `query_type` -- from the eval set itself, when its records carry one.

Every row reports `n` alongside a bootstrap confidence interval
(core/metrics.py). Small slices will not show significant differences,
and the interval is what makes that visible in the table instead of
discovered afterwards.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from lyrics_search.config import ExperimentConfig
from lyrics_search.contracts import Chunk, RawSong
from lyrics_search.core.metrics import mean_ci, ndcg_at_k, recall_at_k, reciprocal_rank
from lyrics_search.core.sections import CHUNKING_WHOLE_SONG
from lyrics_search.eval_set import EvalQuery, load_eval_set
from lyrics_search.genre import genre_path, load_genre_lookup
from lyrics_search.paths import raw_path as _raw_path
from lyrics_search.paths import work_dir as _work_dir
from lyrics_search.retrievers.loading import load_chunk_lookup, load_song_corpus
from lyrics_search.runners.search import build_retriever

MIXED = "mixed"
UNKNOWN = "(unknown)"   # metadata absent for this song, distinct from disagreeing
OVERALL = "(overall)"

# Ordered so the printed table reads left to right from strictest to
# most forgiving. Each is f(ranked_song_ids, relevant_song_ids) -> float.
METRICS: dict[str, callable] = {
    "recall@1": lambda r, rel: recall_at_k(r, rel, 1),
    "recall@5": lambda r, rel: recall_at_k(r, rel, 5),
    "recall@10": lambda r, rel: recall_at_k(r, rel, 10),
    "MRR": reciprocal_rank,
    "nDCG@10": lambda r, rel: ndcg_at_k(r, rel, 10),
}


@dataclass(frozen=True)
class QueryOutcome:
    """One query's ranked song_ids plus its per-metric scores."""
    query: EvalQuery
    ranked: tuple[str, ...]
    status: str
    scores: dict[str, float]


@dataclass(frozen=True)
class SliceRow:
    dimension: str
    value: str
    n: int
    # metric name -> (mean, ci_lo, ci_hi)
    stats: dict[str, tuple[float, float, float]]


def _chunks_by_song(chunk_lookup: dict[str, Chunk]) -> dict[str, list[Chunk]]:
    by_song: dict[str, list[Chunk]] = {}
    for chunk in chunk_lookup.values():
        by_song.setdefault(chunk.song_id, []).append(chunk)
    return by_song


def _song_split_label(chunks: list[Chunk]) -> str:
    """`split_by/force_split` for one song. See module docstring.

    No chunks means the song survived ingest but every one of its segments
    was filtered out in preprocess, so it is not in the index at all. That
    is UNKNOWN, not MIXED -- and a query whose relevant song lands here is
    worth seeing, because it cannot be answered by any retriever.
    """
    if not chunks:
        return UNKNOWN
    split_values = {c.split_by for c in chunks}
    split_by = split_values.pop() if len(split_values) == 1 else MIXED
    forced = any(c.force_split for c in chunks)
    return f"{split_by}/force_split={forced}"


def _query_label(query: EvalQuery, song_label: dict[str, str]) -> str:
    """A query's label is the one its relevant songs agree on.

    A song missing from `song_label` gets UNKNOWN rather than MIXED: absent
    metadata and disagreeing metadata are different facts, and folding the
    first into the second would make a lookup with poor coverage look like
    an eval set full of heterogeneous queries.
    """
    labels = {song_label.get(sid, UNKNOWN) for sid in query.relevant_song_ids}
    return labels.pop() if len(labels) == 1 else MIXED


def run_queries(
    config: ExperimentConfig,
    queries: list[EvalQuery],
    data_root: Path | str = "data",
) -> tuple[list[QueryOutcome], float]:
    retriever, load_s = build_retriever(config, data_root=data_root)
    outcomes: list[QueryOutcome] = []
    for q in queries:
        result = retriever.search(q.query, config.retrieval.top_k)
        ranked = tuple(hit.song_id for hit in result.hits)
        scores = {name: fn(ranked, q.relevant_song_ids) for name, fn in METRICS.items()}
        outcomes.append(QueryOutcome(q, ranked, result.status, scores))
    return outcomes, load_s


def _row(dimension: str, value: str, outcomes: list[QueryOutcome]) -> SliceRow:
    stats = {
        name: mean_ci([o.scores[name] for o in outcomes])
        for name in METRICS
    }
    return SliceRow(dimension, value, len(outcomes), stats)


def build_table(
    outcomes: list[QueryOutcome],
    slicers: dict[str, dict[EvalQuery, str]],
) -> list[SliceRow]:
    """`slicers` maps a dimension name to {EvalQuery -> label}.

    Keyed by the EvalQuery itself, not by its text: an eval set may legally
    carry the same query string twice with different relevant sets, and a
    text key would silently merge them and lose one. EvalQuery is a frozen
    dataclass over a tuple, so it is hashable for free.
    """
    rows = [_row(OVERALL, "", outcomes)]
    for dimension, labels in slicers.items():
        buckets: dict[str, list[QueryOutcome]] = {}
        for o in outcomes:
            buckets.setdefault(labels[o.query], []).append(o)
        for value in sorted(buckets):
            rows.append(_row(dimension, value, buckets[value]))
    return rows


def run_eval(
    config: ExperimentConfig,
    eval_set_path: Path | str,
    data_root: Path | str = "data",
) -> tuple[list[SliceRow], list[QueryOutcome]]:
    work_dir = _work_dir(data_root, config.corpus, config.chunking)
    chunk_lookup = load_chunk_lookup(work_dir / "chunks.jsonl")
    songs: dict[str, RawSong] = load_song_corpus(_raw_path(data_root, config.corpus))

    queries = load_eval_set(eval_set_path, songs.keys())
    outcomes, load_s = run_queries(config, queries, data_root=data_root)

    by_song = _chunks_by_song(chunk_lookup)
    split_label = {sid: _song_split_label(by_song.get(sid, [])) for sid in songs}
    translation_label = {sid: str(song.is_translation) for sid, song in songs.items()}

    slicers: dict[str, dict[EvalQuery, str]] = {
        "split_by x force_split": {q: _query_label(q, split_label) for q in queries},
        "is_translation": {q: _query_label(q, translation_label) for q in queries},
    }

    if genre_path(data_root, config.corpus).exists():
        genre = load_genre_lookup(data_root, config.corpus)
        slicers["genre"] = {q: _query_label(q, genre) for q in queries}
    else:
        print(f"[genre] slice skipped -- no {genre_path(data_root, config.corpus)} "
              f"(build it with scripts/build_genre_lookup.py)")

    if any(q.query_type for q in queries):
        slicers["query_type"] = {q: q.query_type or "(none)" for q in queries}

    rows = build_table(outcomes, slicers)
    _print_report(config, eval_set_path, rows, outcomes, load_s)
    return rows, outcomes


def _print_report(
    config: ExperimentConfig,
    eval_set_path: Path | str,
    rows: list[SliceRow],
    outcomes: list[QueryOutcome],
    load_s: float,
) -> None:
    print(
        f"corpus={config.corpus} chunking={config.chunking} "
        f"embedder={config.embedder.name} index={config.index} "
        f"mode={config.retrieval.mode} top_k={config.retrieval.top_k}"
    )
    print(f"eval set: {eval_set_path} ({len(outcomes)} queries), load={load_s:.1f}s")

    non_ok = {}
    for o in outcomes:
        if o.status != "ok":
            non_ok[o.status] = non_ok.get(o.status, 0) + 1
    if non_ok:
        # These score 0 on every metric. Reported separately so a lexical
        # or tfidf arm's OOV rate is not mistaken for poor ranking.
        print(f"non-ok query statuses (scored as 0): {non_ok}")
    if config.chunking == CHUNKING_WHOLE_SONG:
        print("note: the whole_song arm has one chunk per song, so the "
              "split_by x force_split slice degenerates to a single cell.")

    header = f"{'dimension':<24}{'value':<28}{'n':>5}  " + "  ".join(
        f"{name:>22}" for name in METRICS
    )
    print()
    print(header)
    print("-" * len(header))
    for row in rows:
        cells = []
        for name in METRICS:
            mean, lo, hi = row.stats[name]
            cells.append(f"{mean:.3f} [{lo:.3f},{hi:.3f}]".rjust(22))
        print(f"{row.dimension:<24}{row.value:<28}{row.n:>5}  " + "  ".join(cells))


def main() -> None:
    import argparse

    from lyrics_search.config import load_config

    parser = argparse.ArgumentParser(description="EVAL-PREP §6 eval runner")
    parser.add_argument("--config", required=True, help="path to an ExperimentConfig YAML")
    parser.add_argument("--eval-set", required=True, help="path to an eval-set JSONL")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--json-out", default=None, help="also write the table as JSON")
    args = parser.parse_args()

    config = load_config(args.config)
    rows, _outcomes = run_eval(config, args.eval_set, data_root=args.data_root)

    if args.json_out:
        payload = [
            {"dimension": r.dimension, "value": r.value, "n": r.n,
             "stats": {k: list(v) for k, v in r.stats.items()}}
            for r in rows
        ]
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
