"""EVAL-PREP §6: runners/eval.py.

Two kinds of test here, for two different failure modes.

The unit tests over `_song_split_label` / `_query_label` / `build_table`
pin the bucketing rules, because those are the part that can be wrong
while everything still runs and prints a plausible table. A query filed
into the wrong slice does not raise; it just moves a number.

The end-to-end tests build the golden mini corpus and score a real eval
set against it. They assert structure -- that every slice's `n` values
sum to the total, that a dimension appears exactly when its input does --
and never a metric value: retrieval quality on four songs is not a
result, and EVAL-PREP is explicit that no comparison result is to be
reported from this phase.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lyrics_search.config import ExperimentConfig, RetrievalConfig
from lyrics_search.contracts import Chunk
from lyrics_search.core.text import make_song_id
from lyrics_search.eval_set import EvalQuery
from lyrics_search.runners.build import run_build
from lyrics_search.runners.eval import (
    METRICS,
    MIXED,
    OVERALL,
    UNKNOWN,
    QueryOutcome,
    _query_label,
    _song_split_label,
    build_table,
    run_eval,
)
from lyrics_search.sources.jsonl_file import JsonlFileSource

GOLDEN = Path(__file__).parent / "golden" / "spec03_mini_corpus.jsonl"

SUNSHINE = make_song_id("The Sunbeams", "Walking On Sunshine Morning")
RAIN = make_song_id("Storm Riders", "Rain And Thunder Night")
MOUNTAIN = make_song_id("Summit Seekers", "Mountain Climbing Journey")
OCEAN = make_song_id("Deep Blue", "Ocean Waves Forever")


def _config(chunking: str = "sections") -> ExperimentConfig:
    return ExperimentConfig(
        corpus="demo", embedder="tfidf", index="numpy", chunking=chunking,
        retrieval=RetrievalConfig(mode="hybrid", top_k=10, return_n=5),
    )


def _write_eval_set(path: Path, records: list[dict]) -> Path:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def _default_records() -> list[dict]:
    return [
        {"_meta": "skipped, not a query"},
        {"query": "walking on sunshine", "relevant_song_ids": [SUNSHINE]},
        {"query": "rain and thunder", "relevant_song_ids": [RAIN]},
        {"query": "climbing the mountain", "relevant_song_ids": [MOUNTAIN]},
        {"query": "ocean waves crashing", "relevant_song_ids": [OCEAN]},
    ]


def _chunk(song_id: str, position: int, split_by: str, force_split: bool = False) -> Chunk:
    return Chunk(
        chunk_id=f"{song_id}:{position}", song_id=song_id, section=None,
        text="whatever", position=position, split_by=split_by, force_split=force_split,
    )


# --- bucketing rules ---------------------------------------------------


def test_a_songs_split_label_is_the_value_all_its_chunks_share():
    chunks = [_chunk("s", 0, "bracket_tag"), _chunk("s", 1, "bracket_tag")]
    assert _song_split_label(chunks) == "bracket_tag/force_split=False"


def test_a_song_whose_chunks_disagree_on_split_by_is_mixed():
    chunks = [_chunk("s", 0, "bracket_tag"), _chunk("s", 1, "blank_line")]
    assert _song_split_label(chunks) == f"{MIXED}/force_split=False"


def test_force_split_is_true_when_any_chunk_needed_length_packing():
    chunks = [_chunk("s", 0, "bracket_tag"), _chunk("s", 1, "bracket_tag", force_split=True)]
    assert _song_split_label(chunks) == "bracket_tag/force_split=True"


def test_a_song_with_no_chunks_is_unknown_not_mixed():
    """It survived ingest but every segment was filtered in preprocess, so
    it is not in the index -- a different fact from "its chunks disagree"."""
    assert _song_split_label([]) == UNKNOWN


def test_a_query_takes_the_label_its_relevant_songs_agree_on():
    query = EvalQuery("q", (SUNSHINE, RAIN))
    labels = {SUNSHINE: "True", RAIN: "True"}
    assert _query_label(query, labels) == "True"


def test_a_query_whose_relevant_songs_disagree_is_mixed_not_assigned_to_either():
    query = EvalQuery("q", (SUNSHINE, RAIN))
    labels = {SUNSHINE: "True", RAIN: "False"}
    assert _query_label(query, labels) == MIXED


def test_a_song_absent_from_the_lookup_is_unknown_not_mixed():
    query = EvalQuery("q", (SUNSHINE,))
    assert _query_label(query, {}) == UNKNOWN


def test_unknown_and_a_real_label_still_disagree_and_give_mixed():
    query = EvalQuery("q", (SUNSHINE, RAIN))
    assert _query_label(query, {SUNSHINE: "rock"}) == MIXED


# --- build_table -------------------------------------------------------


def _outcome(query: EvalQuery) -> QueryOutcome:
    return QueryOutcome(query, (), "ok", {name: 0.0 for name in METRICS})


def test_build_table_always_emits_an_overall_row_first():
    outcomes = [_outcome(EvalQuery("a", ("x",))), _outcome(EvalQuery("b", ("y",)))]
    rows = build_table(outcomes, {})
    assert len(rows) == 1
    assert rows[0].dimension == OVERALL
    assert rows[0].n == 2


def test_slice_n_values_sum_to_the_total_for_every_dimension():
    queries = [EvalQuery(f"q{i}", (f"s{i}",)) for i in range(6)]
    outcomes = [_outcome(q) for q in queries]
    slicers = {
        "parity": {q: str(i % 2) for i, q in enumerate(queries)},
        "third": {q: str(i % 3) for i, q in enumerate(queries)},
    }
    rows = build_table(outcomes, slicers)
    for dimension in ("parity", "third"):
        assert sum(r.n for r in rows if r.dimension == dimension) == len(outcomes)


def test_build_table_keys_on_the_query_object_not_its_text():
    """Two records may legally share a query string with different relevant
    sets. Keying on the text would merge them and lose one."""
    a = EvalQuery("same text", (SUNSHINE,))
    b = EvalQuery("same text", (RAIN,))
    assert a != b
    outcomes = [_outcome(a), _outcome(b)]
    rows = build_table(outcomes, {"dim": {a: "left", b: "right"}})
    assert {r.value: r.n for r in rows if r.dimension == "dim"} == {"left": 1, "right": 1}


def test_every_row_carries_an_interval_for_every_metric():
    outcomes = [_outcome(EvalQuery(f"q{i}", ("x",))) for i in range(4)]
    rows = build_table(outcomes, {})
    for row in rows:
        assert set(row.stats) == set(METRICS)
        for mean, lo, hi in row.stats.values():
            assert lo <= mean <= hi


# --- end to end --------------------------------------------------------


@pytest.fixture
def built_corpus(tmp_path):
    config = _config()
    run_build(config, data_root=tmp_path, source=JsonlFileSource(GOLDEN))
    return config, tmp_path


def test_run_eval_scores_every_query_and_reports_the_total(built_corpus, capsys):
    config, data_root = built_corpus
    eval_set = _write_eval_set(data_root / "eval.jsonl", _default_records())
    rows, outcomes = run_eval(config, eval_set, data_root=data_root)

    assert len(outcomes) == 4  # the _meta record is not a query
    overall = [r for r in rows if r.dimension == OVERALL]
    assert len(overall) == 1
    assert overall[0].n == 4
    assert "4 queries" in capsys.readouterr().out


def test_the_structural_slices_are_always_present(built_corpus):
    config, data_root = built_corpus
    eval_set = _write_eval_set(data_root / "eval.jsonl", _default_records())
    rows, outcomes = run_eval(config, eval_set, data_root=data_root)

    for dimension in ("split_by x force_split", "is_translation"):
        assert sum(r.n for r in rows if r.dimension == dimension) == len(outcomes)


def test_the_genre_slice_is_skipped_with_a_note_when_the_lookup_is_absent(built_corpus, capsys):
    config, data_root = built_corpus
    eval_set = _write_eval_set(data_root / "eval.jsonl", _default_records())
    rows, _outcomes = run_eval(config, eval_set, data_root=data_root)

    assert not any(r.dimension == "genre" for r in rows)
    assert "[genre] slice skipped" in capsys.readouterr().out


def test_the_genre_slice_appears_when_the_lookup_is_present(built_corpus):
    config, data_root = built_corpus
    genre = {SUNSHINE: "pop", RAIN: "rock", MOUNTAIN: "rock", OCEAN: "country"}
    with open(data_root / "demo" / "genre.jsonl", "w", encoding="utf-8") as f:
        for song_id, value in genre.items():
            f.write(json.dumps({"song_id": song_id, "genre": value}) + "\n")

    eval_set = _write_eval_set(data_root / "eval.jsonl", _default_records())
    rows, outcomes = run_eval(config, eval_set, data_root=data_root)

    genre_rows = {r.value: r.n for r in rows if r.dimension == "genre"}
    assert genre_rows == {"pop": 1, "rock": 2, "country": 1}
    assert sum(genre_rows.values()) == len(outcomes)


def test_the_query_type_slice_appears_only_when_records_carry_one(built_corpus):
    config, data_root = built_corpus

    without = _write_eval_set(data_root / "plain.jsonl", _default_records())
    rows, _ = run_eval(config, without, data_root=data_root)
    assert not any(r.dimension == "query_type" for r in rows)

    typed = _default_records()
    for i, record in enumerate(r for r in typed if "query" in r):
        record["query_type"] = "paraphrase" if i % 2 else "literal"
    with_types = _write_eval_set(data_root / "typed.jsonl", typed)
    rows, _ = run_eval(config, with_types, data_root=data_root)
    assert {r.value: r.n for r in rows if r.dimension == "query_type"} == {
        "literal": 2, "paraphrase": 2,
    }


def test_a_query_with_relevant_songs_from_two_genres_lands_in_mixed(built_corpus):
    config, data_root = built_corpus
    with open(data_root / "demo" / "genre.jsonl", "w", encoding="utf-8") as f:
        for song_id, value in {SUNSHINE: "pop", RAIN: "rock"}.items():
            f.write(json.dumps({"song_id": song_id, "genre": value}) + "\n")

    eval_set = _write_eval_set(data_root / "eval.jsonl", [
        {"query": "sunshine and rain", "relevant_song_ids": [SUNSHINE, RAIN]},
        {"query": "ocean waves", "relevant_song_ids": [OCEAN]},
    ])
    rows, _ = run_eval(config, eval_set, data_root=data_root)

    genre_rows = {r.value: r.n for r in rows if r.dimension == "genre"}
    assert genre_rows == {MIXED: 1, UNKNOWN: 1}  # OCEAN is not in the lookup


def test_an_unknown_relevant_song_id_fails_before_any_query_runs(built_corpus):
    config, data_root = built_corpus
    eval_set = _write_eval_set(data_root / "eval.jsonl", [
        {"query": "whatever", "relevant_song_ids": ["deadbeefdeadbeef"]},
    ])
    with pytest.raises(ValueError, match="deadbeefdeadbeef"):
        run_eval(config, eval_set, data_root=data_root)


def test_the_whole_song_arm_degenerates_to_a_single_split_cell(tmp_path, capsys):
    """EVAL-PREP §3: one chunk per song means no boundary and no forced
    split, so the dimension collapses -- correctly, and visibly."""
    config = _config(chunking="whole_song")
    run_build(config, data_root=tmp_path, source=JsonlFileSource(GOLDEN))
    eval_set = _write_eval_set(tmp_path / "eval.jsonl", _default_records())
    rows, outcomes = run_eval(config, eval_set, data_root=tmp_path)

    split_rows = [r for r in rows if r.dimension == "split_by x force_split"]
    assert len(split_rows) == 1
    assert split_rows[0].value == "none/force_split=False"
    assert split_rows[0].n == len(outcomes)
    assert "degenerates to a single cell" in capsys.readouterr().out
