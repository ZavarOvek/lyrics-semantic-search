"""EVAL-PREP §5: the eval-set loader.

Two behaviours carry the weight here and both are about miscounting
rather than crashing: `_meta` records must not become zero-scoring
queries, and an unknown song_id must not become a silent miss. The rest
is ordinary shape validation, tested mainly to confirm the failure names
the offending line.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lyrics_search.eval_set import EvalQuery, load_eval_set

KNOWN = {"aaa", "bbb", "ccc"}
GOLDEN = Path(__file__).parent / "golden" / "spec03_eval_queries.jsonl"


def _write(tmp_path: Path, records: list) -> Path:
    path = tmp_path / "eval.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) if not isinstance(r, str) else r)
            f.write("\n")
    return path


def test_loads_a_well_formed_set(tmp_path):
    path = _write(tmp_path, [
        {"query": "one", "relevant_song_ids": ["aaa"], "query_type": "paraphrase"},
        {"query": "two", "relevant_song_ids": ["bbb", "ccc"]},
    ])
    queries = load_eval_set(path, KNOWN)
    assert queries == [
        EvalQuery("one", ("aaa",), "paraphrase"),
        EvalQuery("two", ("bbb", "ccc"), None),
    ]


def test_meta_records_are_skipped_not_scored(tmp_path):
    path = _write(tmp_path, [
        {"_meta": "methodology warning"},
        {"query": "one", "relevant_song_ids": ["aaa"]},
    ])
    queries = load_eval_set(path, KNOWN)
    assert len(queries) == 1
    assert queries[0].query == "one"


def test_a_meta_record_is_skipped_even_when_it_also_looks_like_a_query(tmp_path):
    """`_meta` is the signal, not the absence of other fields -- a record
    carrying both must still not be counted."""
    path = _write(tmp_path, [
        {"_meta": "note", "query": "x", "relevant_song_ids": ["aaa"]},
        {"query": "real", "relevant_song_ids": ["aaa"]},
    ])
    assert [q.query for q in load_eval_set(path, KNOWN)] == ["real"]


def test_the_bootstrap_golden_file_loads_with_its_meta_record_skipped():
    with open(GOLDEN, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    known = {sid for r in records for sid in r.get("relevant_song_ids", [])}
    queries = load_eval_set(GOLDEN, known)
    assert len(queries) == len(records) - 1  # the one leading _meta record
    assert all("_meta" not in q.query for q in queries)


def test_an_unknown_song_id_fails_loudly_and_names_it(tmp_path):
    path = _write(tmp_path, [
        {"query": "one", "relevant_song_ids": ["aaa"]},
        {"query": "two", "relevant_song_ids": ["aaa", "deadbeef"]},
    ])
    with pytest.raises(ValueError) as exc:
        load_eval_set(path, KNOWN)
    message = str(exc.value)
    assert "deadbeef" in message
    assert ":2:" in message  # the offending line


def test_blank_lines_are_ignored(tmp_path):
    path = tmp_path / "eval.jsonl"
    path.write_text(
        '\n{"query": "one", "relevant_song_ids": ["aaa"]}\n\n',
        encoding="utf-8",
    )
    assert len(load_eval_set(path, KNOWN)) == 1


def test_duplicate_relevant_ids_are_collapsed(tmp_path):
    path = _write(tmp_path, [{"query": "one", "relevant_song_ids": ["aaa", "aaa", "bbb"]}])
    assert load_eval_set(path, KNOWN)[0].relevant_song_ids == ("aaa", "bbb")


@pytest.mark.parametrize("record,fragment", [
    ({"relevant_song_ids": ["aaa"]}, "`query`"),
    ({"query": "  ", "relevant_song_ids": ["aaa"]}, "`query`"),
    ({"query": "one"}, "relevant_song_ids"),
    ({"query": "one", "relevant_song_ids": []}, "relevant_song_ids"),
    ({"query": "one", "relevant_song_ids": "aaa"}, "relevant_song_ids"),
    ({"query": "one", "relevant_song_ids": [1]}, "not a string"),
    ({"query": "one", "relevant_song_ids": ["aaa"], "query_type": 7}, "`query_type`"),
])
def test_malformed_records_fail_loudly(tmp_path, record, fragment):
    path = _write(tmp_path, [record])
    with pytest.raises(ValueError, match=fragment.replace("`", "")):
        load_eval_set(path, KNOWN)


def test_a_non_object_line_fails_loudly(tmp_path):
    path = _write(tmp_path, [["query", "one"]])
    with pytest.raises(ValueError, match="expected a JSON object"):
        load_eval_set(path, KNOWN)


def test_invalid_json_names_the_line(tmp_path):
    path = _write(tmp_path, ["{not json"])
    with pytest.raises(ValueError, match=":1:"):
        load_eval_set(path, KNOWN)


def test_a_set_with_no_queries_at_all_fails(tmp_path):
    path = _write(tmp_path, [{"_meta": "only a note"}])
    with pytest.raises(ValueError, match="no queries"):
        load_eval_set(path, KNOWN)


def test_a_missing_file_names_the_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="nope.jsonl"):
        load_eval_set(tmp_path / "nope.jsonl", KNOWN)
