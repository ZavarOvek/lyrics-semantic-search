"""EVAL-PREP §5: the eval-set loader.

Two behaviours carry the weight here and both are about miscounting
rather than crashing: `_meta` records must not become zero-scoring
queries, and an unknown song_id must not become a silent miss. The rest
is ordinary shape validation, tested mainly to confirm the failure names
the offending line.
"""

from __future__ import annotations

import json
import re
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
    path = _write(
        tmp_path,
        [
            {"query": "one", "relevant_song_ids": ["aaa"], "query_type": "paraphrase"},
            {"query": "two", "relevant_song_ids": ["bbb", "ccc"]},
        ],
    )
    queries = load_eval_set(path, KNOWN)
    assert queries == [
        EvalQuery("one", ("aaa",), "paraphrase"),
        EvalQuery("two", ("bbb", "ccc"), None),
    ]


def test_meta_records_are_skipped_not_scored(tmp_path):
    path = _write(
        tmp_path,
        [
            {"_meta": "methodology warning"},
            {"query": "one", "relevant_song_ids": ["aaa"]},
        ],
    )
    queries = load_eval_set(path, KNOWN)
    assert len(queries) == 1
    assert queries[0].query == "one"


def test_a_meta_record_is_skipped_even_when_it_also_looks_like_a_query(tmp_path):
    """`_meta` is the signal, not the absence of other fields -- a record
    carrying both must still not be counted."""
    path = _write(
        tmp_path,
        [
            {"_meta": "note", "query": "x", "relevant_song_ids": ["aaa"]},
            {"query": "real", "relevant_song_ids": ["aaa"]},
        ],
    )
    assert [q.query for q in load_eval_set(path, KNOWN)] == ["real"]


def test_the_bootstrap_golden_file_loads_with_its_meta_record_skipped():
    with open(GOLDEN, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    known = {sid for r in records for sid in r.get("relevant_song_ids", [])}
    queries = load_eval_set(GOLDEN, known)
    assert len(queries) == len(records) - 1  # the one leading _meta record
    assert all("_meta" not in q.query for q in queries)


def test_an_unknown_song_id_fails_loudly_and_names_it(tmp_path):
    path = _write(
        tmp_path,
        [
            {"query": "one", "relevant_song_ids": ["aaa"]},
            {"query": "two", "relevant_song_ids": ["aaa", "deadbeef"]},
        ],
    )
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


@pytest.mark.parametrize(
    "record,fragment",
    [
        ({"relevant_song_ids": ["aaa"]}, "`query`"),
        ({"query": "  ", "relevant_song_ids": ["aaa"]}, "`query`"),
        ({"query": "one"}, "relevant_song_ids"),
        ({"query": "one", "relevant_song_ids": []}, "relevant_song_ids"),
        ({"query": "one", "relevant_song_ids": "aaa"}, "relevant_song_ids"),
        ({"query": "one", "relevant_song_ids": [1]}, "not a string"),
        ({"query": "one", "relevant_song_ids": ["aaa"], "query_type": 7}, "`query_type`"),
    ],
)
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
    with pytest.raises(FileNotFoundError, match=re.escape("nope.jsonl")):
        load_eval_set(tmp_path / "nope.jsonl", KNOWN)


# --- stratum, and strict field validation (EVAL-AUTO §1, §3) ---


def test_stratum_is_loaded_and_defaults_to_none(tmp_path):
    path = _write(
        tmp_path,
        [
            {"query": "one", "relevant_song_ids": ["aaa"], "stratum": "translation"},
            {"query": "two", "relevant_song_ids": ["bbb"]},
        ],
    )
    loaded = load_eval_set(path, KNOWN)
    assert [q.stratum for q in loaded] == ["translation", None]


@pytest.mark.parametrize("stratum", ["main", "structure", "translation"])
def test_every_declared_stratum_is_accepted(tmp_path, stratum):
    path = _write(tmp_path, [{"query": "one", "relevant_song_ids": ["aaa"], "stratum": stratum}])
    assert load_eval_set(path, KNOWN)[0].stratum == stratum


def test_an_unrecognised_stratum_fails_rather_than_becoming_its_own_row(tmp_path):
    path = _write(tmp_path, [{"query": "one", "relevant_song_ids": ["aaa"], "stratum": "maim"}])
    with pytest.raises(ValueError, match="stratum"):
        load_eval_set(path, KNOWN)


def test_an_unknown_field_fails_and_names_itself(tmp_path):
    """A misspelled field must not be ignored.

    This is the whole point of the strict check: with `startum` silently
    dropped, every query loads with `stratum=None`, the run completes,
    and the headline table is computed over a sample that was supposed to
    be filtered. Nothing in the output would look wrong.
    """
    path = _write(tmp_path, [{"query": "one", "relevant_song_ids": ["aaa"], "startum": "main"}])
    with pytest.raises(ValueError, match="startum"):
        load_eval_set(path, KNOWN)


def test_unknown_fields_inside_meta_records_are_still_skipped(tmp_path):
    """`_meta` is documentation and may carry anything."""
    path = _write(
        tmp_path,
        [
            {"_meta": "notes", "generated_by": "x", "seed": 1},
            {"query": "one", "relevant_song_ids": ["aaa"]},
        ],
    )
    assert len(load_eval_set(path, KNOWN)) == 1


def test_the_bootstrap_golden_set_still_loads_without_strata(tmp_path):
    """The pre-stratum file must keep working; strictness is about
    unknown fields, not about newly-required ones."""
    ids = set()
    with open(GOLDEN, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if "_meta" not in record:
                ids.update(record["relevant_song_ids"])
    loaded = load_eval_set(GOLDEN, ids)
    assert loaded and all(q.stratum is None for q in loaded)
