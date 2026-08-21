"""The recheck queue that drives `thematic_filter.py --only flagged`.

Regression guard for a defect that only ever showed up in the second pass:
selection used to read the live `mark`, so re-marking a theme `d` or `?` left
it flagged, kept it at the head of the pending list, and redrew the same
screen forever. Only `k` advanced. Nothing in the first pass touched that
path, so a smoke test of pass one passed while pass two was unusable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from thematic_filter import KEEP_FROM_MARK, select, stamp_recheck_queue


def themes() -> list[dict]:
    return [
        {"theme_id": "t01", "text": "a", "mark": "keep", "keep": True},
        {"theme_id": "t02", "text": "b", "mark": "drop", "keep": False},
        {"theme_id": "t03", "text": "c", "mark": "unsure", "keep": None},
        {"theme_id": "t04", "text": "d", "keep": None},
    ]


def ids(selected: list[dict]) -> list[str]:
    return [t["theme_id"] for t in selected]


def test_stamp_marks_the_drops_and_the_unsures_only() -> None:
    ts = themes()
    assert stamp_recheck_queue(ts) == 2
    assert ids(select(ts, "flagged")) == ["t02", "t03"]


def test_stamp_copies_the_pass_one_verdict_aside() -> None:
    ts = themes()
    stamp_recheck_queue(ts)
    assert [t.get("mark_before_recheck") for t in ts] == [None, "drop", "unsure", None]


def test_stamp_is_idempotent_while_a_queue_is_outstanding() -> None:
    """An interrupted second pass must resume, not start over."""
    ts = themes()
    stamp_recheck_queue(ts)
    ts[1]["mark"] = "keep"  # t02 re-decided
    ts[1].pop("recheck")
    assert stamp_recheck_queue(ts) == 0
    assert ids(select(ts, "flagged")) == ["t03"]


@pytest.mark.parametrize("remark", ["drop", "unsure", "keep"])
def test_a_re_decided_theme_leaves_the_queue_whatever_it_was_marked(remark: str) -> None:
    """The defect: `drop` and `unsure` used to keep the theme pending."""
    ts = themes()
    stamp_recheck_queue(ts)
    theme = select(ts, "flagged")[0]
    theme["mark"] = remark
    theme["keep"] = KEEP_FROM_MARK[remark]
    theme.pop("recheck", None)
    assert theme["theme_id"] not in ids(select(ts, "flagged"))


def test_queue_empties_after_every_flagged_theme_is_re_decided() -> None:
    ts = themes()
    stamp_recheck_queue(ts)
    while pending := select(ts, "flagged"):
        pending[0]["mark"] = "drop"
        pending[0].pop("recheck")
    assert select(ts, "flagged") == []


def test_a_further_run_stamps_a_third_pass_over_what_is_still_flagged() -> None:
    ts = themes()
    stamp_recheck_queue(ts)
    for t in ts:
        t.pop("recheck", None)
    ts[1]["mark"] = "keep"  # t02 survived pass two, t03 is still unsure
    assert stamp_recheck_queue(ts) == 1
    assert ids(select(ts, "flagged")) == ["t03"]


def test_new_selects_the_unreviewed_and_flagged_does_not() -> None:
    ts = themes()
    assert ids(select(ts, "new")) == ["t04"]
    stamp_recheck_queue(ts)
    assert "t04" not in ids(select(ts, "flagged"))


def test_all_selects_everything_regardless_of_stamps() -> None:
    ts = themes()
    stamp_recheck_queue(ts)
    assert ids(select(ts, "all")) == ["t01", "t02", "t03", "t04"]
