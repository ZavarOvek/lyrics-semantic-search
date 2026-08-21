"""`thematic_pool.load_themes`'s exclusion rule.

`keep: null` is ambiguous on its own: a theme that has never been reviewed
and a theme the second filtering pass left `unsure` both carry it. Only
`mark` tells them apart, and the two must be treated oppositely -- an
unreviewed theme stays in the pool so a half-finished filter cannot
silently shrink it, but a theme explicitly left undecided is a verdict and
must not be graded as if the human meant to include it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from thematic_pool import load_themes


def write_themes(path: Path, themes: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"_meta": {"source": "test"}}, ensure_ascii=False) + "\n")
        for theme in themes:
            f.write(json.dumps(theme, ensure_ascii=False) + "\n")


def theme(theme_id: str, **extra) -> dict:
    base = {"theme_id": theme_id, "text": theme_id, "tier": "short"}
    base.update(extra)
    return base


def test_never_reviewed_is_kept(tmp_path: Path) -> None:
    """No `mark` at all: filtering has not run yet, so nothing is excluded."""
    path = tmp_path / "themes.jsonl"
    write_themes(path, [theme("t01", keep=None)])
    assert [t["theme_id"] for t in load_themes(path)] == ["t01"]


def test_keep_false_is_excluded(tmp_path: Path) -> None:
    path = tmp_path / "themes.jsonl"
    write_themes(path, [theme("t01", mark="drop", keep=False)])
    with pytest.raises(SystemExit, match="yielded no themes"):
        load_themes(path)


def test_mark_unsure_is_excluded_even_though_keep_is_null(tmp_path: Path) -> None:
    """The case this rule exists for: reviewed twice, still undecided."""
    path = tmp_path / "themes.jsonl"
    write_themes(path, [theme("t01", mark="unsure", keep=None)])
    with pytest.raises(SystemExit, match="yielded no themes"):
        load_themes(path)


def test_mark_keep_is_included(tmp_path: Path) -> None:
    path = tmp_path / "themes.jsonl"
    write_themes(path, [theme("t01", mark="keep", keep=True)])
    assert [t["theme_id"] for t in load_themes(path)] == ["t01"]


def test_a_filtered_mix_keeps_exactly_the_keeps(tmp_path: Path) -> None:
    path = tmp_path / "themes.jsonl"
    write_themes(
        path,
        [
            theme("t01", mark="keep", keep=True),
            theme("t02", mark="drop", keep=False),
            theme("t03", mark="unsure", keep=None),
            theme("t04", mark="keep", keep=True),
        ],
    )
    assert [t["theme_id"] for t in load_themes(path)] == ["t01", "t04"]
