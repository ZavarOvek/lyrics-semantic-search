"""Keyboard-layout folding for the manual-review interfaces.

Regression test for a bug that made `thematic_filter.py` look frozen: with a
Cyrillic layout active the physical `k` key sends `л`, which matched no
branch, so the tool redrew the same theme forever while behaving exactly as
written. Nothing crashed and nothing was logged.

`scripts/` is not a package, so it is put on the path the same way the
scripts put the repo root on it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from tui import LAYOUT, normalize_key


@pytest.mark.parametrize(
    ("pressed", "expected"),
    [
        ("л", "k"),  # keep
        ("в", "d"),  # drop
        ("г", "u"),  # undo
        ("й", "q"),  # quit
        ("у", "e"),  # expand
        (".", "/"),  # the ЙЦУКЕН position of the `/?` key
        (",", "?"),
    ],
)
def test_cyrillic_folds_to_the_key_at_the_same_position(pressed: str, expected: str) -> None:
    assert normalize_key(pressed) == expected


@pytest.mark.parametrize("key", ["k", "d", "u", "q", "e", "?", "/", "0", "1", "2"])
def test_latin_and_digits_pass_through(key: str) -> None:
    assert normalize_key(key) == key


def test_uppercase_is_folded_too() -> None:
    assert normalize_key("Л") == "k"
    assert normalize_key("K") == "k"


def test_empty_key_stays_empty() -> None:
    """`read_key` returns "" for the second unit of an arrow key."""
    assert normalize_key("") == ""


def test_layout_table_is_a_bijection() -> None:
    """A duplicate on either side would silently shadow a binding."""
    assert len(set(LAYOUT.values())) == len(LAYOUT)


def test_every_bound_key_is_reachable_from_a_cyrillic_layout() -> None:
    """The bindings both interfaces use must each have a Cyrillic source."""
    bound = {"k", "d", "u", "q", "e", "?", "/"}
    reachable = {normalize_key(ch) for ch in LAYOUT}
    assert bound <= reachable, f"unreachable on a Cyrillic layout: {sorted(bound - reachable)}"
