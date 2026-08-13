"""EVAL-THEMATIC §4: the theme-filtering interface for the 70 draft themes.

    ./.venv/Scripts/python.exe scripts/thematic_filter.py

Same shape as `thematic_label.py` -- one item per screen, one keypress, no
mouse -- but this is corpus selection, not relevance judgement, so two things
differ deliberately:

  * Themes come in file order and are never shuffled. Blinding exists to stop
    the arm identity leaking into a judgement; nothing is being judged here,
    and a stable order is what makes the sitting resumable and reviewable.
  * The decision is written back into the themes file itself rather than to a
    separate log, because `keep` is an attribute of the theme.

Keys:

    k   keep
    d   deletion candidate
    ?   unsure -- goes to the second pass together with the d's
    u   undo the last decision
    q   save and quit

## Three states, and why `keep` alone cannot hold them

`keep: null` already means "not yet reviewed" -- `thematic_pool.py` relies on
it, and treats an unreviewed theme as still in play so that a half-finished
filter cannot silently shrink the pool. So "unsure" cannot also be `null`:
resume would offer the same theme forever, and the second pass would have no
way to find it.

Each theme therefore carries `mark` in {keep, drop, unsure}, and `keep` is
derived from it:

    mark=keep    -> keep=true     survives
    mark=drop    -> keep=false    dropped, `thematic_pool.py` skips it
    mark=unsure  -> keep=null     still in play, revisited in pass two

Presence of `mark` is what "reviewed" means. `keep` stays exactly as
`thematic_pool.py` already reads it, so nothing downstream changes.

## The second pass

§4 wants deletion candidates re-probed through a different retriever before
they are cut. That is a fresh probe run plus another sitting, not a second
tool:

    ./.venv/Scripts/python.exe scripts/probe_themes.py \
        --config configs/full_hybrid_numpy_tfidf.yaml --out data/probe_tfidf.txt
    ./.venv/Scripts/python.exe scripts/thematic_filter.py \
        --only flagged --probe data/probe_tfidf.txt

`--only flagged` revisits exactly the d's and the ?'s, showing the previous
mark, so the second pass is confirmation rather than a blank re-decision.

## Persistence

Every keypress rewrites the themes file through a temp file and `os.replace`.
Undo removes the highest `mark_seq`, a counter rather than a timestamp
because keypresses come faster than the one-second resolution of a clock and
ties would make undo pick an arbitrary theme.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, ".")  # run from the repo root, as every other script here does

MARKS = {"k": "keep", "d": "drop", "?": "unsure", "/": "unsure"}
KEEP_FROM_MARK = {"keep": True, "drop": False, "unsure": None}
TIERS = ("short", "medium", "expanded")
# §2's "50 themes" sized the labelling effort; it was never a quota. However
# many themes survive the filter is the number of themes, and the count that
# actually matters is downstream: how many judgements the survivors imply.
# Displaying a fixed goal instead would invite trimming a good theme, or
# keeping a weak one, to hit a round number.
POOLED_PER_THEME = 21.3  # measured on the 3-theme smoke pool; indicative only
CLEAR = "\x1b[2J\x1b[H"
HIT_RE = re.compile(r"^ {0,4}(\d+)\. (.*)$")


def read_key() -> str:
    """One keypress, no Enter. Imported lazily so this module still loads
    on a machine without the other platform's terminal library."""
    if os.name == "nt":
        import msvcrt

        ch = msvcrt.getch()
        if ch in (b"\x00", b"\xe0"):  # function/arrow key: consume the second byte
            msvcrt.getch()
            return ""
        if ch == b"\x03":
            raise KeyboardInterrupt
        return ch.decode("utf-8", errors="replace").lower()

    import termios
    import tty

    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
    if ch == "\x03":
        raise KeyboardInterrupt
    return ch.lower()


def load_themes_file(path: Path | str) -> tuple[dict, list[dict]]:
    """The `_meta` record and the themes, kept apart so a rewrite can put
    `_meta` back exactly where it was."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"missing themes file {path}")
    meta: dict = {}
    themes: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "_meta" in record:
                meta = record
            else:
                themes.append(record)
    if not themes:
        raise SystemExit(f"{path} contains no themes")
    return meta, themes


def save_themes_file(path: Path | str, meta: dict, themes: list[dict]) -> None:
    """Atomic rewrite. This file is the only record of the filtering pass."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        if meta:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        for theme in themes:
            f.write(json.dumps(theme, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def parse_probe(path: Path | str) -> dict[str, list[tuple[str, str]]]:
    """Probe output -> {query text: [(artist - title, fragment), ...]}.

    Written against the format `scripts/probe_themes.py` emits: a `QUERY:`
    header, then right-aligned `N. Artist - Title` lines each followed by an
    indented fragment.
    """
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"missing probe output {path} -- run scripts/probe_themes.py first")
    hits: dict[str, list[tuple[str, str]]] = {}
    current: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("QUERY: "):
            current = line[len("QUERY: ") :].strip()
            hits[current] = []
            continue
        if current is None:
            continue
        match = HIT_RE.match(line)
        if match:
            hits[current].append((match.group(2).strip(), ""))
        elif line.startswith("      ") and hits[current]:
            title, fragment = hits[current][-1]
            joined = f"{fragment} {line.strip()}".strip()
            hits[current][-1] = (title, joined)
    return hits


def select(themes: list[dict], only: str) -> list[dict]:
    if only == "all":
        return list(themes)
    if only == "flagged":
        return [t for t in themes if t.get("mark") in {"drop", "unsure"}]
    return [t for t in themes if "mark" not in t]


def tally(themes: list[dict]) -> dict[str, int]:
    counts = {"keep": 0, "drop": 0, "unsure": 0}
    for theme in themes:
        mark = theme.get("mark")
        if mark in counts:
            counts[mark] += 1
    return counts


def tier_line(themes: list[dict]) -> str:
    """Kept-so-far against the tier's total size.

    Shown live because the tiers are not the same size -- `expanded` is the
    thinnest -- and a filter that cuts them proportionally leaves the smallest
    slice too thin to compare arms within. Steering that is only possible
    while filtering, not after.
    """
    parts = []
    for tier in TIERS:
        total = sum(1 for t in themes if t.get("tier") == tier)
        kept = sum(1 for t in themes if t.get("tier") == tier and t.get("mark") == "keep")
        parts.append(f"{tier} {kept}/{total}")
    return "   ".join(parts)


def truncate(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1].rstrip() + "\u2026"


def render(
    theme: dict,
    hits: list[tuple[str, str]],
    *,
    position: int,
    total: int,
    themes: list[dict],
    width: int,
    missing_probe: bool,
) -> str:
    counts = tally(themes)
    reviewed = sum(counts.values())
    head = (
        f"theme {position}/{total}   reviewed {reviewed}/{len(themes)}   "
        f"keep {counts['keep']}  drop {counts['drop']}  unsure {counts['unsure']}"
        f"   \u2248 {round(counts['keep'] * POOLED_PER_THEME):,} judgements"
    )
    rule = "=" * width
    thin = "-" * width
    previous = f"   [previously: {theme['mark']}]" if theme.get("mark") else ""
    words = theme.get("words", len(theme["text"].split()))
    body = [
        f"{CLEAR}{head}",
        f"kept by tier:  {tier_line(themes)}",
        rule,
        truncate(
            f"THEME:  {theme['text']}   ({theme.get('tier', '?')}, {words}w){previous}", width
        ),
        rule,
    ]
    if missing_probe:
        body.append("(no probe output for this theme -- deciding on the wording alone)")
    else:
        for i, (title, fragment) in enumerate(hits, 1):
            body.append(truncate(f"{i:>4}. {title}", width))
            if fragment:
                body.append(truncate(f"      {fragment}", width))
    body += [
        thin,
        "[k] keep   [d] drop candidate   [?] unsure   [u] undo   [q] save and quit",
    ]
    return "\n".join(body) + "\n"


def run(themes_path: str, probe_path: str, only: str) -> None:
    meta, themes = load_themes_file(themes_path)
    probe = parse_probe(probe_path)
    if not any(t["text"] in probe for t in themes):
        raise SystemExit(
            f"{probe_path} has no output for any of the {len(themes)} themes -- "
            f"it was probably generated from a different theme list"
        )

    seq = max((t.get("mark_seq", 0) for t in themes), default=0)
    order = {id(t): i for i, t in enumerate(themes, 1)}
    width = min(shutil.get_terminal_size((80, 24)).columns - 1, 100)

    while True:
        pending = select(themes, only)
        if not pending:
            break
        theme = pending[0]
        print(
            render(
                theme,
                probe.get(theme["text"], []),
                position=order[id(theme)],
                total=len(themes),
                themes=themes,
                width=width,
                missing_probe=theme["text"] not in probe,
            ),
            end="",
            flush=True,
        )

        key = read_key()
        if key == "q":
            break
        if key == "u":
            marked = [t for t in themes if "mark_seq" in t]
            if marked:
                last = max(marked, key=lambda t: t["mark_seq"])
                last.pop("mark", None)
                last.pop("mark_seq", None)
                last["keep"] = None
                seq = max((t.get("mark_seq", 0) for t in themes), default=0)
                save_themes_file(themes_path, meta, themes)
            continue
        if key in MARKS:
            seq += 1
            theme["mark"] = MARKS[key]
            theme["keep"] = KEEP_FROM_MARK[MARKS[key]]
            theme["mark_seq"] = seq
            save_themes_file(themes_path, meta, themes)

    save_themes_file(themes_path, meta, themes)
    counts = tally(themes)
    reviewed = sum(counts.values())
    print(f"{CLEAR}saved {themes_path}")
    print(f"  reviewed: {reviewed}/{len(themes)}")
    for mark in ("keep", "drop", "unsure"):
        print(f"  {mark}: {counts[mark]}")
    print(f"  kept by tier:  {tier_line(themes)}")
    print(
        f"  implies \u2248 {round(counts['keep'] * POOLED_PER_THEME):,} judgements "
        f"at ~{POOLED_PER_THEME} pooled candidates per theme"
    )
    if counts["unsure"] or counts["drop"]:
        print(f"  second pass: {counts['drop'] + counts['unsure']} themes, --only flagged")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--themes", default="data/eval/thematic/themes-draft70.jsonl")
    parser.add_argument("--probe", default="data/probe_themes.txt")
    parser.add_argument(
        "--only",
        default="new",
        choices=("new", "flagged", "all"),
        help="new: unreviewed only (default). flagged: the d's and ?'s, for pass two. all: everything.",
    )
    args = parser.parse_args()
    run(args.themes, args.probe, args.only)


if __name__ == "__main__":
    main()
