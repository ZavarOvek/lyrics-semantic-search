"""EVAL-THEMATIC §5: the manual relevance-labelling interface.

    ./.venv/Scripts/python.exe scripts/thematic_label.py \
        --pool data/eval/thematic/pool.jsonl \
        --judgements data/eval/thematic/judgements.jsonl

One song per screen, three grades, single keypress, no mouse. Roughly
1100-1750 judgements get made through this, so anything that costs an
extra keystroke costs it a thousand times over.

Keys:

    2   the song really is about this
    1   tangential -- the theme is present but not the point
    0   not about this
    ?   don't know (kept out of the metrics, counted in the report)
    e   expand / collapse the lyrics
    u   undo the last judgement
    q   save and quit

## What this program deliberately cannot do

It reads `pool.jsonl` and `raw.jsonl`. It does not read
`pool_provenance.jsonl`, and it never opens any file containing model
judgements.

That is not a policy, it is the reason the two are separate files. §6
requires that the assessor not see the model's grade before setting their
own, because a visible suggestion anchors the hand on exactly the
ambiguous cases the agreement measurement is made of -- what would be
recorded then is not agreement but influence. §8 requires the same for the
arm name and the rank. Neither value is in any structure this program
loads, so there is no field to render and no flag to get wrong.

Candidate order is likewise decided by `thematic_pool.py` and arrives as a
plain list. Warm-up themes come in natural order and are marked as such;
the rest are shuffled. This program does not know which is which beyond
printing the marker, and cannot recover a rank from the list it is given.

## Persistence

Every judgement rewrites `judgements.jsonl` through a temp file and
`os.replace`, so an interrupted write cannot leave a truncated record. A
thousand judgements is not one sitting: quitting at any point and
restarting resumes at the first unjudged candidate. Undo pops the last
record, which is what makes a mistyped key recoverable rather than
permanent.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")  # run from the repo root, as every other script here does

from lyrics_search.contracts import RawSong
from lyrics_search.paths import raw_path
from lyrics_search.retrievers.loading import load_song_corpus

GRADES = {"0": 0, "1": 1, "2": 2}
DONT_KNOW = {"?", "/"}
COLLAPSED_LINES = 8
CLEAR = "\x1b[2J\x1b[H"


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


def load_pool(path: Path | str) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"missing pool {path} -- run scripts/thematic_pool.py first")
    themes = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "_meta" not in record:
                themes.append(record)
    if not themes:
        raise SystemExit(f"{path} contains no themes")
    return themes


def load_judgements(path: Path | str) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_judgements(path: Path | str, records: list[dict]) -> None:
    """Atomic rewrite. Small file, and a partial write here would corrupt
    hours of manual work."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def lyrics_block(song: RawSong, *, expanded: bool) -> str:
    lines = [ln.rstrip() for ln in song.text_raw.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    if expanded:
        return "\n".join(lines)
    shown = lines[:COLLAPSED_LINES]
    hidden = len(lines) - len(shown)
    body = "\n".join(shown)
    if hidden > 0:
        body += f"\n\n    ... {hidden} more lines -- press [e] to expand"
    return body


def render(
    theme: dict,
    song: RawSong,
    *,
    expanded: bool,
    theme_no: int,
    theme_total: int,
    cand_no: int,
    cand_total: int,
    done: int,
    overall: int,
) -> str:
    warm = "  [warm-up: natural order]" if theme.get("warmup") else ""
    head = (
        f"theme {theme_no}/{theme_total}   candidate {cand_no}/{cand_total}   "
        f"overall {done}/{overall}{warm}"
    )
    return (
        f"{CLEAR}{head}\n"
        f"{'=' * 78}\n"
        f"THEME:  {theme['text']}   ({theme['tier']})\n"
        f"{'=' * 78}\n"
        f"{song.artist} \u2014 {song.title}\n"
        f"{'-' * 78}\n"
        f"{lyrics_block(song, expanded=expanded)}\n"
        f"{'-' * 78}\n"
        f"[2] about this   [1] tangential   [0] not about this   "
        f"[?] don't know\n"
        f"[e] expand       [u] undo         [q] save and quit\n"
    )


def run(pool_path: str, judgements_path: str, data_root: str, corpus: str) -> None:
    themes = load_pool(pool_path)
    judgements = load_judgements(judgements_path)
    songs = load_song_corpus(raw_path(data_root, corpus))

    queue = [(t, song_id) for t in themes for song_id in t["candidates"]]
    overall = len(queue)
    if not overall:
        raise SystemExit("pool is empty -- nothing to judge")

    judged = {(r["theme_id"], r["song_id"]) for r in judgements}
    theme_index = {t["theme_id"]: i for i, t in enumerate(themes, 1)}
    expanded = False

    while True:
        pending = [(t, s) for (t, s) in queue if (t["theme_id"], s) not in judged]
        if not pending:
            break
        theme, song_id = pending[0]
        song = songs.get(song_id)
        if song is None:
            raise SystemExit(
                f"song_id {song_id} is in the pool but not in raw.jsonl -- "
                f"the pool and the corpus came from different ingests"
            )
        cand_total = len(theme["candidates"])
        cand_no = theme["candidates"].index(song_id) + 1
        print(
            render(
                theme,
                song,
                expanded=expanded,
                theme_no=theme_index[theme["theme_id"]],
                theme_total=len(themes),
                cand_no=cand_no,
                cand_total=cand_total,
                done=len(judged),
                overall=overall,
            ),
            end="",
            flush=True,
        )

        key = read_key()
        if key == "q":
            break
        if key == "e":
            expanded = not expanded
            continue
        if key == "u":
            if judgements:
                last = judgements.pop()
                judged.discard((last["theme_id"], last["song_id"]))
                save_judgements(judgements_path, judgements)
            expanded = False
            continue
        if key in GRADES or key in DONT_KNOW:
            record = {
                "theme_id": theme["theme_id"],
                "song_id": song_id,
                "grade": GRADES.get(key),
                "dont_know": key in DONT_KNOW,
                "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            judgements.append(record)
            judged.add((theme["theme_id"], song_id))
            save_judgements(judgements_path, judgements)
            expanded = False

    save_judgements(judgements_path, judgements)
    dk = sum(1 for r in judgements if r["dont_know"])
    print(f"{CLEAR}saved {len(judgements)}/{overall} judgements to {judgements_path}")
    print(f"  don't know: {dk}")
    for grade in (2, 1, 0):
        print(f"  grade {grade}: {sum(1 for r in judgements if r['grade'] == grade)}")
    remaining = overall - len(judgements)
    print(f"  remaining: {remaining}" if remaining else "  complete")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pool", default="data/eval/thematic/pool.jsonl")
    parser.add_argument("--judgements", default="data/eval/thematic/judgements.jsonl")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--corpus", default="full")
    args = parser.parse_args()
    run(args.pool, args.judgements, args.data_root, args.corpus)


if __name__ == "__main__":
    main()
