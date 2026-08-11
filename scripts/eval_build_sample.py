"""EVAL-AUTO §1: draw the 500-song sample the known-item set is built from.

Writes one record per song to `data/eval/auto/sample.jsonl`, each carrying
its stratum and the fragment the `paraphrase` query will be generated
from. Counts go to the terminal, never the fragments -- those are
copyrighted lyrics.

## Why the sample is not proportional

Two of the three slices that matter are rare. A proportional draw of 500
songs yields roughly 15 songs whose chunks carry no structural markup and
2-3 translations. Neither `n` can show anything, so the sample
deliberately over-samples both: 400 songs stratified by genre, plus 50
structural and 50 translations.

The price is that the combined set describes no population, which is why
`stratum` is written into every record and the headline table is computed
on `main` alone (see `runners/eval.py --stratum`).

## Why assignment runs scarcest-first

The pools overlap: a translation may also have unmarked structure, and
either may belong to any genre. Assigning `main` first would consume
songs the scarce strata need -- the translation pool holds 139 songs
in this corpus, so losing even a few to an earlier draw measurably
shrinks what is left. Running `translation` -> `structure` -> `main`
means each stratum draws from a pool the later ones can still afford to
lose.

Strata are disjoint by construction, and the script asserts it rather
than trusting the argument: a song assigned twice would be scored twice
and would make the slice `n`s stop summing to the total.

## What is deliberately not here

The title and artist are not written into the record. The generator must
not see them -- titles are usually a phrase lifted from the hook, they
are indexed together with the lyrics, and a title word therefore hands
the lexical baseline an anchor while registering as zero overlap against
the fragment. Since the leak would be concentrated on songs whose titles
contain a distinctive word, it would survive regeneration and bias the
set in a fixed direction. They stay in the corpus, where the leak check
can score against them; they just never reach the generator.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")  # run from the repo root, as every other script here does

from lyrics_search.eval_set import STRATUM_MAIN, STRATUM_STRUCTURE, STRATUM_TRANSLATION
from lyrics_search.paths import corpus_dir, raw_path, work_dir

MIN_FRAGMENT_CHARS = 80
GENRES = ("country", "pop", "rap", "rock")
STRUCTURE_SPLITS = frozenset({"none", "plain_label"})


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _pick_fragment(chunks: list[dict], rng: random.Random) -> dict | None:
    """A random chunk over the length floor, never automatically the first.

    Songs open with a chorus often enough that taking position 0 would
    over-represent the most repeated and least distinctive text in the
    corpus, and would make the sample look homogeneous for a reason that
    has nothing to do with the songs.
    """
    usable = [c for c in chunks if len(c["text"]) >= MIN_FRAGMENT_CHARS]
    return rng.choice(usable) if usable else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--corpus", default="full")
    parser.add_argument("--chunking", default="sections")
    parser.add_argument("--per-genre", type=int, default=100)
    parser.add_argument("--structure", type=int, default=50)
    parser.add_argument("--translation", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--out", default="data/eval/auto/sample.jsonl")
    args = parser.parse_args()

    root = Path(args.data_root)
    genre_file = corpus_dir(root, args.corpus) / "genre.jsonl"
    chunks_file = work_dir(root, args.corpus, args.chunking) / "chunks.jsonl"
    raw_file = raw_path(root, args.corpus)
    for required in (genre_file, chunks_file, raw_file):
        if not required.exists():
            raise FileNotFoundError(f"missing corpus artifact: {required}")

    out_path = Path(args.out)
    if out_path.exists():
        raise FileExistsError(
            f"{out_path} already exists. Delete it deliberately or pass --out "
            f"elsewhere -- silently redrawing the sample would detach the "
            f"generated queries from the songs they were generated for."
        )

    genre = {r["song_id"]: r["genre"] for r in _read_jsonl(genre_file)}
    songs = {r["song_id"]: r for r in _read_jsonl(raw_file)}

    chunks_by_song: dict[str, list[dict]] = {}
    for chunk in _read_jsonl(chunks_file):
        chunks_by_song.setdefault(chunk["song_id"], []).append(chunk)

    # Only songs that survived into the index can be a known item: a song
    # with no chunks cannot be retrieved by any dense arm, so including it
    # would score as a miss for every arm and measure nothing.
    eligible = {sid for sid in songs if chunks_by_song.get(sid)}

    translation_pool = sorted(sid for sid in eligible if songs[sid]["is_translation"])
    # The condition is "all chunks share one value, and that value is
    # structural" -- not "every chunk is structural". `runners/eval.py`
    # labels a song by the `split_by` its chunks agree on and files any
    # disagreement under `mixed`, so a song mixing `none` with
    # `plain_label` never reaches the structural cell even though both of
    # its values qualify. Drawing such songs into the stratum would fill it
    # with rows that scatter into `mixed` and leave the cell as thin as
    # before. In this corpus that is 130 songs, 2540 against 2410.
    structure_pool = sorted(
        sid
        for sid in eligible
        if {c["split_by"] for c in chunks_by_song[sid]} <= STRUCTURE_SPLITS
        and len({c["split_by"] for c in chunks_by_song[sid]}) == 1
    )
    print(f"eligible songs        : {len(eligible)}")
    print(f"translation pool      : {len(translation_pool)}")
    print(f"structure pool        : {len(structure_pool)}")
    print(f"overlap of the two    : {len(set(translation_pool) & set(structure_pool))}")

    rng = random.Random(args.seed)
    assigned: dict[str, str] = {}

    def draw(pool: list[str], n: int, stratum: str) -> None:
        available = [sid for sid in pool if sid not in assigned]
        if len(available) < n:
            raise ValueError(
                f"stratum {stratum} needs {n} songs but only {len(available)} "
                f"are available after earlier strata took theirs. Sample sizes "
                f"are a design decision -- fix them, do not quietly take fewer."
            )
        for sid in rng.sample(available, n):
            assigned[sid] = stratum

    # Scarcest pool first. See the module docstring.
    draw(translation_pool, args.translation, STRATUM_TRANSLATION)
    draw(structure_pool, args.structure, STRATUM_STRUCTURE)
    for name in GENRES:
        pool = sorted(sid for sid in eligible if genre.get(sid) == name)
        draw(pool, args.per_genre, STRATUM_MAIN)

    expected = args.translation + args.structure + args.per_genre * len(GENRES)
    if len(assigned) != expected:
        raise AssertionError(f"strata overlap: {len(assigned)} songs assigned, expected {expected}")

    records = []
    skipped = 0
    for song_id in sorted(assigned):
        fragment = _pick_fragment(chunks_by_song[song_id], rng)
        if fragment is None:
            # Recorded, not dropped: a sample smaller than designed has to
            # be visible as a number rather than inferred from a count.
            records.append(
                {"song_id": song_id, "stratum": assigned[song_id], "skipped": "no_chunk_over_min"}
            )
            skipped += 1
            continue
        records.append(
            {
                "song_id": song_id,
                "stratum": assigned[song_id],
                "chunk_id": fragment["chunk_id"],
                "position": fragment["position"],
                "section": fragment["section"],
                "split_by": fragment["split_by"],
                "force_split": fragment["force_split"],
                "genre": genre.get(song_id),
                "is_translation": songs[song_id]["is_translation"],
                "text": fragment["text"],
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    usable = [r for r in records if "skipped" not in r]
    print(f"\nassigned              : {len(assigned)}")
    print(f"usable fragments      : {len(usable)}")
    print(f"skipped (short chunks): {skipped}")
    print(f"by stratum            : {dict(Counter(r['stratum'] for r in usable))}")
    print(
        f"main by genre         : {dict(Counter(r['genre'] for r in usable if r['stratum'] == STRATUM_MAIN))}"
    )
    print(f"translations in sample: {sum(1 for r in usable if r['is_translation'])}")
    print(f"fragment at position 0: {sum(1 for r in usable if r['position'] == 0)}")
    print(f"seed                  : {args.seed}")
    print(f"out                   : {out_path}")


if __name__ == "__main__":
    main()
