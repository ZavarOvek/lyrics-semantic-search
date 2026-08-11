"""EVAL-PREP §5: the eval-set format and its loader.

One JSONL record per query:

    {"query": "...", "relevant_song_ids": ["..."], "query_type": "...",
     "stratum": "..."}

`query_type` and `stratum` are both optional so the pre-existing
bootstrap file (`tests/golden/spec03_eval_queries.jsonl`, which has
neither) still loads; the real eval set carries both, since they are what
the per-type and per-stratum breakdowns slice on. Nothing here decides
*what* the types or strata are -- the query scheme is designed separately
(EVAL-PREP "What not to do"), and this loader is deliberately agnostic to
it.

`stratum` exists because the automatic track over-samples on purpose
(EVAL-AUTO §1): 400 songs stratified by genre, plus 50 chosen for their
chunk structure and 50 translations, both of which are far rarer than
that in the population. The headline table is computed on `main` alone,
and the over-sampled strata are reported beside it. Mixing them would
report a number that describes no population -- so which stratum a query
belongs to has to survive into the loaded object, not just into the file.

Three rules, all about not silently miscounting:

`_meta` records are skipped. The convention already exists in the
bootstrap file, whose leading record carries a warning that its queries
are song titles and therefore favour lexical retrieval. Skipping it is
what keeps that warning from being scored as a query with no relevant
songs -- i.e. as a guaranteed zero dragging every mean down.

Unknown `song_id`s fail loudly, naming the offending id and the line it
came from. The failure mode this exists to prevent is silent and
directional: an id that is not in the corpus can never be retrieved, so
it scores as a miss, and a typo or a stale eval set therefore looks
exactly like a retriever that is bad at that query. Recall in particular
would be understated by a fixed factor with nothing in the output
suggesting why.

Unknown *fields* fail loudly too, for the same reason one level up. A
loader that ignores what it does not recognise turns a misspelled
`startum` into a file where every query silently belongs to no stratum;
the run then completes, prints a table, and the table is wrong in a way
nothing in it reveals. Rejecting the field costs one clear error message
and makes that outcome impossible.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

META_KEY = "_meta"

STRATUM_MAIN = "main"
STRATUM_STRUCTURE = "structure"
STRATUM_TRANSLATION = "translation"
STRATA = (STRATUM_MAIN, STRATUM_STRUCTURE, STRATUM_TRANSLATION)

KNOWN_FIELDS = frozenset({"query", "relevant_song_ids", "query_type", "stratum"})


@dataclass(frozen=True)
class EvalQuery:
    query: str
    relevant_song_ids: tuple[str, ...]
    query_type: str | None = None
    stratum: str | None = None


def _fail(path: Path, lineno: int, message: str) -> None:
    raise ValueError(f"{path}:{lineno}: {message}")


def load_eval_set(path: Path | str, known_song_ids: Iterable[str]) -> list[EvalQuery]:
    """Load and validate an eval set.

    `known_song_ids` is required rather than optional: an optional corpus
    would make the id check skippable, and a skipped check here is
    indistinguishable in the output from a retriever that missed. Callers
    pass the song_ids of the corpus the run is about to be scored against.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"missing eval set {path}")
    known = set(known_song_ids)

    queries: list[EvalQuery] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                _fail(path, lineno, f"not valid JSON -- {exc}")
            if not isinstance(record, dict):
                _fail(path, lineno, f"expected a JSON object, got {type(record).__name__}")
            if META_KEY in record:
                continue

            unknown = sorted(set(record) - KNOWN_FIELDS)
            if unknown:
                _fail(
                    path,
                    lineno,
                    f"unknown field(s) {unknown} -- known fields are "
                    f"{sorted(KNOWN_FIELDS)}. Ignoring an unrecognised field "
                    f"would let a misspelling produce a table that is wrong "
                    f"without looking wrong.",
                )

            query = record.get("query")
            if not isinstance(query, str) or not query.strip():
                _fail(path, lineno, f"`query` must be a non-empty string, got {query!r}")

            ids = record.get("relevant_song_ids")
            if not isinstance(ids, list) or not ids:
                _fail(path, lineno, f"`relevant_song_ids` must be a non-empty list, got {ids!r}")
            for song_id in ids:
                if not isinstance(song_id, str):
                    _fail(path, lineno, f"relevant_song_ids entry {song_id!r} is not a string")
                if song_id not in known:
                    _fail(
                        path,
                        lineno,
                        f"relevant_song_id {song_id!r} is not in the corpus "
                        f"({len(known)} songs) -- it can never be retrieved, so "
                        f"leaving it in would score as a retrieval miss.",
                    )

            query_type = record.get("query_type")
            if query_type is not None and not isinstance(query_type, str):
                _fail(path, lineno, f"`query_type` must be a string or absent, got {query_type!r}")

            # Constrained to the known strata, unlike `query_type`. The
            # strata are fixed by the sampling design and each one has a
            # defined relationship to the population; an unrecognised value
            # is a sampling bug, and silently making it its own row would
            # split a stratum in two and halve both `n`s.
            stratum = record.get("stratum")
            if stratum is not None and stratum not in STRATA:
                _fail(
                    path,
                    lineno,
                    f"`stratum` must be one of {list(STRATA)} or absent, got {stratum!r}",
                )

            queries.append(
                EvalQuery(
                    query=query,
                    relevant_song_ids=tuple(dict.fromkeys(ids)),  # dedupe, keep order
                    query_type=query_type,
                    stratum=stratum,
                )
            )

    if not queries:
        raise ValueError(f"{path}: contains no queries (only {META_KEY} records or blanks)")
    return queries
