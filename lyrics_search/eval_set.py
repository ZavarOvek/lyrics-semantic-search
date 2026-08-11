"""EVAL-PREP §5: the eval-set format and its loader.

One JSONL record per query:

    {"query": "...", "relevant_song_ids": ["..."], "query_type": "..."}

`query_type` is optional so the pre-existing bootstrap file
(`tests/golden/spec03_eval_queries.jsonl`, which has none) still loads;
the real eval set is expected to carry it, since it is what item 6's
per-type breakdown slices on. Nothing here decides *what* the types are
-- the query scheme is designed separately (EVAL-PREP "What not to do"),
and this loader is deliberately agnostic to it.

Two rules, both about not silently miscounting:

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
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

META_KEY = "_meta"


@dataclass(frozen=True)
class EvalQuery:
    query: str
    relevant_song_ids: tuple[str, ...]
    query_type: str | None = None


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

            queries.append(
                EvalQuery(
                    query=query,
                    relevant_song_ids=tuple(dict.fromkeys(ids)),  # dedupe, keep order
                    query_type=query_type,
                )
            )

    if not queries:
        raise ValueError(f"{path}: contains no queries (only {META_KEY} records or blanks)")
    return queries
