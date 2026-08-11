"""One-off: eyeball whether a theme has anything to find in the corpus.

    .venv/Scripts/python.exe scripts/probe_themes.py themes.txt

    # second opinion, before deleting the themes that found nothing:
    .venv/Scripts/python.exe scripts/probe_themes.py rejects.txt \
        --config configs/full_hybrid_numpy_tfidf.yaml --out data/probe_rejects_tfidf.txt

Input is a plain text file, one query per line; blank lines and lines
starting with `#` are skipped so a draft file can carry section headers.
Output is a plain text file under data/ (outside git, like every other
real-data artifact), written incrementally so a long run can be tailed and
a crash halfway through does not lose what already ran. An existing --out
is never overwritten without --overwrite: two runs under different configs
are meant to be read side by side, not to replace one another.

The second-opinion step is the one that matters for set construction. A
theme dropped because the bge-m3 hybrid found nothing may be a theme a
lexical arm would have won -- discarding it quietly biases the query set
toward what one system already does well. So only the rejects get
re-probed, and only before deletion. tfidf is the right second config
precisely because it is maximally unlike bge-m3: another neural embedder
would miss lexically in much the same places.

This is a drafting aid, not evaluation. It computes no metrics, scores
nothing, and writes no artifact any eval step reads. It exists to answer
one question per theme, before the query set is fixed: does the corpus
contain material this wording can reach at all?

Two deliberate omissions:

- Fusion scores are not printed. RRF scores carry no absolute meaning by
  construction (that is why RRF was chosen over weighted score summation
  -- see notes/decisions/retrieval-design.md), so displaying them would
  invite reading the output as a quality ranking. The judgement this
  output supports is binary: are a few genuinely relevant songs present
  in the top-N, or not.
- Only a one-to-two-line fragment of the matching chunk is shown, never
  the chunk in full and never the song. The corpus is copyrighted text;
  a fragment is enough to recognise why something matched.

The model is loaded once for the whole list, not per query -- a bge-m3
load costs ~19s against ~26ms per warm query, so per-query loading would
dominate the run entirely.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")  # run from the repo root, as every other script here does

from lyrics_search.config import load_config
from lyrics_search.contracts import RawSong, SearchResult
from lyrics_search.paths import raw_path
from lyrics_search.retrievers.loading import load_song_corpus
from lyrics_search.runners.search import build_retriever

DEFAULT_CONFIG = "configs/full_hybrid_faiss.yaml"
DEFAULT_OUT = "data/probe_themes.txt"
RULE = "=" * 78


def read_queries(path: Path | str) -> list[str]:
    """One query per line; blanks and `#` comments dropped.

    Raises rather than returning an empty list: a probe run over no
    queries would otherwise write a valid-looking empty report.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"missing queries file {path}")
    queries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            queries.append(line)
    if not queries:
        raise ValueError(f"{path} contains no queries (only blanks and # comments)")
    return queries


def snippet(text: str, *, max_lines: int = 2, max_chars: int = 90) -> str:
    """The first `max_lines` non-empty lines of `text`, each capped.

    Pure: the caller decides what to do with the result. A cut line ends
    in an ellipsis so a truncated fragment is never mistaken for a
    complete one.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()][:max_lines]
    capped = [
        ln if len(ln) <= max_chars else ln[: max_chars - 1].rstrip() + "\u2026" for ln in lines
    ]
    return " / ".join(capped)


def format_result(query: str, result: SearchResult, songs: dict[str, RawSong], top_n: int) -> str:
    """Render one query's top-`top_n` as a block of readable text."""
    lines = [RULE, f"QUERY: {query}", RULE]

    status = f"status={result.status}"
    if result.branch_statuses:
        branches = "  ".join(f"{k}={v}" for k, v in sorted(result.branch_statuses.items()))
        status += f"   branches: {branches}"
    lines.append(status)

    if result.status != "ok":
        lines.append("  -- no usable signal in this query, nothing was searched.")
    elif not result.hits:
        lines.append("  -- searched, but nothing matched.")

    for rank, hit in enumerate(result.hits[:top_n], start=1):
        song = songs.get(hit.song_id)
        if song is None:
            raise KeyError(
                f"song_id {hit.song_id} is in the index but not in raw.jsonl "
                f"-- the index and the corpus were built from different ingests."
            )
        lines.append(f"{rank:>4}. {song.artist} \u2014 {song.title}")
        lines.append(f"      {snippet(hit.best_chunk.text)}")

    lines.append("")
    return "\n".join(lines)


def probe(
    queries_path: Path | str,
    config_path: Path | str = DEFAULT_CONFIG,
    out_path: Path | str = DEFAULT_OUT,
    data_root: Path | str = "data",
    overwrite: bool = False,
) -> Path:
    queries = read_queries(queries_path)
    config = load_config(config_path)
    out_path = Path(out_path)
    if out_path.exists() and not overwrite:
        # These outputs are meant to be read side by side -- a second-opinion
        # run under a different config is compared against the first, not
        # substituted for it. Clobbering the earlier file by default would
        # destroy the comparison silently, and re-running costs minutes.
        raise FileExistsError(
            f"{out_path} already exists -- pass --out with a different path to keep "
            f"both runs for comparison, or --overwrite to replace this one."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[probe] {len(queries)} queries, loading {config.embedder.name} ...", file=sys.stderr)
    retriever, load_s = build_retriever(config, data_root=data_root)
    songs = load_song_corpus(raw_path(data_root, config.corpus))
    print(f"[probe] loaded in {load_s:.1f}s, {len(songs)} songs in corpus", file=sys.stderr)

    top_n = config.retrieval.return_n
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(
            f"# theme probe -- drafting aid, not evaluation. no metrics computed.\n"
            f"# config={config_path}  corpus={config.corpus}  embedder={config.embedder.name}\n"
            f"# index={config.index}  retrieval={config.retrieval.mode}  top_n={top_n}\n"
            f"# {len(queries)} queries from {queries_path}\n\n"
        )
        for i, query in enumerate(queries, start=1):
            t0 = time.time()
            result = retriever.search(query, config.retrieval.top_k)
            f.write(format_result(query, result, songs, top_n))
            f.flush()
            print(
                f"[probe] {i}/{len(queries)} {time.time() - t0:.2f}s  "
                f"status={result.status} hits={len(result.hits)}",
                file=sys.stderr,
            )

    print(f"[probe] wrote {out_path}", file=sys.stderr)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("queries", help="text file with one query per line")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--overwrite", action="store_true", help="replace --out if it already exists"
    )
    args = parser.parse_args()
    probe(args.queries, args.config, args.out, args.data_root, args.overwrite)


if __name__ == "__main__":
    main()
