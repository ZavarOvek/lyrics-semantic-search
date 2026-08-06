# Project status

Entry point for picking this project back up. Per-phase detail lives in
`notes/reports/`; durable design rationale lives in `notes/decisions/`.

## Where things stand

Complete and published. The repository is public, the demo builds and
searches from a fresh clone with no flags, and the full 30,000-song corpus
has been built and measured end to end.

What exists:

- **Offline pipeline** — `ingest → preprocess → embed → index`, with
  stage-level caching on chained keys and `--force` to bypass.
- **Online search** — dense, lexical and hybrid retrieval. All state is
  loaded at query time, never fitted or rebuilt.
- **Four embedders** behind one protocol, selected from config: `bge-m3`,
  `e5-base`, `fasttext-avg`, `tfidf`.
- **Two index backends** — `numpy` (exact) and `faiss` (HNSW).
- **176 tests**, `core/` at 100% line coverage. With the base
  `requirements.txt` 172 pass and 4 skip: two need `faiss-cpu` from
  `requirements-full.txt`, two need real corpus data that is deliberately
  not in git.

Measured on the full corpus: 30,000 songs produce 208,411 chunks, of which
181,471 survive preprocessing. Embedding takes about 14 minutes on an
RTX 3050 Laptop and yields 355 MB of vectors. Repeat queries run at
55.6 ms. Full numbers and caveats are in the phase reports.

## How it got here

Development ran in phases, each closed by its own report. The reports are
append-only: they record the state at the time of writing, including
figures later superseded, because the corrections are part of the record.

| Phase | Delivered | Report |
|---|---|---|
| Phase 0 | Vertical-slice script; first look at real data | `reports/phase0-findings.md` |
| Offline pipeline | Package structure, four-level chunking cascade, four embedders, both index backends; dev and full corpora built for real | `reports/spec02-report.md` |
| Offline pipeline patch | Language-filter decision, song-filter proof, fitted-state persistence, zero-signal filtering, `split_by`/`force_split` split | `reports/spec02-patch-report.md` |
| Online branch | Config and registry, dense/lexical/hybrid retrieval, stage caching; faiss non-ASCII-path bug found and fixed | `reports/spec03-report.md` |
| Online branch patch | Parameterized embedder registry, environment verification, real-corpus cache verification, comparison claims softened to what the sample supports | `reports/spec03-patch-report.md` |
| Tests and release | Property tests, `core/` coverage, synthetic demo corpus, bilingual README, dependency split, publication | `reports/spec04-report.md` |

## What's next

The eval phase has not been started. Its groundwork is in place and its
central choice is deliberately left open.

Already prepared for it:

- `core/` metrics slots and the eval-set format convention, including the
  `_meta` sentinel record that loaders skip.
- Slice dimensions carried on the data itself: `split_by` × `force_split`
  (does retrieval degrade where songs had no structural markup?),
  `is_translation`, and genre from the source metadata.
- `aggregate` tested against the degenerate one-chunk-per-song case, which
  makes a whole-song-as-one-vector arm viable — `bge-m3` handles 8192
  tokens, so it is the only embedder here that can hold an entire song in
  one vector.

Two things must not be carried forward:

1. **The bootstrap eval set in `tests/golden/spec03_eval_queries.jsonl` is
   not a benchmark.** Its queries are song titles, which typically recur
   verbatim in the lyrics, so it systematically favours lexical retrieval.
   The file carries this warning in its own `_meta` record.
2. **The comparison figures in `reports/spec03-patch-report.md` are a floor
   check, not a result.** At n=20 the Wilson intervals for all three
   retrievers overlap heavily; no ranking among them is supportable.

Still outstanding regardless of query design: `e5-base`, `tfidf` and
`fasttext-avg` have only been run at dev scale. `tfidf` in particular needs
a storage decision before a full-corpus run — its vectors are dense in the
current `Embedder` protocol, and at 181k chunks with a full-corpus
vocabulary that is tens of gigabytes.

## Where to look

- **`notes/decisions/`** — design rationale that outlives any single phase.
  `retrieval-design.md` (RRF fusion, the dense/lexical granularity split,
  max-not-mean aggregation, translations flagged rather than filtered,
  cross-song duplicates left separate), `lang-check-decision.md` (a
  language-confidence filter evaluated with real calibration data and cut),
  `faiss-nonascii-path-windows.md` (a Windows file-I/O trap in faiss and
  the way around it).
- **`notes/reports/`** — one file per phase, chronological, with small
  companion findings merged in as appendices.
- **`scripts/`** — reusable one-off and benchmark scripts that need real,
  non-git data and produce output meant for a human to read rather than for
  CI: `spec03_latency_bench.py`, `spec03_retriever_comparison.py`,
  `full_index_sanity_check.py`, `dedup_check_full.py`.
- **`tests/`** — the pytest suite. `test_fitted_state_persistence.py`
  verifies that the two fit-dependent embedders reproduce identical vectors
  across separate processes; it skips cleanly without real corpus data.

These notes are English-only, unlike the bilingual README — they are
working detail rather than an introduction to the project.
