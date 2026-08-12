# Project status

Entry point for picking this project back up. Per-phase detail lives in
`notes/reports/`; durable design rationale lives in `notes/decisions/`.

## Where things stand

The code base is complete and the repository is public. The demo builds and
searches from a fresh clone with no flags, and the full 30,000-song corpus
has been built and measured end to end with all four embedders.

The active phase is **eval**. Its automatic, known-item track is complete:
985 gated queries over 498 songs, all nine arms run, full analysis in
`reports/eval-auto-report.md`. The thematic track has not started. The
earlier freeze on API budget is lifted; everything under "Decisions taken
before the freeze" is still settled and should not be reopened.

What exists:

- **Offline pipeline** — `ingest → preprocess → embed → index`, with
  stage-level caching on chained keys and `--force` to bypass.
- **Online search** — dense, lexical and hybrid retrieval. All state is
  loaded at query time, never fitted or rebuilt.
- **Four embedders** behind one protocol, selected from config: `bge-m3`,
  `e5-base`, `fasttext-avg`, `tfidf`.
- **Two index backends** — `numpy` (exact) and `faiss` (HNSW).
- **Two chunking arms** — `sections` and `whole_song`.

## Tests and quality

- **290 tests.** In the full local environment, with `faiss`, `gensim` and
  the real corpora present: **290 passed, 0 skipped**. The last figure
  verified in a clean clone on the base `requirements.txt` was **263
  passed, 5 skipped of 268** — CI run 31450986114 on commit `ed29dc8`,
  identical on 3.11 and 3.12; the suite has grown since, so the clean-clone
  split will differ.
- Every skip is a named missing dependency, never a hidden failure. The
  five at `ed29dc8` were three `pytest.importorskip("faiss")` sites in
  `test_indexes.py` and both parametrizations of
  `test_fitted_state_persistence.py`, which need a real `data/dev` build.
- **`lyrics_search/core/` is at 100% line coverage**, held there in CI by
  `pytest --cov=lyrics_search/core --cov-fail-under=100`; CI reports
  `Total coverage: 100.00%`.
- **`ruff check` and `ruff format --check`**, pinned to `ruff==0.16.2` in
  the workflow. `ruff` is in neither requirements file — CI installs it as
  its own step, so matching it locally needs `pip install ruff==0.16.2`.
- Tests run on Python 3.11 and 3.12.

## Corpus, as built

30,000 songs produce 208,411 chunks, of which 181,471 survive
preprocessing. All 26,940 rejects are chunk-level: 23,296
`chunk_duplicate_block` and 3,644 `chunk_too_short`. No song-level
rejects on this corpus.

All four embedders were built in one session against the same
`chunks.jsonl` (`input_hash 8fd3b50a…`), so the figures are comparable:

| embedder | dim | vectors | dropped, no signal | elapsed | peak VRAM | artifact |
|---|---|---|---|---|---|---|
| `bge-m3` | 1024 | 181,471 | 0 | 992.9 s | 1387.9 MB | 354.4 MiB fp16 |
| `e5-base` | 768 | 181,471 | 0 | 304.3 s | 692.4 MB | 265.8 MiB fp16 |
| `fasttext-avg` | 300 | 181,328 | 143 | 22.1 s | — | 103.8 MiB fp16 |
| `tfidf` | 76,649 | 181,453 | 18 | 4.9 s | — | 24.9 MiB sparse |

`e5-base` records `token_overflow_count: 3` against its 512-token limit
(longest chunk observed: 834 tokens). `bge-m3` overflows nothing — its
limit is 8192.

Genre lookup: `data/full/genre.jsonl`, 30,000 rows, exactly 7,500 each of
`country`, `rap`, `pop` and `rock`.

## What the eval phase added

- `core/metrics.py` — `recall_at_k`, `reciprocal_rank`,
  `mean_reciprocal_rank` and `ndcg_at_k` as pure functions, plus `mean_ci`
  for bootstrap intervals.
- `eval_set.py` — `EvalQuery` and `load_eval_set`, which skips `_meta`
  records and takes `known_song_ids` as required rather than optional: a
  skipped id check is indistinguishable in the output from a retriever
  that missed.
- `runners/eval.py` — the runner, with slices built from the data rather
  than from results (`split_by × force_split`, `genre`, `query_type`).
- `tfidf` moved to sparse CSR and its vocabulary cap was lifted: **76,649
  terms instead of 20,000**, in a 24.9 MiB compressed artifact.
- **Whole-song chunking, and `chunking` became a path component** —
  `data/<corpus>/<chunking>/…`, per `lyrics_search/paths.py`. Without it
  the two arms would overwrite each other's artifacts. `raw.jsonl` stays
  at the corpus root, ingest being the one stage that does not depend on
  chunking.
- Genre kept as a separate lookup table (`scripts/build_genre_lookup.py`,
  output into `data/`) rather than a field on `RawSong`. The source column
  is named `tag`; it is a dataset field, not a classification produced
  here.
- fastText converted to a binary `KeyedVectors` artifact — below.

## fastText load, and the cache-key trap it left

`scripts/convert_fasttext_vectors.py` produces
`cc.en.300.limit500000.kv`, loaded with `mmap='r'`. Measured on this
machine:

| | `.vec` text | `.kv` binary |
|---|---|---|
| word-vector load, 500k words | 231.3 s | 0.33 s |
| full `build_retriever`, fasttext arm | — | 5.83 s |
| query, cold / warm | — | 3.96 ms / 0.40 ms |

The `.vec` figure moves a great deal with page-cache state — 177 s, 231 s
and 498 s have all been measured on the same file. The `.kv` figure does
not. Identity was verified bit-for-bit: `np.array_equal` over the whole
500,000 × 300 matrix, with vocabulary order identical.

**The trap.** Both fasttext configs now name the `.kv` artifact, and the
embed stage's `stage_config` includes `params`, so the cache key moved:

```
embed  cache_key  e39c4c851eb12c42b100d0eccc4169249bfa2a03   (on disk, .vec)
                → e62544a86638b6634d2b22db4d0dbabb4c297a1d   (config,  .kv)
index  cache_key  26418502d793152bd306e64cdeebf7c0aa77f9c5
       input_hash e39c4c85…  — chains off the embed key, so it moves too
```

The next `build` on the fasttext arm **without `--force`** will therefore
recompute embed and index to produce vectors bit-identical to the ones
already on disk. That is correct behaviour, not a bug to route around: the
key is a function of the declared inputs, and the path is one of them. It
is written down only so the wasted pass is recognised as expected.

Search and eval are unaffected. They load from disk and never consult
cache keys, checking `fitted_state_sha1` and `dim` against `meta.json`
instead.

## Decisions taken before the freeze

1. **Two tracks** — an automatic known-item track and a thematic track
   with manual labelling. Reported separately, never averaged together.
2. **The "verbatim fragment" query type is deliberately cut.** Its result
   is known in advance, and in a table it would read as an advantage for
   lexical retrieval that was built into the task.
3. **LLM generation of queries is allowed.** Sending text to an API to be
   transformed is processing, not publication; the prohibition covers git
   and reproduction.
4. **An LLM as relevance judge is forbidden.** The automatic track's
   queries are already model-generated; judging them with a model as well
   would close the measurement on itself. A human judges.
5. **The run matrix is 9, not 24:** 4 dense + 4 hybrid + 1 lexical.
   `lexical` depends on neither the embedder nor the chunking —
   `build_retriever` never resolves an embedder in that mode, and BM25 is
   built over `text_raw` from `raw.jsonl`, which is shared across arms.
   `chunk_lookup` affects only which fragment is displayed, not the
   ranking. The eight nominal cells would return bit-identical numbers.
6. **The whole-song arm is out of scope for eval.** The config
   (`configs/full_dense_faiss_whole_song.yaml`) exists; the full-corpus
   artifacts do not. Only `data/dev/whole_song/` is built, tfidf only, 500
   songs to 500 chunks. Reason: 30k sequences of ~800 tokens against 181k
   of ~60, attention being quadratic in length, plus OOM risk on 4 GB VRAM
   at an 8192 context. A separate task, after both tracks.
7. **The automatic track samples 500 songs with deliberate
   over-sampling:** 400 `main` stratified by genre, 50 `structure`
   (`split_by` = `none` or `plain_label`), 50 `translation`. Stratum
   assignment runs from the scarcest pool outward — `translation` →
   `structure` → `main` — because the pools overlap. Measured on the built
   corpus: the translation pool holds 139 songs, the structure pool 2,410,
   and 3 songs fall in both.

## What remains

- Thematic track: 50 themes, pool depth 5, manual labelling in three
  grades. Not started, and not to be started without an explicit
  go-ahead.
- A 70-theme draft exists; the probe output through hybrid + `bge-m3` is
  in `data/probe_themes.txt` (70 blocks). A human does the filtering, and
  candidates for deletion are re-probed through
  `configs/full_hybrid_numpy_tfidf.yaml` first, because the error being
  guarded against is a false discard.

## Not to be carried forward

- **`tests/golden/spec03_eval_queries.jsonl` is not a benchmark.** 21
  records: one `_meta` and 20 queries, every one of them a song title.
  Titles typically recur verbatim in the lyrics, so the set systematically
  favours BM25. The file says so in its own `_meta` record.
- **The 14/15/16-out-of-20 figures in `reports/spec03-patch-report.md` are
  a floor check, not a result.** At n=20 the intervals overlap heavily; no
  ranking of the arms is supportable from them.

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
| Eval, automatic track | LLM query generation at batch 10, the rare-term leakage gate with three-attempt retry, 985 queries over 498 songs, the nine-arm matrix with per-arm GPU telemetry | `reports/eval-auto-report.md` |

Since `spec04-report.md` the repository has also been through a portfolio
standardization pass (ruff, CI, the coverage gate), which has no report of
its own.

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
  non-git data and produce output meant for a human to read rather than
  for CI: `spec03_latency_bench.py`, `spec03_retriever_comparison.py`,
  `full_index_sanity_check.py`, `dedup_check_full.py`,
  `build_genre_lookup.py`, `convert_fasttext_vectors.py`,
  `probe_themes.py`, `generate_demo_corpus.py`,
  `eval_gate_queries.py` (the rare-term leakage gate and its retry
  accounting), `eval_matrix.py` (the nine-arm run, with `nvidia-smi`
  sampled around every arm).
- **`tests/`** — the pytest suite. `test_fitted_state_persistence.py`
  verifies that the two fit-dependent embedders reproduce identical
  vectors across separate processes; it skips cleanly, per embedder, when
  the real data or `gensim` is missing.

These notes are English-only, unlike the bilingual README — they are
working detail rather than an introduction to the project.
