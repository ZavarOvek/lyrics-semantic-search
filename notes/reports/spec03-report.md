# SPEC-03 Phase 1b completion report

Online query branch, config/registry, stage-level build caching. Code
added: `contracts.py` (`SearchResult`), `core/fusion.py`, `core/aggregate.py`,
`core/cache.py`->`lyrics_search/cache.py`, `registry.py`, `config.py`,
`retrievers/{loading,dense,lexical,hybrid}.py`, `runners/{build,search}.py`,
`indexes/{numpy_index,faiss_index}.py` save/load, `configs/*.yaml`. Changed:
`runners/embed.py` (optional cache-key kwargs), `core/text.py` (promoted
`tokenize_words`), `embedders/fasttext_avg.py` (use the promoted helper).
110 tests collected, 108 passed + 2 skipped (faiss not installed in this
environment -- see below).

## §0.1: re-check of SPEC-02-PATCH item 3's acceptance check

Full writeup in Appendix A below. Summary: the
original check was **not buggy** -- both fit-dependent embedders really do
reproduce identical vectors across separate processes via `load_fit()`. The
"all-zero" note in the original report was a misleading debug artifact: it
printed `vec[:5]` of a 9079-dim sparse tfidf vector, and the query's 8
matched vocabulary terms happened to land at indices >= 83, so the first 5
slots were zero by coincidence, not because the vector was empty. Confirmed
all 8 query tokens are genuinely in the fitted vocabulary, and confirmed
there is no fit-time/query-time tokenization desync risk (`load_fit()`
reassigns the whole `TfidfVectorizer` object; `encode()` uses that same
object's `.transform()` regardless of how it got fitted -- one code path,
not two that could drift). Fixed `verify_fitted_state_persistence.py` to
print `nnz`/`norm`/`sum` instead of a positional slice; re-run shows
`nnz=8, norm=1.0` for tfidf and `nnz=300, norm=1.0` for fasttext-avg,
identical across processes. `lang-check-decision.md` §0.2 was
separately amended to soften an evidentiary overclaim flagged by the same
SPEC-03 §0 pass (not a code issue).

## §1/§2: query-time loading + empty/OOV query handling

`retrievers/loading.py` enforces "the online branch never fits/builds, only
loads" mechanically: `load_fitted_embedder()` raises `FileNotFoundError`
naming the exact missing path for a missing `meta.json` or fitted-state
file, and raises `ValueError` on a fitted-state SHA1 mismatch or a
post-`load_fit()` `dim` mismatch against what `embed.py` recorded at build
time. `warmup()` is a mandatory throwaway `encode()` call after loading, for
every embedder including tfidf. 13 tests in `test_query_loading.py`, all
green.

`SearchResult.status` (`contracts.py`) distinguishes:
- `"empty_query"` -- query string empty/whitespace after `.strip()`,
  caught before ever calling `encode()`.
- `"query_out_of_vocabulary"` -- query vector norm <= `1e-6`
  (`DenseRetriever`) or zero BM25-vocabulary token matches
  (`LexicalRetriever`); the index/BM25 is never queried in this case, so
  there are no meaningless all-tied-at-zero "matches".
- `"ok"` -- genuine search, `hits` reflects real ranking (possibly empty
  if nothing scored above zero).

`HybridRetriever.branch_statuses` additionally surfaces *partial*
degradation (e.g. dense OOV but lexical still found keyword hits) as
`{"dense": "query_out_of_vocabulary", "lexical": "ok"}` with an overall
`"ok"` status -- covered by
`test_hybrid_partial_degradation_visible_in_branch_statuses`. Verified
end-to-end (not just at the retriever-unit level) in
`test_end_to_end_search.py::test_empty_query_end_to_end`, which runs a real
`build()` then a real empty-string `search()` through the full pipeline.

## §3/§4: retriever comparison, dev corpus, real bge-m3 (20 queries)

Script: `scripts/spec03_retriever_comparison.py` (moved from `notes/` during
the SPEC-04 notes refactor). Eval set:
`tests/golden/spec03_eval_queries.jsonl` -- 20 songs sampled evenly across
`data/dev/raw.jsonl` (every `500//20`-th song), query = that song's own
title, judgment = that song's `song_id`. This is a cheap bootstrap
methodology (no manual lyrics reading/relevance judging), not a rigorous
IR benchmark -- a retriever that can't surface a song for a query built
from its own title is failing an easy case, so it's a floor/sanity check,
not a ceiling.

Recall@10 (return_n=10, top_k=50):

| retriever | hits | Recall@10 |
|---|---|---|
| dense (bge-m3) | 14/20 | 0.70 |
| lexical (BM25) | 15/20 | 0.75 |
| **hybrid (RRF)** | **16/20** | **0.80** |

Hybrid strictly dominates both individual branches on this eval set --
consistent with the intended purpose of RRF fusion (each branch's misses
are partly independent: `'Still Around'` and `'STRANGERS'` are dense
misses that lexical recovers; `'23'` is a lexical miss that dense
recovers; hybrid gets both). 4/20 queries (`'Bring it Back'`,
`'You Spilt Beer On My Canvas'`, `'Nightstands'`,
`'Every Friend I Ever had...'`) miss on all three retrievers -- titles
that apparently don't recur verbatim/closely in their own lyrics, an
expected limitation of the title-as-query bootstrap, not a retriever bug.

Aggregation-by-maximum (§4) is unit-tested directly in
`tests/test_aggregate.py` (7 tests) including
`test_song_never_appears_twice` (a song can never produce two Hits even
when several of its chunks match) and
`test_one_chunk_per_song_degenerate_case` (the function needs no special
case when every song has exactly one chunk -- relevant because bge-m3's
real `max_seq_length=8192`, found in SPEC-02-PATCH, makes
whole-song-as-one-chunk a real config, not just a theoretical edge case).

## §7: cold vs warm latency, full corpus (~181k chunks)

Script: `scripts/spec03_latency_bench.py` (moved from `notes/` during the
SPEC-04 notes refactor). **faiss is not installed in this
environment** (`pip show faiss` / `import faiss` both fail; this also
accounts for the 2 skipped tests in `test_indexes.py`, which
`pytest.importorskip("faiss")`) -- benchmarked `NumpyIndex` (exact
brute-force dot product) instead, which is still the right number for
"warm per-query overhead" since faiss's build cost is a separate, already
one-time-measured concern (SPEC-02: ~187s HNSW build at full scale).

This machine also has **no CUDA** (`torch.cuda.is_available() == False`),
so bge-m3 runs on CPU:

| stage | time |
|---|---|
| load `chunks.jsonl` lookup (181471 chunks) | 2.20s |
| `NumpyIndex.build()` (181471 x 1024 float32) | 0.63s |
| bge-m3 model load (incl. cudnn/kernel warmup) | 7.16s |
| mean cold query (5 distinct queries, model already loaded) | 183.9ms |
| mean warm query (10 repeats x 5 queries) | 195.3ms |

**<100ms warm target: FAIL** (195.3ms). Root-cause breakdown (separately
timed `encode()` vs `index.search()` for one query, 3 repeats):

```
encode=156.8ms search=30.2ms
encode=150.8ms search=30.6ms
encode=151.1ms search=30.8ms
```

The retrieval-infrastructure layer itself (`NumpyIndex.search()` over
181k x 1024 float32) is **~30ms, comfortably within budget**. The failure
is entirely attributable to bge-m3's own CPU inference cost for a single
short query string (~150ms) -- not the code this phase built. On a GPU
(sentence-transformers + CUDA + fp16, the codepath `BGEM3Embedder` already
supports and this project's earlier full-corpus embed run at 860s used) a
single-string encode is typically a low-single-digit-ms operation, which
would land the same query comfortably under the 100ms target. This is
flagged as an **environment limitation of this particular test run** (no
GPU available here), not a finding about the retrieval architecture --
worth re-measuring on GPU hardware before treating the target as
genuinely unmet.

## Flagged from Phase C: real faiss bug found and fixed

`faiss.write_index()`/`read_index()` (the standard save/load API) call
into faiss's C++ file I/O, which **fails to open any path containing
non-ASCII characters on Windows** -- and this project's own repository
root is such a path, so every real save/load in this
repo would have failed, not just a contrived test case. Fixed in
`indexes/faiss_index.py` by switching to
`faiss.serialize_index()`/`deserialize_index()` (in-memory bytes, written
to disk via Python's own `open()`, never touching the filesystem from
C++). Documented in the module's own docstring. Covered by
`test_faiss_index_save_load_roundtrip` (skipped in this environment since
faiss isn't installed here, but was exercised and confirmed passing
against a real faiss install during Phase C).

## §6: stage-level build caching

`cache.py`: `cache_key = sha1(canonical_json(stage_config) + "|" +
input_hash)`, chained (`input_hash` = previous stage's own `cache_key`,
`ROOT_INPUT_HASH` sentinel for ingest). Flat-file stages (ingest,
preprocess) use a `<artifact>.meta.json` sidecar written strictly after
the artifact via `atomic_write_bytes()`; directory stages (embed, index)
build into a `.building_tmp` sibling (with `meta.json` written inside it
before the swap) then `atomic_replace_dir()` into place in one step --
verified this two-step directory swap is necessary on Windows
(`os.replace()` on a non-empty directory raises `PermissionError`, unlike
POSIX `rename(2)`). `--force` bypasses every check.

Verified end-to-end in `test_end_to_end_search.py`:
`test_rebuild_skips_all_fresh_stages` (a second `run_build()` call prints
`skip` for all 4 stages) and `test_force_bypasses_cache_and_rebuilds`
(with `force=True`, none of them do). 8 e2e tests total, all against a
local `tests/golden/spec03_mini_corpus.jsonl` fixture via
`JsonlFileSource` -- the real `data/dev`/`data/full` corpora were
deliberately never touched by automated tests this phase (they predate
SPEC-03 and lack the sidecar meta files caching would need, so pointing
`build` at them would trigger an unplanned full re-ingest/re-embed).

## §5: registry + config

`registry.py`: `@register_embedder`/`@register_index` name->factory maps,
lazy imports inside each factory so importing the registry itself never
pulls in torch/faiss. `fasttext-avg` deliberately not registered (needs a
`vectors_path` that doesn't fit the zero-arg factory shape, per its own
docstring). `config.py`: pydantic `ExperimentConfig`/`RetrievalConfig`,
`extra="forbid"`, cross-validated against the registry's actual
registered names at `load_config()` time (a typo or `fasttext-avg` in a
YAML file fails loudly at load time, not deep inside a build/search run).
19 tests (`test_registry.py` + `test_config.py`). 3 example configs in
`configs/`: `dev_dense_numpy.yaml`, `dev_hybrid_numpy.yaml`,
`full_dense_faiss.yaml` (numpy default for dev, faiss for full, per §5) --
all 3 verified to `load_config()` successfully.

## Not in scope for this phase (unchanged from spec)

- Query-side language detection / non-English query handling.
- Any UI/API layer beyond the two CLIs (`build --config X.yaml`,
  `search --config X.yaml "query" [--repeat N]`).
- Re-ranking beyond RRF fusion.
- GPU latency re-measurement (flagged above, not performed -- no GPU
  available in this environment).

## Maintenance

`sentence-transformers` and `rank_bm25` were both missing from this
environment's Python install and were installed via `pip install` during
this session (needed for the real-bge-m3 comparison and for
`LexicalRetriever` respectively) -- both were already declared
dependencies of code from earlier phases, just not present in this
particular environment until now. `faiss` remains not installed here;
the faiss code path is implemented and was tested against a real faiss
install in an earlier phase (Phase C), but could not be re-verified in
this session's environment.

## Appendix A: re-check of SPEC-02-PATCH item 3's acceptance check (§0.1 detail)

(notes refactor: merged in from the former standalone
`notes/spec03-item0.1-zero-vector-finding.md`, referenced from §0.1 above.)

### A.1 What SPEC-03 flagged

`notes/reports/spec02-patch-report.md` (item 3) footnoted the tfidf
acceptance run with: "tfidf's query vector happens to be all-zero because
none of the query's words are in the fitted dev-corpus vocabulary". SPEC-03
§0.1 pointed out this made the check meaningless as originally printed --
two processes agreeing on an all-zero vector is indistinguishable from two
processes that both silently failed to load state (e.g. a `load_fit()`
that's a no-op bug would *also* produce two identical zero vectors). It
required either confirming the zero vector was genuinely due to an OOV
query (and finding a different query to test the nonzero path), or -- if
not genuinely OOV -- finding the real bug.

### A.2 Investigation

The query was `"walking alone in the rain thinking about you"`, checked
against `data/dev/embeddings/tfidf/fitted_state.joblib`
(`TfidfVectorizer(max_features=20000, norm="l2")`, all other params at
sklearn defaults: `ngram_range=(1,1)`, `lowercase=True`,
`token_pattern=r'(?u)\b\w\w+\b'`, `stop_words=None`). Loaded vocabulary size:
9079.

Checked every content word from the query against `vectorizer.vocabulary_`
directly:

| token | in vocabulary | index |
|---|---|---|
| walking | yes | 8649 |
| alone | yes | 235 |
| rain | yes | 6238 |
| thinking | yes | 8046 |
| about | yes | 83 |
| you | yes | 9018 |
| in | yes | 3999 |
| the | yes | 8020 |

All 8 tokens (the `\b\w\w+\b` pattern requires 2+ word chars, so single-letter
words would be dropped, but there are none here) are in the fitted
vocabulary. The query is **not** OOV.

### A.3 Root cause: a misleading debug print, not a bug

The original script printed `vec[:5]` -- the first 5 of 9079 dimensions.
Since the 8 nonzero entries above land at indices 83, 235, 3999, 6238, 8020,
8046, 8649, 9018, none of them fall in `[0:5]`, so the printed slice was all
zeros purely by coincidence of index layout, not because the vector itself
was empty.

Re-running with the actual vector stats
(`tests/test_fitted_state_persistence.py`, the promoted successor of the
original `notes/verify_fitted_state_persistence.py` script, fixed to print
`nnz`/`norm`/`sum` instead of a positional slice):

```
[tfidf] process A: nnz=8, norm=1.000000, sum=2.586803 | process B: nnz=8, norm=1.000000, sum=2.586803 | identical=True
[fasttext-avg] process A: nnz=300, norm=1.000000, sum=-0.142774 | process B: nnz=300, norm=1.000000, sum=-0.142774 | identical=True
PASS: both fit-dependent embedders produce identical vectors across separate processes.
```

`nnz=8` matches the 8 vocabulary hits exactly. `norm=1.0` matches the
vectorizer's `norm="l2"` setting. Both are bit-identical across two genuinely
separate subprocesses, for both embedders.

### A.4 Was there a tokenization desync risk?

SPEC-03 §0.1 also asked whether a real fit-time vs. query-time tokenization
mismatch was possible (different lowercasing/token pattern/normalization
between the two code paths), which would be a serious bug independent of
this particular query. There is no such risk in the current code:
`load_fit()` deserializes and reassigns the *entire* `TfidfVectorizer`
object (`self._vectorizer = joblib.load(path)`), and the same
`self._vectorizer.transform(...)` call in `encode()` is used regardless of
whether the vectorizer arrived via `fit()` or `load_fit()`. There is only
one tokenization code path (inside the persisted sklearn object itself), not
two independently-maintained ones that could drift apart.

### A.5 Conclusion

Item 3's original acceptance check was valid in substance (state really is
persisted and reproduced identically cross-process) but was reported with a
misleading and factually incorrect explanation ("all-zero because OOV"),
caused entirely by printing an uninformative slice of a high-dimensional
vector. No code bug existed. Fixed by changing the diagnostic output to
`nnz`/`norm`/`sum`, which correctly show the vector is nonzero and
L2-normalized as expected. `notes/reports/spec02-patch-report.md` is left
as-is (historical record of that session); this appendix is the correction
of record.
