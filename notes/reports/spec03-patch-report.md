# SPEC-03-PATCH completion report

Patch to SPEC-03 Phase 1b (`notes/reports/spec03-report.md`), not a new
phase, addressing all 6 items from the patch spec. `notes/reports/spec03-report.md`
is left as originally written (per the SPEC-02-PATCH precedent); the
corrected/superseded numbers and claims from its §3 and §7 live here
instead. Code changed: `lyrics_search/registry.py`, `lyrics_search/config.py`,
`lyrics_search/runners/build.py`, `lyrics_search/runners/search.py`,
`tests/test_registry.py`, `tests/test_config.py`. New:
`configs/dev_dense_numpy_fasttext.yaml`, `configs/dev_dense_numpy_tfidf.yaml`,
`configs/dev_dense_numpy_e5base.yaml`, `notes/decisions/faiss-nonascii-path-windows.md`.

**Interpreter used for all work in this patch:**
`./.venv/Scripts/python.exe` (this project's standing environment rule, and
per item 2 below). Every command in this report ran through that
interpreter, never the global `python`.

## Item 2: environment verification (read first -- explains why §3/§7 numbers change)

The original SPEC-03 report was produced against the **global** Python
install, which lacks CUDA torch, faiss, and (initially) sentence-transformers.
That produced two concrete defects, both confirmed and now fixed by
switching to `.venv`:

```
torch 2.6.0+cu124, cuda available: True
faiss 1.14.3
sentence-transformers 5.6.1
```

- **2 previously-skipped faiss tests now run for real** (`pytest.importorskip("faiss")`
  no longer skips). Full suite: **115 passed, 0 skipped** (`.venv`).
- **§7's 195.3ms "FAIL" latency number was a CPU-only artifact**, not a
  finding about the retrieval architecture -- exactly as the original report's
  own caveat predicted ("worth re-measuring on GPU hardware before treating
  the target as genuinely unmet"). Corrected measurement below.

### §7 corrected: first-pass vs repeat query latency, full corpus (~181k chunks), GPU

**Naming correction (SPEC-04 §0.3):** the original SPEC-03 table called
these columns "cold" and "warm", but both are measured *after*
`build_retriever()`'s own `warmup(embedder)` call, which SPEC-03 made
mandatory -- so neither column is a true cold start. Relabeled below to
"first pass" (first query after load+warmup) and "repeats" (subsequent
queries) to describe what's actually measured. The real cold start --
Phase 0 measured it at 476ms for a single unwarmed encode call -- still
exists, it's just no longer visible as a separate line item: it's now
folded into the ~10s "model load" row below, which is exactly what
`warmup()` was added to do (absorb the one-time JIT/cudnn-autotune cost
into load time so it never lands on a real query).

Script: `scripts/spec03_latency_bench.py` (moved from `notes/` during the
SPEC-04 notes refactor), now additionally timing `encode()`
and `index.search()` separately (previously only the combined
`retriever.search()` call was measured). Still benchmarks `NumpyIndex`, not
`FaissIndex` -- deliberate, see the script's own header: it isolates the
retrieval-layer's own search cost from HNSW's approximate-search behavior,
and is the more conservative (slower) of the two at full-corpus scale, so
it's the right one to hold the <100ms target to.

First attempt (mid-session) measured mean repeat = 130.4ms, an apparent
regression from an earlier clean run's 81.0ms. Root-caused to GPU contention
from other processes sharing the same 4GB laptop GPU (`nvidia-smi` showed
Minecraft's `javaw.exe` actively rendering at the time, alongside Discord/
Vivaldi's own GPU-accelerated compositing). Re-measured after those were
closed (`nvidia-smi`: 308MiB/4096MiB, 19% util immediately before the run
below) for a clean, reproducible number:

| stage | time |
|---|---|
| load `chunks.jsonl` lookup (181471 chunks) | 1.34s |
| `NumpyIndex.build()` (181471 x 1024 float32) | 0.68s |
| bge-m3 model load (incl. CUDA/cudnn warmup -- true cold start absorbed here) | 9.98s |
| mean first-pass query (5 distinct queries, model already loaded+warmed) | 53.2ms |
| mean repeat query (10 repeats x 5 queries) | **55.6ms** |
| &nbsp;&nbsp;of which mean `encode()` | 21.4ms |
| &nbsp;&nbsp;of which mean `index.search()` | 32.4ms |

**<100ms repeat-query target: PASS** (55.6ms). The retrieval-infrastructure layer
(`NumpyIndex.search()` over 181k x 1024 float32, ~32ms) and the model's own
GPU inference (`encode()`, ~21ms) are now both comfortably within budget
individually and combined -- confirming the original report's own
hypothesis: the CPU environment, not the code, was the entire cause of the
prior FAIL.

Practical note for future benchmarking on this machine: this GPU is a
shared 4GB laptop part: 3D-heavy foreground applications (games in
particular) measurably move these numbers (55.6ms clean vs 130.4ms
contended, same code, same machine, ~30 minutes apart). Re-measure with
`nvidia-smi` checked immediately beforehand if a number here looks off.

## Item 1: registry factory shape -- fasttext-avg now registrable from YAML

`fasttext-avg` needs a `vectors_path` constructor arg, which didn't fit the
original zero-arg factory shape (`registry.py` previously hard-coded
`fasttext-avg` as *not* registered, and `config.py` had a dedicated test
asserting that config-loading it raised). Redesigned:

- `registry.py`: factories now accept `**params`, forwarded straight to the
  real class `__init__` (no separate params-schema translation layer).
  `@register_embedder(name, required_params=(...))` records which params
  are required *without instantiating anything* -- `required_embedder_params(name)`
  is queryable at config-load time. `fasttext-avg` registered with
  `required_params=("vectors_path",)`.
- `config.py`: new `EmbedderConfig(name: str, params: dict = {})`. A
  `field_validator(mode="before")` coerces a plain string
  (`embedder: bge-m3`) into `{"name": "bge-m3"}`, so all 3 pre-existing
  configs load unchanged -- full backward compatibility, not a breaking
  change. `_validate_registry_names()` now also checks
  `registry.required_embedder_params(config.embedder.name)` against
  `config.embedder.params` and raises `ValueError` ("missing required
  params") at `load_config()` time if any are absent -- fails loud before
  any instantiation, per SPEC-00 §3.2.
- `build.py`/`search.py` updated at every `config.embedder` use site;
  embed stage's cache `stage_config` now includes `params`, so a
  params-only config change (e.g. a different `vectors_path`) correctly
  invalidates the embed-stage cache.
- Tests: `test_registry.py` (`test_fasttext_avg_is_registered`,
  `test_fasttext_avg_declares_vectors_path_required`,
  `test_bge_m3_has_no_required_params`, `test_tfidf_resolves_with_params`)
  and `test_config.py` (`test_load_config_fasttext_avg_without_vectors_path_raises`,
  `test_load_config_fasttext_avg_with_vectors_path_loads`,
  `test_load_config_embedder_plain_string_shorthand_still_works`).

**Live proof, not just unit tests**: ran real `build` CLI invocations, via
`.venv`, against `configs/dev_dense_numpy.yaml` (bge-m3, string shorthand),
`configs/dev_hybrid_numpy.yaml` (bge-m3, shared cache), the new
`configs/dev_dense_numpy_tfidf.yaml`, `configs/dev_dense_numpy_e5base.yaml`,
and `configs/dev_dense_numpy_fasttext.yaml` (object form: `embedder: {name:
fasttext-avg, params: {vectors_path: data/models/fasttext/cc.en.300.vec}}`).
**All 4 embedders (bge-m3, e5-base, tfidf, fasttext-avg) built successfully
from pure YAML config, zero code changes needed between them.**

## Item 3: sidecar-meta on real corpora -- verified, both corpora

The original report deliberately never pointed `build` at `data/dev` or
`data/full` (they predate SPEC-03 and lack the `.meta.json`/`meta.json`
sidecar cache files), so the cache-skip behavior was proven only against a
synthetic fixture. Chose the "run `build` for real, once per corpus" option
(the patch spec left this choice open) over retroactively fabricating meta
files, since a real run also doubles as the item-1 live-YAML proof above.

**Dev corpus** (`data/dev`, 500 songs, 2951 chunks): ran `dev_dense_numpy.yaml`
twice back-to-back -- second run printed `skip -- fresh` for all 4 stages
(ingest, preprocess, embed, index). Same for `dev_hybrid_numpy.yaml` (shares
the bge-m3 embed cache with the first config, correctly reused it).

**Full corpus** (`data/full`, 30000 songs): first-ever real `build` run for
this corpus under SPEC-03's caching:
```
Ingested 30000 songs
Chunks: 208411 generated, 181471 kept, 26940 rejected
[bge-m3] 181471 chunks -> dim=1024, 1377.7s, norm_ok=True, dropped_no_signal=0, peak_vram=1388MiB
[index:faiss] built (cache_key=64181d33ba65)
Build complete: data\full
```
Re-ran the identical config immediately after -- **all 4 stages printed
`skip -- fresh`.** Both invariants (chunk counts, embed dim/norm) match the
original SPEC-02(-PATCH) full-corpus numbers exactly, confirming no drift
from the registry/config changes.

**Observation (not a bug, not fixed, out of scope for this patch):** running
`dev_dense_numpy_tfidf.yaml` a second time immediately after
`dev_dense_numpy_fasttext.yaml` printed `[index:numpy] built`, not `skip`,
which briefly looked like a caching regression. Root cause: `build.py`'s
index artifact path is `<corpus_dir>/indexes/<index_type>` -- keyed only by
index type, not by embedder name. Two different embedder configs sharing
the same corpus and index type (`numpy`) therefore share one on-disk index
slot; building index-type `numpy` for embedder B after embedder A correctly
invalidates and rebuilds that shared slot (the cache-key check is working
exactly as designed -- it detected the requested cache key didn't match
what was on disk, because a different build had run in between). Confirmed
this is genuine same-config skip behavior, not a broken check, by
immediately re-running the identical tfidf config a third time: all 4
stages then printed `skip -- fresh`. Each config in this repo represents
one experiment (SPEC-02/03 design intent); running several different
embedders against the same corpus+index-type back-to-back for smoke-testing
(as this patch's item-1 verification did) is not a real usage pattern this
needs to optimize for, so left as-is.

## Item 4: comparison claims -- softened, ceiling-adjusted table added

The original §3's "hybrid strictly dominates both individual branches" is
**not statistically supported** at n=20: 95% Wilson score confidence
intervals overlap heavily across all three retrievers. Re-ran
`spec03_retriever_comparison.py` after all above changes -- results
reproduced exactly (dense 14/20, lexical 15/20, hybrid 16/20), confirming
determinism.

**Raw, n=20 (all queries):**

| retriever | hits | Recall@10 | 95% CI (Wilson) |
|---|---|---|---|
| dense (bge-m3) | 14/20 | 0.70 | [0.48, 0.86] |
| lexical (BM25) | 15/20 | 0.75 | [0.53, 0.89] |
| hybrid (RRF) | 16/20 | 0.80 | [0.58, 0.92] |

All three intervals overlap substantially -- at this sample size, none of
the three pairwise differences (dense vs lexical, lexical vs hybrid, dense
vs hybrid) is distinguishable from noise. "Strictly dominates" overstates
what n=20 can support.

The report also **understated** hybrid's result by not adjusting for a
ceiling: 4/20 queries (`'Bring it Back'`, `'You Spilt Beer On My Canvas'`,
`'Nightstands'`, `'Every Friend I Ever had...'`) miss on **all three**
retrievers -- these titles apparently don't recur verbatim/closely in their
own lyrics, making them unanswerable by this eval set's own title-as-query
construction, not a retriever failure. Excluding them (denominator 16, not
20):

**Ceiling-adjusted, n=16 (excluding the 4 always-miss queries):**

| retriever | hits | Recall@10 | 95% CI (Wilson) |
|---|---|---|---|
| dense (bge-m3) | 14/16 | 0.875 | [0.64, 0.97] |
| lexical (BM25) | 15/16 | 0.9375 | [0.72, 0.99] |
| hybrid (RRF) | 16/16 | **1.0** | [0.81, 1.0] |

Softened conclusion: on this bootstrap set, hybrid answers every query that
is answerable by the eval set's own construction (16/16), and its point
estimate is highest at both n=20 and n=16 (consistent with RRF fusion's
intended purpose -- dense misses `'Still Around'`/`'STRANGERS'` are
lexical hits, lexical's miss on `'23'` is a dense hit, hybrid recovers
both) -- but at this sample size the difference from lexical alone is not
statistically distinguishable, and should not be reported as such. A larger,
non-title-derived eval set (deferred to the eval phase, out of scope here
per the patch spec) is needed before any dominance claim can be made with
confidence.

## Item 5: eval-set methodology risk -- labeled bootstrap-only everywhere

Title-as-query systematically favors lexical retrieval (a song's title
typically recurs verbatim in its own lyrics), so `lexical > dense` on this
set is very likely a construction artifact of the eval methodology, not a
property of the retrievers. Added the exact required wording in all three
required places:

- `tests/golden/spec03_eval_queries.jsonl`: new leading sentinel record,
  `{"_meta": "Bootstrap sanity set only. Queries are song titles, which
  typically recur verbatim in the lyrics; this systematically favours
  lexical retrieval. NOT to be reused as the eval-phase benchmark."}` --
  `spec03_retriever_comparison.py`'s query-loading loop now explicitly
  skips any record containing `_meta` so it's never miscounted as a 21st
  query (verified: re-run still counts exactly 20).
- `spec03_retriever_comparison.py`'s own header docstring: same caveat,
  spelled out in full.
- This report (§4/§5 above) and the ceiling-adjusted table's own framing.

## Item 6: faiss non-ASCII Windows path bug -- standalone note written

`notes/decisions/faiss-nonascii-path-windows.md` -- symptom, root cause (faiss's C++
`fopen()` doesn't reliably handle non-ASCII paths on Windows), why this
specific repo is hit (the checkout's root directory contains non-ASCII
characters), the fix (`serialize_index()`/`deserialize_index()` via
Python's own `open()`, already implemented in `indexes/faiss_index.py`
since Phase C), and test coverage (`test_faiss_index_save_load_roundtrip`,
now genuinely exercised -- not skipped -- in this patch's `.venv` runs).
Written standalone, intended for later README reuse per the patch spec.

## Full pytest suite, final state

```
./.venv/Scripts/python.exe -m pytest tests -q
115 passed
```
0 skipped -- both faiss tests that were previously
`pytest.importorskip`-skipped now run for real against a genuine faiss
install.

## Not in scope for this patch (per the patch spec, unchanged)

- Designing a real, non-title-derived eval set -- owner's decision.
- Running e5-base/tfidf/fasttext-avg comparisons on the full 30k corpus.
- Reranking, whole-song-as-one-chunk arm, any UI/API work.
