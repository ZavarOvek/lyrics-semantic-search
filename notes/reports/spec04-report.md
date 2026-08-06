# SPEC-04 completion report

Revision 2 of SPEC-04. Written incrementally as each numbered step in the
owner's execution order completes -- see `notes/STATUS.md` for the current
overall project state; this file is the per-phase detail for SPEC-04
specifically.

## notes/ refactor + SPEC-04 §2 data-hygiene audit

Combined into one pass per the owner's instruction, done alongside §0
rather than at SPEC-04's own §2 position. `.py` scripts moved out of
`notes/` into `scripts/` (reusable, real-data-dependent, human-interpreted
output) or `tests/` (real pytest, `verify_fitted_state_persistence.py`
promoted to `tests/test_fitted_state_persistence.py` with a clean skip when
real data is absent). `.md` files reorganized into `notes/decisions/`
(durable design rationale) and `notes/reports/` (one file per phase, small
companion findings merged in as appendices rather than left standalone).
Cross-references inside the moved reports/scripts updated to the new
paths. See `notes/STATUS.md` for the resulting structure.

**§2 data-hygiene audit result: clean, no changes needed.** Checked both
files in `tests/golden/` and every file in `notes/` (after the reorg above)
for full song-lyrics text:

- `tests/golden/spec03_mini_corpus.jsonl` -- already fully synthetic (4
  songs, fabricated artist/title names, generic fabricated lyrics text).
  Not real copyrighted content.
- `tests/golden/spec03_eval_queries.jsonl` -- contains only song titles and
  hashed `song_id`s (plus the bootstrap-methodology `_meta` record); no
  lyrics text at all.
- `notes/reports/*.md`, `notes/decisions/*.md` -- quote song/artist
  *titles* as examples (dedup groups, artist-pollution scraper matches,
  vocabulary tokens) but never full lyrics. One single-line lyrics
  fragment appears in `notes/decisions/lang-check-decision.md`, quoted to
  illustrate the single lowest-confidence song in the language-ID
  calibration -- judged here as a short fragment used for a genuine
  technical point, well within the "short fragments for
  reproduction/illustration are fine, full texts are not" line, so left
  as-is.

No redactions or synthetic replacements were needed.

**Later correction:** that judgement was reversed before publication. The
repository states that no song lyrics are published and excludes `data/`
from git for that reason; a quoted line in a notes file contradicted it
regardless of length. The fragment was removed from the decision file and
from the sentence above.

## §0.1: index path now includes the embedder name

**Problem (as filed):** `<corpus_dir>/indexes/<index_type>` was keyed only
by index type, so two different embedder configs sharing a corpus+index-type
shared one on-disk slot and evicted each other -- exactly the pattern the
eval phase needs (several embedders, same corpus, same index type).

**Fix:** `lyrics_search/runners/build.py` and `lyrics_search/runners/search.py`
now both use `<corpus_dir>/indexes/<embedder>/<index_type>`.

**Verified on real dev-corpus data** (not just synthetic fixtures):
```
data/dev/indexes/bge-m3/numpy
data/dev/indexes/tfidf/numpy
```
Both built and coexist on disk. Ran `dev_dense_numpy.yaml` then
`dev_dense_numpy_tfidf.yaml` back-to-back, then repeated both a second
time: **every stage on every config printed `skip -- fresh`, regardless of
which config ran last** -- the shared-slot eviction from SPEC-03-PATCH's
report is gone. Also rebuilt `data/full`'s faiss index at its new path
(`data/full/indexes/bge-m3/faiss`) since the old orphaned
`data/full/indexes/faiss` was removed; full pytest suite still 115 passed
after the change (synthetic fixtures don't hardcode index paths, so no
test changes were needed).

## §0.2: full-corpus bge-m3 embed time -- GPU contention, not a regression

**Problem (as filed):** SPEC-03-PATCH's full-corpus embed took 1377.7s vs
848.0s (SPEC-02-PATCH) and 860.3s (SPEC-02) -- the two earlier numbers
agree within 1.5%, the new one is 62% higher, despite identical work
(181471 chunks, dim=1024, norm_ok=True, dropped_no_signal=0, peak VRAM
1388MiB unchanged across all three).

**Investigation.** A full re-run solely to test this wasn't done (the spec
explicitly allows deferring to the next natural full run + `nvidia-smi`
comparison, since a dedicated re-run is a ~23-minute GPU-exclusive cost).
Instead, gathered the evidence already available plus one cheap additional
data point:

- Identical peak VRAM (1388MiB) across all three runs rules out a
  memory-pressure or batch-size-driven explanation -- same algorithm, same
  data, same memory footprint.
- This exact GPU (RTX 3050 Laptop, 4GB, shared with the desktop session)
  was already shown, within the *same* SPEC-03-PATCH session, to be
  sensitive to concurrent foreground GPU load: the latency benchmark
  measured 130.4ms mean repeat-query with Minecraft (`javaw.exe`) actively
  rendering, vs 55.6ms clean immediately after closing it -- a 2.35x
  slowdown from contention alone, same code, ~30 minutes apart.
- One further data point gathered just now, incidental to §0.3's script
  re-verification: with Minecraft running again in the background
  (confirmed via `nvidia-smi --query-compute-apps`), the same latency
  script measured 91.2ms mean repeat-query -- a third, independent
  contention-correlated slowdown on this machine, at a different
  contention level than either of the other two runs.
- 1377.7s / 848.0s = 1.62x -- the same *direction* and a broadly
  comparable *order of magnitude* to the contention-driven slowdowns
  observed directly on this GPU, though for a ~23-minute sustained
  encode job (not five short queries), thermal throttling on a laptop chip
  is an equally plausible contributor and isn't distinguishable from pure
  contention without power/thermal telemetry this investigation didn't
  collect.

**Conclusion (best available without a dedicated re-run):** the 1377.7s
figure is very likely explained by GPU contention and/or thermal
throttling from concurrent desktop use during that particular background
run, not a regression in the embedding code -- the identical VRAM/chunk/dim
numbers rule out a code-path difference, and this same GPU has now been
shown three separate times in one session to degrade measurably under
concurrent load. **Not fully confirmed**: this is a hypothesis backed by
strong circumstantial evidence (three same-session, same-hardware
contention correlations, all in the same direction), not a controlled
repeat of the exact 23-minute job with and without contention. Per the
patch spec, confirmation is deferred to the next full-corpus rebuild that
happens for other reasons -- check `nvidia-smi` immediately before and
during that run and compare wall-clock time against the 848.0s/860.3s
baseline.

## §0.3: latency table relabeled -- "first pass" / "repeats", not "cold" / "warm"

**Problem (as filed):** the "cold"/"warm" column names in SPEC-03's and
SPEC-03-PATCH's latency tables are misleading -- both columns are measured
*after* `warmup()` (mandatory since SPEC-03 §1), so neither is a true cold
start. The real cold start (Phase 0: ~476ms for one unwarmed encode call)
is actually still there, just absorbed into the ~10-15s "embedder load"
step, which is exactly what making `warmup()` mandatory was for.

**Fix:**
- `notes/reports/spec03-patch-report.md`'s §7 section renamed to "first-pass
  vs repeat query latency", table rows renamed accordingly, and a new
  explanatory paragraph added stating explicitly where the true cold start
  went (see that file for the exact wording).
- `scripts/spec03_latency_bench.py` (moved out of `notes/`, per the notes
  refactor below): variable names, print labels, and the header
  comment all renamed `cold`/`warm` -> `first_pass`/`repeat`; the script
  now also calls `retrievers.loading.warmup()` explicitly after
  constructing the embedder (mirroring `build_retriever()`'s own real
  sequence exactly, not just relying on `BGEM3Embedder.__init__`'s
  internal warmup call, which happens to make the two equivalent for
  bge-m3 specifically but wouldn't for an embedder that doesn't self-warm).

The canonical clean-GPU numbers from SPEC-03-PATCH's report (55.6ms mean
repeat query) are unchanged and still stand as the report's number -- this
item only corrects labeling and adds the explanatory note, not the
measurement itself. A post-rename smoke-test re-run of the script (see
§0.2 above, third bullet) confirmed the renamed code still runs correctly
and produces numbers in the same ballpark under the same (this time
Minecraft-contended) conditions.

## §1: core/ test coverage -- 100%, plus hypothesis property tests

**Baseline:** `lyrics_search/core/` (six pure, I/O-free modules) was at 74%
line coverage overall. `sections.py` -- the four-level chunking cascade,
the largest and most branch-heavy module in `core/` -- was the outlier at
only 60% (64 missing lines), covered only incidentally through
`test_preprocess_song_filter.py`'s integration test, with no dedicated
test file of its own. `dedupe.py` was at 92% (1 missing branch).
`aggregate.py`, `fusion.py`, `rejects.py`, and `text.py` were already at
100%.

**Tooling added:** `pytest-cov`, `coverage`, and `hypothesis` were not
previously installed or pinned anywhere in the project; added to
`requirements.txt` (`coverage==7.15.2`, `hypothesis==6.164.0`,
`pytest-cov==7.1.0`, plus hypothesis's transitive dep
`sortedcontainers==2.4.0`).

**New test files:**

- `tests/test_sections.py` -- direct unit coverage for the chunking
  cascade: `classify_bracket` and `classify_plain_label_line`'s
  classification branches (canonical/credits/noise, compound
  Pre-Chorus/Post-Chorus, case-insensitivity, the deliberately
  conservative ≤2-token guard); `chunk_song`'s levels 1+2 (bracket_tag +
  plain_label boundaries combined positionally, leading text, noise
  stripping, mixed-boundary-type ordering), level 3 (blank-line
  refinement), and level 4 (force-split length packing, including the
  `warn()` callback firing once per oversized segment with the correct
  pre-split word count). Closing the last few branches required directly
  testing three private helpers (`_find_bracket_boundaries`,
  `_pack_by_length`, `_split_long_line`) the same way
  `test_preprocess_song_filter.py` already directly tests
  `_song_reject_reason` -- one branch (`_find_bracket_boundaries`'s
  noise-skip) is genuinely unreachable through `chunk_song`'s public path
  because `_strip_noise_brackets()` already removes noise brackets
  upstream of it, and `_pack_by_length`'s normal multi-line accumulate/
  flush loop needs multi-line input, not the single-overlong-line inputs
  the end-to-end force-split tests happened to all use.
- `tests/test_dedupe.py` -- direct unit coverage for `dedupe_blocks`:
  no-duplicates, exact-repeat, normalization-insensitive duplicate,
  empty input, first-occurrence-wins ordering.
- `tests/test_properties.py` -- hypothesis-based property tests
  (`@given` + `st.text`/`st.lists`), complementing the example-based
  tests above with randomized checks across many generated shapes:
  - RRF (`reciprocal_rank_fusion`): output ids == union of input ids, no
    duplicate output ids, single-list fusion preserves input order
    exactly, all scores strictly positive, and an idempotency property --
    fusing a list with an exact copy of itself doesn't reshuffle the
    relative order (every score simply doubles uniformly).
  - `chunk_song` idempotency: an already-minimal segment (short enough to
    survive level 4 untouched, no markup) chunks to exactly itself again
    when fed back in.
  - `dedupe_blocks` idempotency: running it again on exactly the texts it
    already kept keeps all of them and flags none as duplicates.
  - `normalize_for_dedupe` idempotency: applying it to its own output is
    a no-op.

**A genuine edge case surfaced by the hypothesis run itself:** the
`chunk_song` idempotency property's word-generating strategy assumed any
letters-only word list under `HARD_CEILING_WORDS` has "no markup" --
false for a 1-2 word list that happens to equal a plain-language section
label (e.g. `["BRIDGE"]`, or `["Pre", "Chorus"]`): per
`classify_plain_label_line`'s own definition that *is* a level-2 boundary,
and a label with nothing following it correctly collapses to zero
segments rather than reproducing as one. Not a `chunk_song` bug --
hypothesis's random shrinking found a real gap in the test's own
strategy. Fixed by filtering the strategy through
`classify_plain_label_line` itself (excluding word lists that would be
classified as a label), which is exactly the kind of gap property-based
testing is meant to catch that example-based tests wouldn't.

**Result:** `lyrics_search/core/` now at **100% line coverage** across all
six modules (`sections.py` 160/160, `dedupe.py` 13/13). 55 new tests
added; full suite (`pytest -q`, no `--cov`) at **170 passed, 2 skipped**
(the 2 skips are the pre-existing `pytest.importorskip("faiss")` tests,
environment-dependent, unrelated to this work) -- 172 total, matching the
pre-existing count exactly, no regressions.

**Correction, found while verifying §3 below:** the numbers directly
above were measured against the global `python`, not
`./.venv/Scripts/python.exe` -- exactly the environment mistake this
project's notes already record twice before (SPEC-02's ingest crash,
SPEC-03's CPU-instead-of-GPU latency mismeasurement), and this
session made it a third time when reinstalling `hypothesis`/`pytest-cov`/
`coverage` earlier in §1. Re-run through `.venv` (which does have faiss):
**172 passed, 0 skipped** -- the 2 "skips" were never a real
environment-dependent gap, just this session pointing at the wrong
interpreter. Core coverage is unaffected (100% either way; faiss isn't
part of `core/`).

## §3: synthetic demo corpus

**Goal:** a small, fully-synthetic, CPU-only corpus that anyone can build
and search in seconds, without downloading the real dataset, a model
checkpoint, or a GPU -- a "try it now" path distinct from `dev`/`full`.

**Corpus generation.** `scripts/generate_demo_corpus.py` fabricates 200
songs across 20 hand-written themes (Sunrise Valley, Midnight Highway,
Ocean Drift, Autumn Ember, Winter Hollow, Neon Skyline, Wildflower Field,
Silver Rain, Desert Mirage, Mountain Echo, Firelight Circle, Paper
Airplane, Electric Pulse, Quiet Harbor, Golden Hour, Stargazer, Broken
Compass, Velvet Curtain, River Bend, Glass House), 10 title/artist
variants each. Every theme carries its own bank of 6 verse lines, 6
chorus lines, and 4 bridge lines -- all originally composed for this
project, not derived from any real song -- sampled per song via a seeded
`random.Random(SEED)` for full reproducibility. Artist names are
combinatorially generated from adjective/noun pools (20 each) with
uniqueness enforced across the run. The script both generates the corpus
and runs it through the real build pipeline in one call, via an
in-memory `Source` implementation (`_InMemorySource`) so the exact same
`run_build()` code path used for `dev`/`full` is exercised -- no
demo-only ingestion logic to separately trust.

**Config/pipeline changes required.** `ExperimentConfig.corpus` was a
strict `Literal["dev", "full"]`; extended to
`Literal["dev", "full", "demo"]` in `lyrics_search/config.py`. Separately,
`build.py`'s `_ingest_stage_config()` special-cased `"full"` and fell
through to a `"dev"`-labeled stage_config (with dev's `sample_size`/`seed`
fields) for *anything else* -- a latent cache-key mislabeling bug that
would have silently mis-tagged a `"demo"` corpus's ingest cache entry as
if it were `dev`. Fixed by giving `"demo"` its own explicit branch
returning `{"corpus": "demo"}` (no sample_size/seed -- the demo set is a
fixed list, not a sample of anything).

**Where it lives on disk.** `configs/demo.yaml` pins
`corpus: demo, embedder: tfidf, index: numpy, retrieval.mode: hybrid` --
the same tfidf+BM25-hybrid combination already proven CPU-only/
no-download in `tests/test_end_to_end_search.py`. Building with
`--data-root data_demo` resolves the corpus directory to
`data_demo/demo/`, which lines up with the pre-existing `.gitignore`
exception `!data_demo/**/*.jsonl` (added before this work, presumably in
anticipation of exactly this).

**Git hygiene.** Checking `git add -n data_demo` before finalizing showed
more than intended would be staged: alongside the small `raw.jsonl`/
`chunks.jsonl`/`rejects.jsonl` text files, it also picked up
`fitted_state.joblib` (a 24 KB sklearn TF-IDF pickle) and duplicate
`chunk_ids.json`/`meta.json` files under both `embeddings/tfidf/` and
`indexes/tfidf/numpy/` -- only `vectors.npy` was already excluded by the
blanket `*.npy` rule. These are fully rebuildable in seconds by
re-running the generator script, so committing them would only ever add
stale/partial binary artifacts. Added `data_demo/**/embeddings/` and
`data_demo/**/indexes/` to `.gitignore`; re-checked with `git add -n
data_demo`, now exactly 5 files: `raw.jsonl`, `raw.jsonl.meta.json`,
`chunks.jsonl`, `chunks.jsonl.meta.json`, `rejects.jsonl` (empty -- 0
rejects out of 200 songs).

**Verification.** Ran the generator: 200 songs ingested (200 passed, 0
rejected) -> 500 chunks (all split on `bracket_tag`, 0 force_split) -> 0
rejected -> tfidf embed (vocab dim 647) -> numpy index built, all in
well under a second. Ran two real search queries against the built
corpus via `python -m lyrics_search.runners.search --config
configs/demo.yaml ... --data-root data_demo`: `"sunrise valley"` and
`"dancing electric pulse night"` -- both returned thematically correct,
relevant hits, dense and lexical branches both `ok`, load ~780-800ms,
cold_query ~2ms.

**New tests.** `tests/test_config.py`: `demo` accepted as a valid
`corpus` value, an invalid corpus string still rejected, and
`configs/demo.yaml` itself loads cleanly with the exact settings the
generator script builds against (catches config/script drift).
`tests/test_end_to_end_search.py`: a full build-then-search round trip
with `corpus="demo"` against the existing golden fixture (not the full
200-song generator, for speed), confirming the new Literal value
resolves and flows through the real pipeline end to end.

**Result:** full suite (`pytest -q`) at **174 passed, 2 skipped** through
the global interpreter used at the time -- corrected below.

**Correction:** those 2 skips were the wrong-interpreter artifact
described in §1's own correction paragraph above (global `python`
instead of `./.venv/Scripts/python.exe`), not a real
environment-dependent gap. Re-run through `.venv`: **176 passed, 0
skipped** (up from the corrected 172), no regressions.

## §4: bilingual README

`README.md` (English, default) and `README.uk.md` (Ukrainian), each
with a language-switch link at the top, mirroring `tfidf-sum`'s existing
bilingual convention (author's other public project) rather than
inventing a new one. Content: features, the demo-corpus quick start
(§3), architecture (offline stage-cached pipeline / online load-only
search), the `ExperimentConfig` YAML shape, a design-rationale section
incorporating `notes/decisions/retrieval-design.md`'s five items nearly
unchanged (that file's own stated purpose -- see its header), the real
full-corpus latency table from `notes/reports/spec03-patch-report.md`,
testing instructions, a data/copyright section explaining why `data/`
is never committed, and a status/roadmap pointer to `notes/STATUS.md`.
No license section added -- publishing (and any licensing decision that
goes with it) is explicitly the owner's own separate step (§6, not yet
authorized).

## §5: Definition of Done

Checklist compiled from this project's own standing rules and the SPEC-04
work above (no separate external DoD text was provided for this revision
beyond the owner's own step ordering) --
each item re-verified directly rather than assumed from earlier partial
runs:

- [x] **Full test suite green.** `./.venv/Scripts/python.exe -m pytest -q`
  -- **176 passed, 0 skipped.** (Confirms the §1/§3 wrong-interpreter
  correction above: the project's own `.venv` has faiss, so all faiss
  tests actually run rather than being silently skipped.)
- [x] **`core/` coverage.** `--cov=lyrics_search.core
  --cov-report=term-missing` -- **100% (250/250 statements)** across all
  six modules, re-confirmed through `.venv` alongside the full-suite run
  above.
- [x] **No copyrighted lyrics text in what would be committed.** §2's
  audit (above) covered `tests/golden/` and `notes/`; this pass adds the
  new demo corpus itself -- every line of `data_demo/demo/{raw,chunks}
  .jsonl` is `scripts/generate_demo_corpus.py`-fabricated text, not
  derived from any real song (see that script's own module docstring).
- [x] **`data/` never tracked.** `git ls-files | grep -E '^data/'` and
  `git ls-files | grep -E '\.npy$'` both empty (this project's standing
  pre-commit check, re-run here).
- [x] **`data_demo/` tracks only the intended flat files.**
  `git add -n data_demo` -- exactly `raw.jsonl`, `raw.jsonl.meta.json`,
  `chunks.jsonl`, `chunks.jsonl.meta.json`, `rejects.jsonl` (5 files);
  `embeddings/`/`indexes/` correctly excluded by this pass's `.gitignore`
  addition (§3 above).
- [x] **Bilingual README present, cross-linked, and accurate.**
  `README.md` / `README.uk.md` written (§4); their test-count claim was
  caught out of date by *this* checklist item (same wrong-interpreter
  numbers as §1/§3) and corrected to 176/0 before this checklist was
  considered satisfied.
- [x] **`notes/STATUS.md` reflects actual current state.** Updated: §0,
  notes refactor, §2, §1, §3, §4 all marked done with their real result
  numbers; only §5 (this section) and §6 (publish, explicitly deferred)
  remain.
- [x] **No stray working-tree artifacts.** `.coverage` (pytest-cov
  output) and `.pytest_cache/` were untracked and not previously
  gitignored -- added to `.gitignore` this pass. A `nul` file (a Windows
  side effect of an earlier misdirected shell command during this
  session, not project output) was found and deleted.
- [x] **No `--force` / destructive git operations used.** All work this
  pass is plain edits, new files, and (already-staged, from an earlier
  session) `git mv` renames -- nothing has been pushed, force-pushed, or
  amended.

**Not done, deliberately:** §6 (publish) -- per the owner's original
instruction and this project's own process rule ("nothing pushed before
SPEC-04's owner confirmation"), this stops here for review. The working
tree currently has everything needed for one SPEC-04 commit (see `git
status`); no commit has been made yet -- left for the owner to review
first, consistent with stopping at the end of every spec rather than
rolling straight into the next irreversible step.
