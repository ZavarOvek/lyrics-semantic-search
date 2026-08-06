# SPEC-02 completion report: offline pipeline

Implements the ingest -> preprocess -> embed -> index pipeline as a proper
package (`lyrics_search/`), replacing the phase0.py vertical slice. Pure
logic lives in `core/` (zero I/O, zero heavy deps); all I/O is isolated in
`runners/`. Everything below was run for real, twice: once on a 500-song
dev sample (fast iteration/debugging) and once on the full 30000-song
corpus (final numbers). Code is unchanged between the two runs.

## Package layout

```
lyrics_search/
  contracts.py            RawSong, Chunk, Hit, Reject + Source/Embedder/Retriever Protocols
  core/                    pure functions, no I/O
    text.py                normalization, song_id, block-hash helpers
    sections.py             4-level chunking cascade
    dedupe.py                duplicate-block detection within a song
    fusion.py                 stub (SPEC-03)
    rejects.py                 RejectReason enum + Reject dataclass
  sources/                 Source implementations
    hf_dataset.py            HFDatasetSource (mrYou/lyrics-dataset via HF Hub)
    jsonl_file.py             JsonlFileSource (golden fixtures / re-ingest)
    genius_scraper.py          GeniusScraperSource (stub, NotImplementedError)
  embedders/                Embedder implementations
    bge_m3.py, e5.py, tfidf.py, fasttext_avg.py
  indexes/                   Retriever implementations
    numpy_index.py            exact brute-force dot product
    faiss_index.py             approximate HNSW (faiss)
  runners/                    imperative shell, all I/O
    ingest.py, preprocess.py, embed.py
```

## 1. Ingest

`HFDatasetSource` downloads the dataset parquet via `hf_hub_download`
(cached under the user's HF hub cache), reads it with pandas, and -- this is
the one deliberate deviation from a naive "just iterate the dataframe"
approach -- **sorts by the dataset's own `id` column after sampling**, so
that repeated runs with the same `sample_size`/`seed` produce byte-identical
`raw.jsonl` regardless of any internal non-determinism in pandas' sampling
order. Verified by diffing two consecutive dev-corpus runs: identical.

`run_ingest` dedupes on `song_id` (first occurrence wins) and prints a loud
WARNING for every collision skipped, rather than hard-failing the whole run
-- true (artist, title) collisions are plausible at 30k scale and shouldn't
abort ingestion of the other 29999 songs.

| | dev (500) | full (30000) |
|---|---|---|
| songs ingested | 500 | 30000 |
| song_id collisions (skipped) | 0 | 0 |
| `raw.jsonl` size | ~1.2M | 48M |
| wall time | ~2s | 3.5s (parquet already HF-cached) |

## 2. Preprocess

`run_preprocess`: detects `is_translation` per song (regex `\btranslations?\b`
against artist OR title, catches both Genius's "... Genius English
Translations" pseudo-artist and "... English Translation" title suffixes),
applies a song-level filter (empty text / instrumental-only / <10 words),
then for surviving songs runs the 4-level chunking cascade
(`core/sections.py`) followed by within-song duplicate-block removal
(`core/dedupe.py`), then a chunk-level filter (<15 chars survives as
`chunk_too_short`).

Two count invariants are asserted at runtime (fail loudly, not silently,
per SPEC-00 §3.2):
```python
assert len(passed_song_ids) + len(song_rejects) == len(songs)
assert len(surviving_chunks) + len(chunk_rejects) == pre_filter_chunk_total
```
Both held on every run, dev and full.

| | dev (500) | full (30000) |
|---|---|---|
| songs passed / rejected | 500 / 0 | 30000 / 0 |
| `is_translation` | 6 (1.2%) | 139 (0.46%) |
| chunks generated (pre-filter) | 3408 | 208411 |
| chunks kept | 2951 | 181471 |
| chunks rejected: `chunk_duplicate_block` | 404 | 23296 |
| chunks rejected: `chunk_too_short` | 53 | 3644 |
| split_by: `bracket_tag` | 1343 | 79912 |
| split_by: `blank_line` | 1491 | 93262 |
| split_by: `plain_label` | 41 | 4262 |
| split_by: `length` (level-4 forced) | 533 | 30975 |
| length-ceiling WARNINGs (>120 words, force-split) | 175 | 10289 |
| wall time | ~4s | 20.4s |

The dev-corpus `is_translation` count (6/500) matches phase0's independently
derived manual finding from SPEC-01 exactly -- good cross-validation that
the regex heuristic is sound rather than coincidentally right on a small
sample. Full-corpus rate (0.46%) is proportionally lower but plausible: the
dev sample happened to catch a slightly translation-heavy random draw.

Both reject categories were spot-checked by hand at dev scale in the
previous session (duplicate_block catches genuinely repeated
choruses/blocks; too_short catches ad-libs, producer credits and stray
section labels) and the full-corpus rejection *rates* per category are
close to proportional to the dev-corpus rates (`duplicate_block`:`too_short` ratio
88:12 dev vs. 86:14 full), consistent with the same logic operating
correctly at both scales.

## 3. Full-corpus dedup check

See Appendix A below for detail. Summary: **0**
`song_id` / normalized-(artist,title) collisions at 30k scale (matches
ingest's own zero-WARNING result); **19 groups / 38 records (0.13%)** of
byte-identical lyrics text hiding under different artist/title metadata --
all manually confirmed as genuine alternate versions (covers, live takes,
acoustic/reimagined versions, work tapes, remixes), not ingestion bugs. No
pipeline change made; this is out of scope for SPEC-02's within-song
dedup and arguably shouldn't be auto-merged (a search user may want a cover
to surface separately from the original).

## 4. Embed

Four embedders behind one `Embedder` protocol; `TfidfEmbedder` and
`FastTextAvgEmbedder` additionally expose `fit(corpus_texts)`, called by
`run_embed` before `encode()` when present (documented deviation from the
otherwise-stateless embedder shape). All vectors L2-normalized and saved as
float16.

**Bug found and fixed this stage** (see git history): `peak_vram_mb` was
leaking the *previous* CUDA embedder's stale peak-memory-allocated stat
into the next, CPU-only embedder's `meta.json` (tfidf initially reported
1627 MiB despite touching no GPU). Fixed by gating the stat behind
`getattr(embedder, "device", None) == "cuda"` and explicitly freeing GPU
memory (`del embedder; torch.cuda.empty_cache()`) between embedders in the
runner's CLI loop.

Dev corpus (2951 chunks), full corpus (181471 chunks, bge-m3 only run per
SPEC-02's minimum requirement -- e5-base/tfidf/fasttext-avg were validated
at dev scale only, all four share the same runner code path so the
bge-m3 full run stands as the scale proof):

| embedder | dim | dev: time / peak VRAM / norm_ok | full: time / peak VRAM / norm_ok |
|---|---|---|---|
| bge-m3 | 1024 | 14.1s / 1172 MiB / True | 860.3s (14.3 min) / 1388 MiB / True |
| e5-base | 768 | 5.3s / 606 MiB / True | -- (not re-run at full scale) |
| tfidf | 9079 (fit-dependent) | 0.2s / n/a / False* | -- |
| fasttext-avg | 300 | 0.35s (encode only) / n/a / False* | -- |

\* tfidf and fasttext-avg both show a small number of zero-norm vectors
(3/2951 and 2/2951 respectively) on degenerate inputs (acronym-spelled
text with no 2+ letter tokens, dash-only lines, OOV-only proper nouns).
Investigated by hand and confirmed genuine zero-signal inputs for these
model types, not a bug -- documented rather than "fixed".

Peak VRAM (1388 MiB at full scale) stays comfortably inside the 4GB RTX
3050 budget with ample headroom, at batch_size=16 with fp16 weights.

`data/full/embeddings/bge-m3/vectors.npy` is 355M (181471 x 1024 float16).

## 5. Index

`NumpyIndex` (exact, O(n) dot product) and `FaissIndex` (`IndexHNSWFlat`,
`METRIC_INNER_PRODUCT`, m=32/ef_construction=200/ef_search=64) built and
compared against each other on real bge-m3 embeddings, at both scales:

**Dev scale (2951 vectors, 20 queries: 10 thematic + 10 paraphrased
quotes):** top-1 identical on all 20; mean top-10 overlap 9.90/10; 18/20
queries with a perfect top-10 match (other 2 differ only at rank 9-10).

**Full scale (181471 vectors, 5 sanity queries):** top-1 and full top-10
identical on all 5 (mean overlap 10.00/10). Build time: NumpyIndex ~0s
(just stores the array), FaissIndex ~187s (one-time HNSW graph
construction). Search time: NumpyIndex ~29.8ms/query, FaissIndex
~1.2ms/query -- a ~25x speedup that wasn't visible at dev scale (2951
vectors is small enough that brute force is already sub-millisecond-ish
and dominated by Python overhead). At 181k vectors the approximate index's
practical value becomes clear while search quality stays effectively
identical to exact search.

## Known limitations / deferred items

- Only bge-m3 was run end-to-end on the full 30k corpus per SPEC-02's
  stated minimum; e5-base/tfidf/fasttext-avg were validated at dev scale
  only. Re-running them at full scale is mechanical (same runner, same
  code path already proven correct at both scales for bge-m3); not done in
  this phase, to control run time and disk use.
- `GeniusScraperSource` remains a stub (`NotImplementedError`) -- SPEC-02
  didn't require a working scraper, only the interface + a documented plan
  for what a real implementation needs (robots.txt/rate-limit compliance,
  local caching, retry/backoff).
- Cross-song exact-lyrics duplicates (19 groups, 38 records) are flagged
  but not merged/removed -- see §3.
- `core/fusion.py` is still a stub, reserved for SPEC-03 (fused ranking
  across embedders).

## Appendix A: full-corpus (30k) dedup check

(notes refactor: merged in from the former standalone
`notes/spec02-full-corpus-dedup.md`, referenced from §3 above.)

Ran `scripts/dedup_check_full.py` against `data/full/raw.jsonl` (30000 songs,
ingested from the full `mrYou/lyrics-dataset` with no sampling).

### A.1-2. song_id / normalized (artist, title) collisions

**0 collisions** at 30k scale, either via `song_id`
(`sha1(norm(artist)|norm(title))[:16]`, contracts.py) or via an independent
recomputation of the same normalized key. The two checks agree exactly, as
they should since they're the same normalization applied twice.

This matches what the ingest run itself already implied: `run_ingest` logs a
WARNING for every song_id collision it skips, and the full-corpus ingest run
printed none. The dev-corpus run (500 songs) also showed 0 collisions but
"proves nothing" at that scale per SPEC-02's own caution -- 30k confirms the
`song_id` scheme holds up at full scale too.

### A.3. Exact `text_raw` duplicates across different metadata

**19 groups, 38 records** (0.13% of the corpus) share byte-identical lyrics
text under *different* artist/title metadata, so they are invisible to the
song_id dedup (which only keys on normalized artist+title, not content).

Manually inspecting all 19 groups: every one is a genuine alternate
version/take of the same underlying song, not an ingestion bug or accidental
double-ingest:

- Studio vs. work-tape/demo: *Kacey Musgraves -- "Dandelion" / "Dandelion Work Tape"*
- Cover vs. original: *Go Radio "Rolling in the Deep" / Emilie Hart "...Cover"*
- Two different credited artists for a traditional/cover song: *Sturgill
  Simpson / Ralph Stanley -- "Poor Rambler"*
- a cappella vs. radio edit: *The Herbaliser -- "Verbal Anime ... a cappella"
  / "...radio"*
- Alternate title for the same recording: *Max Romeo -- "Words of Wisdom" /
  "Dont You Weep"*
- Remix pairs: *Mindless Self Indulgence*, *Vigiland*
- Spelling variant of the same title: *Louis Armstrong -- "...Round And
  Around" / "...Round And Round"*
- Scraper artifact in one artist field ("Die a Happy Man Lyrics - Thomas
  Rhett") vs. a genuine different-artist cover
- Live/session variants: *The Civil Wars*, *Abby Kasch / Jana Kramer*
- Acoustic / reimagined versions: *Levi Hummon*, *Caitlyn Smith*

### A.4. Conclusion

No action taken -- this is out of scope for SPEC-02's dedup requirement,
which targets duplicate *content blocks within a single song*
(`core/dedupe.py`, already exercised: 23296/26940 chunk-level rejects in the
full-corpus preprocess run were `chunk_duplicate_block`). Cross-song
identical-lyrics detection (covers/alternate takes) is a distinct, much
smaller concern (38/30000 records) and arguably shouldn't be silently
merged anyway -- a search user may legitimately want "Rolling in the Deep"
by Emilie Hart to surface separately from the Adele original. Flagging here
for visibility; no pipeline change made.
