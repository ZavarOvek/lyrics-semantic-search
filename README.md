**English** | [Українська](README.uk.md)

[![CI](https://github.com/ZavarOvek/lyrics-semantic-search/actions/workflows/ci.yml/badge.svg)](https://github.com/ZavarOvek/lyrics-semantic-search/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

# lyrics-semantic-search

Semantic search over song lyrics: dense embedding search, lexical (BM25)
search, and a hybrid of the two fused by rank, not raw score. Offline
indexing pipeline through online query serving, built and measured end
to end on a real ~30k-song corpus.

## Features

- **Pluggable embedders** via a name -> factory registry: `bge-m3`
  (multilingual, `max_seq_length=8192`), `e5-base`, `fasttext-avg`
  (static vectors), `tfidf` (no model download at all) — swap by config,
  no code changes.
- **Two index backends**: `numpy` (exact brute-force cosine) and `faiss`.
- **Three retrieval modes**: `dense`, `lexical`, `hybrid` — hybrid fuses
  the two branches with Reciprocal Rank Fusion (RRF), not a weighted
  score sum (see [Design rationale](#design-rationale)).
- **Cached offline pipeline**: `ingest -> preprocess -> embed -> index`,
  each stage skipped when already fresh for its config, chained cache
  keys so an upstream change invalidates everything downstream, `--force`
  to bypass.
- **Functional core, imperative shell**: text transformation logic is
  pure functions with no I/O (`core/`); file/GPU/network access lives
  only in `runners/`. Cheap, fast, deterministic unit testing follows
  directly from that split.
- **No filter drops data silently** — every rejected song/chunk is
  recorded in `rejects.jsonl` with a reason from a closed enum.
- **100% test coverage in `core/`**, including hypothesis-based property
  tests (RRF algebraic properties, idempotency of chunking/dedup/
  normalization) alongside example-based unit tests — 268 tests total.
- **Fully synthetic demo corpus** — 200 originally-composed songs, no
  download, no GPU, builds and searches in seconds on any machine.

## Quick start (synthetic demo corpus)

No real data, no model download, no GPU required — `tfidf` + `numpy` +
`hybrid`, CPU-only:

```bash
pip install -r requirements.txt

# Generates 200 synthetic songs and runs the full build pipeline
# (writes into data_demo/demo/):
python scripts/generate_demo_corpus.py

# Search it:
python -m lyrics_search.runners.search --config configs/demo.yaml \
  "sunrise valley" --data-root data_demo
```

See `configs/demo.yaml` and `scripts/generate_demo_corpus.py` for detail;
`notes/reports/spec04-report.md` §3 for the design writeup.

## Full corpus setup

Only needed for the real `dev`/`full` corpus, the neural embedders
(`bge-m3`, `e5-base`, `fasttext-avg`), or the `faiss` index backend — the
demo above needs none of this. PyPI's default `torch` wheel may not match
your CUDA version, so install it first from the PyTorch index for your
platform/CUDA, then the rest:

```bash
pip install "torch>=2.6" --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-full.txt
```

## Architecture

**Offline (build):** `ingest -> preprocess -> embed -> index`, one config
in, a corpus directory out (`<data_root>/<corpus>/`). Each stage is
skipped when its on-disk artifact is already fresh for the current
stage config; a stage's `input_hash` is the *previous* stage's own cache
key, so re-running an earlier stage invalidates everything after it even
when later stages' own configs didn't change. `--force` bypasses every
freshness check unconditionally — the only way to recover from a stage's
*code* having changed without any config field changing.

**Online (search):** strictly query-time. Every piece of state (fitted
embedder, index, chunk/song lookups) is *loaded*, never fit or built —
the online branch checks `fitted_state_sha1` and `dim` against the
offline branch's own `meta.json` rather than trusting blindly. A warm-up
`encode()` call runs once at load time in both branches, since the first
real call otherwise costs ~476ms instead of the ~40ms every call after it
does.

## Configuration

One YAML file fully describes one run (`ExperimentConfig`, validated by
pydantic, `extra="forbid"` so a typo fails loudly):

```yaml
corpus: dev              # dev | full | demo
chunking: sections        # sections | whole_song
embedder: bge-m3          # plain name, or name + params:
# embedder:
#   name: fasttext-avg
#   params:
#     vectors_path: data/models/fasttext/cc.en.300.vec
index: numpy              # numpy | faiss
retrieval:
  mode: hybrid             # dense | lexical | hybrid
  rrf_k: 60
  top_k: 50
  return_n: 10
```

`load_config()` additionally cross-checks `embedder.name`/`index` against
the registry's currently-registered names, and `embedder.params` against
that embedder's declared required params — so an unregistered name or a
missing required param (e.g. `fasttext-avg`'s `vectors_path`) fails at
config-load time, not deep inside a build/search run. See `configs/*.yaml`
for real examples across every embedder/index/mode combination.

`chunking` is also a path component, so the arms' artifacts sit side by
side instead of evicting each other. `raw.jsonl` stays at the corpus root
because ingest is the one stage that does not depend on chunking:

```
data/<corpus>/raw.jsonl
data/<corpus>/<chunking>/chunks.jsonl
data/<corpus>/<chunking>/embeddings/<embedder>/
data/<corpus>/<chunking>/indexes/<embedder>/<index>/
```

`whole_song` emits exactly one chunk per song. It is only usable with
`bge-m3`, whose real `max_seq_length` is 8192 — no other embedder here
can hold a whole song in one vector.

## Evaluation

A config plus an eval set in, a metrics table out — online only, through
exactly the code path `search` uses, one query at a time:

```bash
python -m lyrics_search.runners.eval --config configs/full_dense_faiss.yaml \
  --eval-set data/eval/queries.jsonl [--json-out results.json]
```

The eval set is JSONL, one record per query. Records carrying a `_meta`
key are skipped rather than scored. A `relevant_song_id` that is not in
the corpus fails the load, naming the id and its line — it could never be
retrieved, so leaving it in would be indistinguishable in the output from
a retrieval miss.

```json
{"query": "...", "relevant_song_ids": ["a1b2c3d4e5f6a7b8"], "query_type": "..."}
```

Recall@{1,5,10}, MRR and nDCG@10 (`core/metrics.py`, pure functions) are
reported overall and per slice, each with `n` and a bootstrap confidence
interval — a small slice cannot show a significant difference, and the
interval is what makes that visible in the table rather than afterwards.
Slices are cut on properties of a query's *relevant* songs, never of what
came back, so two retrievers are always scored on the same partition:
`split_by` × `force_split`, `is_translation`, `query_type`, and genre. A
query whose relevant songs disagree on a property goes to a `mixed`
bucket instead of being assigned to either side or dropped, so every
dimension's `n` values still sum to the total.

Genre comes from `data/<corpus>/genre.jsonl`, built once by
`scripts/build_genre_lookup.py` and read by nothing else in the pipeline
— it is analysis metadata, not an input to retrieval. The slice is
skipped with a printed note when the file is absent. The value is the
source dataset's own `tag` column, not a classification made here.

<a id="design-rationale"></a>

## Design rationale

Full detail and enforcement/test pointers for each item: `notes/decisions/retrieval-design.md`.

**1. Fusion: RRF, not weighted score summation.** Cosine similarity
(dense) and BM25 (lexical) live on incomparable scales that can't be
correctly combined without per-corpus calibration — BM25 is unbounded and
corpus-size-dependent, cosine is bounded in [-1, 1]. Any fixed weighting
would silently favor whichever branch produces larger raw numbers on a
given corpus, not whichever is actually more relevant. RRF instead fuses
on *rank*, which is scale-invariant by construction:

```
score(d) = sum over ranked lists r that contain d of 1 / (k + rank_r(d))
```

**2. Dense on dedup'd chunks, lexical on whole `text_raw`.** Deliberately
different granularity, not an oversight: a literal keyword match anywhere
in a song is a meaningful lexical signal even if the matching passage got
dropped by within-song dedup at chunk level, so re-chunking for BM25
would throw away real recall for no benefit. Rebuilding BM25 from raw
text at load time is cheap and deterministic — no held-out data, nothing
to overfit.

**3. Chunk-to-song aggregation by MAX, not MEAN.** A mean would punish a
long song for a few irrelevant verses even when one chunk is a strong
match — e.g. a 6-chunk song with one chunk at 0.9 and five near 0 would
mean-average to ~0.15 and lose to a short, mediocre-but-uniform 1-chunk
song, even though it contains the single best-matching passage in the
corpus. Max preserves "does this song contain a strong match anywhere,"
which is the actual retrieval question.

**4. Translations flagged, not filtered.** Likely lyric-translation pages
are detected and recorded (`is_translation`) but never rejected — a
translation is still a valid, findable set of lyrics for a user who wants
it, and flagging rather than deleting preserves that choice for
downstream consumers without silently shrinking the corpus.

**5. Cross-song exact-lyrics duplicates not merged.** Songs sharing
byte-identical lyrics under different artist/title metadata (covers,
live takes, acoustic versions — 19 groups / 38 records, 0.13% of the full
corpus, all manually inspected and confirmed genuine) are flagged for
visibility but not merged: someone searching for a specific cover may
legitimately want it to surface separately from the original, not
silently collapsed into one result.

## faiss on Windows: non-ASCII paths

faiss's own `write_index()`/`read_index()` call `fopen()` directly on
the given path. On Windows, that file-open path doesn't reliably handle
non-ASCII bytes — any checkout path containing non-Latin characters
(which is how this was found) causes silent failures or an opaque C++
`RuntimeError`. faiss's own path handling can't be corrected from
Python; it can only be avoided.

Worked around in `lyrics_search/indexes/faiss_index.py` by never
calling faiss's own path-based save/load: `serialize_index()`/
`deserialize_index()` convert the index to/from an in-memory `bytes`
buffer entirely inside faiss's C++ layer (no filesystem access on
faiss's side at all), and plain Python `open()` — which handles Unicode
paths correctly on Windows — does the actual file I/O. Full writeup:
`notes/decisions/faiss-nonascii-path-windows.md`.

## Performance

**Offline (full-corpus build, `bge-m3`, RTX 3050 Laptop 4GB VRAM, fp16,
batch 16):** 30,000 songs -> 208,411 chunks generated, 181,471 kept
after preprocessing. Embedding: ~14 min under clean conditions
(848–860s); a run contended with other GPU workloads measured 23 min
(1378s). Peak VRAM 1388MiB. Result: 355MB of vectors on disk
(181,471 x 1024, float16).

**Online (query, after `warmup()`), same hardware, `numpy` index:**

| Stage | Time |
|---|---|
| Model load (incl. CUDA/cuDNN warmup) | 9.98s |
| Mean first-pass query (model loaded+warmed, 5 distinct queries) | 53.2ms |
| Mean repeat-query latency | 55.6ms |

The <100ms repeat-query target is met with headroom (retrieval over
181k x 1024 float32 vectors: ~32ms; model `encode()` on GPU: ~21ms).
Query latency is sensitive to GPU contention from other applications on
shared laptop hardware — see `notes/reports/spec03-patch-report.md` for
the full measurement writeup and caveats.

## Testing

```bash
pip install -r requirements.txt
pytest -q                                                    # full suite
pytest -q --cov=lyrics_search.core --cov-report=term-missing # core/ coverage
```

268 tests total. With the base `requirements.txt`, 261 pass and 5 skip:
3 require `faiss-cpu` (from `requirements-full.txt`); the other 2 require
a real `data/dev` corpus build plus the fasttext vectors file — real data
artifacts intentionally excluded from git (see
[Data & copyright](#data--copyright)), so those 2 skip on any fresh clone
regardless of which requirements file is installed. `lyrics_search/core/`
— the pure-function transformation layer — is at 100% line coverage in
both cases, including `hypothesis`-based property tests for RRF's
algebraic properties and idempotency of chunking/dedup/text
normalization.

## Data & copyright

Song lyrics are copyrighted text. `data/` (the real dev/full corpora) is
never committed — only code, configs, metadata, and the fully synthetic
demo corpus are public. The `.gitignore` exceptions are narrow and
explicit: `data_demo/**/*.jsonl` (originally-composed demo lyrics only)
and `tests/golden/**/*.jsonl` (short synthetic fixtures, not real lyrics).

The real corpus is built at runtime from
[`mrYou/lyrics-dataset`](https://huggingface.co/datasets/mrYou/lyrics-dataset)
on the Hugging Face Hub (`lyrics_search/sources/hf_dataset.py`), which
this project neither redistributes nor vendors — `ingest` downloads it
into the user's own HF cache. Check that dataset's own terms before
using it; the lyrics in it belong to their respective rights holders,
not to this project or to the dataset's uploader. The genre labels used
for eval slicing are that dataset's own `tag` column, not a
classification produced here.

## License

MIT — see `LICENSE`. The license covers the code in this repository; it
grants no rights to any song lyrics, which remain the property of their
respective copyright holders and are not included here (see
[Data & copyright](#data--copyright) above).

## Status

See `notes/STATUS.md` for the canonical, current phase-by-phase status
and `notes/reports/` for the full chronological record of what was done
and why. The eval phase (query set design, relevance criteria) is
intentionally not started — that scheme is set by the project owner, not
assumed by the implementation.
