# Retrieval design decisions

Consolidated design rationale for the retrieval/aggregation layer, gathered
from scattered docstrings and phase reports into one place (SPEC-04 notes
refactor). The README carries a condensed version of this material for
readers; this file holds the full rationale -- each item below states the
decision, the concrete justification, and where the decision is enforced in
code and covered by tests.

## 1. Fusion: RRF, not weighted score summation

**Decision:** hybrid search combines the dense and lexical branches with
Reciprocal Rank Fusion (`core/fusion.py`), not a weighted sum of raw scores.

**Why:** cosine similarity (dense) and BM25 (lexical) live on incomparable
scales that can't be correctly combined without per-corpus calibration --
BM25 scores are unbounded and corpus-size-dependent, cosine similarity is
bounded in [-1, 1]. Any fixed weighting would silently favor whichever
branch happens to produce larger raw numbers on a given corpus, not
whichever branch is actually more relevant. RRF instead fuses on *rank*,
which is scale-invariant by construction:

```
score(d) = sum over ranked lists r that contain d of 1 / (k + rank_r(d))
```

with 1-based ranks and `k` (default 60) damping the influence of low ranks.

**Enforced by:** `core/fusion.py` (`reciprocal_rank_fusion`), called from
`retrievers/hybrid.py`. Tested in `tests/test_fusion.py` (score formula) and
`tests/test_hybrid_retriever.py` (fused ranking behavior, including the
partial-degradation case where one branch goes out-of-vocabulary and the
other doesn't -- see item 4 of `notes/reports/spec03-report.md`).

## 2. Dense on dedup'd chunks, lexical on whole `text_raw`

**Decision:** `DenseRetriever` searches the deduplicated, chunked index
(`chunks.jsonl` -> embeddings); `LexicalRetriever` searches whole-song
`text_raw` reconstructed at query time from `raw.jsonl`, rebuilding its BM25
index on load rather than persisting one. Deliberately different granularity
between the two branches, not an oversight.

**Why:** a literal keyword match anywhere in a song is a meaningful lexical
signal even if the matching passage happened to get dropped by within-song
dedup at chunk level (`core/dedupe.py`), so re-chunking `text_raw` for BM25
would throw away real recall for no benefit. Rebuilding BM25 from raw text at
load time is cheap and deterministic (no held-out data, no randomness,
nothing to overfit): tokenizing + `BM25Okapi` construction together measured
~0.14s on the 500-song dev corpus, a few seconds extrapolated to the ~30k
full corpus -- well within the "build once at process start" budget, so
there's no need to persist a separate lexical index artifact.

**Enforced by:** `retrievers/lexical.py` (song-level `text_raw` + BM25,
rebuilt from `raw.jsonl`), `retrievers/dense.py` (chunk-level, from the
persisted embedding index). `runners/search.py:build_retriever()` wires each
branch to its own data source (`raw_path` for lexical, `embeddings_dir` +
`index_dir` for dense).

## 3. Chunk-to-song aggregation by MAX, not MEAN

**Decision:** when a song has multiple matching chunks, its song-level score
is the **maximum** of its chunk scores, not the mean. The chunk that produced
the max becomes `Hit.best_chunk` -- the fragment shown to the user as the
reason for the match.

**Why:** a mean would punish a long song for having a few irrelevant verses
even when one chunk is a strong match -- e.g. a 6-chunk song with one
chunk scoring 0.9 and five scoring near 0 would mean-average to ~0.15 and
lose to a short, mediocre-but-uniform 1-chunk song, even though the long
song contains the single best-matching passage in the corpus. Max preserves
"does this song contain a strong match anywhere," which is the actual
retrieval question. The function must also degenerate correctly when every
song has exactly one chunk (relevant because bge-m3's real
`max_seq_length` is 8192, SPEC-02-PATCH, making "whole song as one chunk" a
real config for the eval phase to compare, not just a theoretical edge case)
-- max-aggregation needs no special-casing to support that.

**Enforced by:** `core/aggregate.py`. Tested in `tests/test_aggregate.py`:
`test_max_score_wins_not_mean` (0.9-and-0.1-scoring chunks -> song scores
0.9, not 0.5) and `test_one_chunk_per_song_degenerate_case`.

## 4. Translations flagged, not filtered

**Decision:** `runners/preprocess.py` detects likely lyric-translation pages
(`detect_is_translation()`, a case-insensitive `\btranslations?\b` match
against artist OR title -- catches both Genius's pseudo-artist pattern
like "Genius English Translations" and title suffixes like "(... English
Translation)") and records `is_translation` on the song, but this flag is
**never used to reject or exclude the song** -- it stays in the corpus and
is searchable like any other song.

**Why:** language-ID-based filtering was evaluated separately (see
`notes/decisions/lang-check-decision.md`) and rejected as
unreliable at the point songs are otherwise short enough to be borderline --
by the time a song survives the word-count filter, it already carries enough
signal for near-100%-confident language ID, so there's no useful gap for a
language filter to close. Excluding translation pages outright would also be
a judgment call this project doesn't need to make: a translation is still a
valid, findable set of lyrics-in-English for a user who wants it, and
flagging (rather than deleting) preserves that choice for downstream
consumers/eval without silently shrinking the corpus. Counts are reported
purely as an informational metric (6/500 dev, 139/30000 full --
`notes/reports/spec02-report.md`), never as rejections.

**Enforced by:** `runners/preprocess.py` (`detect_is_translation()`,
`_TRANSLATION_RE`); `_song_reject_reason()` in the same file has no
translation case, confirming the flag never triggers a rejection.

## 5. Cross-song exact-lyrics duplicates not merged

**Decision:** songs sharing byte-identical `text_raw` under *different*
artist/title metadata (covers, live takes, acoustic versions, remixes, work
tapes -- 19 groups / 38 records, 0.13% of the full 30k corpus) are flagged
for visibility but not merged, deduplicated, or removed.

**Why:** this project's within-song dedup (`core/dedupe.py`) targets
duplicate *content blocks inside a single song* (repeated choruses etc.),
which is a distinct concern from *cross-song* identical lyrics under
different credited artists. All 19 groups were manually inspected and
confirmed to be genuine alternate versions, not ingestion bugs -- and
merging them would actively remove information a user might want: someone
searching for "Rolling in the Deep" by Emilie Hart may legitimately want
that cover to surface separately from the Adele original, not silently
collapsed into one result. The dedup check itself, and the full list of
inspected groups, is preserved in `notes/reports/spec02-report.md` Appendix
A (originally a standalone finding, merged into the phase report during the
SPEC-04 notes refactor).

**Enforced by:** no pipeline change was made (deliberately) -- this is a
documented non-decision. The check that produced the finding is
`scripts/dedup_check_full.py`, which runs against a real
`data/full/raw.jsonl`.
