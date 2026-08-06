# SPEC-02-PATCH completion report

Patch to SPEC-02 revision 2 (not a new phase), addressing all 7 items from
the patch spec. Code changed: `core/sections.py`, `core/rejects.py`,
`contracts.py`, `runners/preprocess.py`, `runners/embed.py`,
`embedders/tfidf.py`, `embedders/fasttext_avg.py`, `embedders/bge_m3.py`,
`embedders/e5.py`. New: `tests/test_preprocess_song_filter.py`,
`pyproject.toml` (pytest config), `notes/verify_fitted_state_persistence.py`
(historical name; promoted to a real pytest test and moved to
`tests/test_fitted_state_persistence.py` during the SPEC-04 notes refactor).

## Item 1: language-confidence filter -- decision: NOT implemented

Full reasoning and calibration data in
`notes/decisions/lang-check-decision.md`. Summary: calibrated
`py3langid`'s normalized confidence across both dev (500) and full (30000)
corpora. **0/500 and 0/30000** songs fall below 0.5 confidence; full-corpus
minimum is 0.503, and 29961/30000 songs score exactly 1.0. Manually
inspected all 13 full-corpus songs below 0.9 confidence -- every one is
genuine English lyrics (just short/stylistically unusual), zero garbled
content. Conclusion: by the time a song survives `MIN_SONG_WORDS=10`, it
already carries enough signal for near-100%-confident language ID; there is
no gap between "too short to filter on word count" and "long enough that
language-ID would be unreliable" in this dataset. A `song_low_lang_confidence`
filter would be dead code -- explicitly cut, not silently absent.
`py3langid` is recorded in `requirements.txt` as the tool used to reach this
decision, but is not imported by any runner.

## Item 2: song-level filter proven to execute

`tests/test_preprocess_song_filter.py`, 7 tests, all green (0.11s):
- 5 parametrized unit tests on `_song_reject_reason()` directly: empty text,
  whitespace-only, <10-word text, `[Instrumental]` bracket tag, bare
  "instrumental" text -- each asserted against its exact expected
  `RejectReason`.
- 1 unit test that a normal song returns `None` (survives).
- 1 integration test that calls the real `run_preprocess()` on a synthetic
  `raw.jsonl` mixing good and bad songs, and asserts every bad song lands in
  `rejects.jsonl` with `level="song"` and the correct `reason`, while the
  good song alone survives into `chunks.jsonl`.

This proves the filter is live code exercised end-to-end, not just that the
real corpora happened to be clean -- the "0 rejected at song level" result
on both dev and full corpora means the corpora are clean, not that the
filter is dead.

## Item 3: fitted embedder state persisted + verified cross-process

`TfidfEmbedder`/`FastTextAvgEmbedder` gained `save_fit()`/`load_fit()`
(joblib). `runners/embed.py` now calls `save_fit()` right after `fit()` and
records `fitted_state_file`/`fitted_state_sha1` in `meta.json`. `encode()`
on both embedders now **raises `RuntimeError`** if neither `fit()` nor
`load_fit()` was called -- this fixes a previously-silent bug in
`FastTextAvgEmbedder.encode()`, which used to fall back to an empty/flat IDF
weighting without ever erroring (SPEC-00 §3.2 violation, now fixed).

**Acceptance check** (`notes/verify_fitted_state_persistence.py` at the
time; now `tests/test_fitted_state_persistence.py`, see the SPEC-04 notes
refactor): encodes
the same fixed query in two genuinely separate Python subprocesses (each
constructs a fresh embedder and calls only `load_fit()`, never `fit()`),
for both fit-dependent embedders:
```
[tfidf] process A vector[:5]=[0. 0. 0. 0. 0.], process B vector[:5]=[0. 0. 0. 0. 0.], identical=True
[fasttext-avg] process A vector[:5]=[-0.00313209 ...], process B vector[:5]=[-0.00313209 ...], identical=True
PASS: both fit-dependent embedders produce identical vectors across separate processes.
```
(tfidf's query vector happens to be all-zero because none of the query's
words are in the fitted dev-corpus vocabulary -- expected, and irrelevant to
the check: the point is both processes agree, not that the vector is
non-trivial.)

## Item 4: zero-norm vectors excluded from the index

`run_embed()` now computes norms right after `encode()`, splits chunks into
survivors (norm > `ZERO_NORM_EPS=1e-6`) and zero-signal (norm <= eps), saves
only survivors to `vectors.npy`/`chunk_ids.json`, and logs the rest as
`chunk_no_signal` (new `RejectReason`, level=`chunk`) rows in a per-embedder
`<embeddings>/<name>/rejects.jsonl`.

Dev-corpus result (re-run with the patched code):

| embedder | dropped_no_signal | norm_check_passed |
|---|---|---|
| tfidf | 3 | **True** (was False before the patch) |
| fasttext-avg | 2 | **True** (was False before the patch) |
| bge-m3 | 0 | True |
| e5-base | 0 | True |

`np.allclose(norms, 1.0, atol=1e-2)` now passes cleanly for **all four**
embedders with no exception/footnote needed -- the same 3+2 chunks
previously footnoted as "confirmed genuine zero-signal, not a bug" are now
mechanically excluded and logged instead of silently sitting in
`vectors.npy` as zero vectors.

## Item 5: WARNING semantics retuned + real tokenizer overflow check

`runners/preprocess.py`'s summary no longer calls level-4 length-packing a
"WARNING" -- it's printed as an INFO-level `force_split events` count
(cascade doing its job, not a danger signal). Full-corpus count: **10289**
(unchanged from before, just reframed).

`runners/embed.py` now does the *real* check: for embedders exposing
`token_lengths()`/`max_seq_length` (bge-m3, e5-base -- both gained these
this patch), it tokenizes every surviving chunk without truncation and
compares against the model's actual `max_seq_length`, printing a genuine
`WARNING:` line only if anything would actually overflow. Discovered along
the way: **bge-m3's real `max_seq_length` is 8192** (long-document model,
not the previously-assumed 512), e5-base's is 512 (assumption held).

Full-corpus result: **`token_overflow_count` = 0** (`token_max_observed` =
832, `tokenizer_max_length` = 8192) -- zero real WARNINGs, exactly the
acceptance criterion. Both counts are now recorded in `meta.json` for every
neural-embedder run.

## Item 6: `split_by` / `force_split` split into two `Chunk` fields

`Segment`/`Chunk` now carry `split_by: "bracket_tag"|"plain_label"|
"blank_line"|"none"` (the structural boundary that produced the segment, or
"none" if none was found) and an independent `force_split: bool` (whether
level-4 length-packing was additionally applied on top). The old
`split_by="length"` overwrite is gone.

Full-corpus `split_by` distribution:
```
{'bracket_tag': 96348, 'blank_line': 99574, 'plain_label': 6089, 'none': 6400}
```
Full-corpus `split_by` x `force_split` cross-tab (pre-filter):
```
{'blank_line/force_split=False': 93262, 'blank_line/force_split=True': 6312,
 'bracket_tag/force_split=False': 79912, 'bracket_tag/force_split=True': 16436,
 'none/force_split=False': 1490, 'none/force_split=True': 4910,
 'plain_label/force_split=False': 4262, 'plain_label/force_split=True': 1827}
```
Verified this reconciles exactly with the pre-patch numbers on the dev
corpus by hand (old `split_by="length"` count == new `none/force_split=False`
+ sum of all `force_split=True` rows).

## Item 7: artist-field Genius-scraper pollution

Full detail in Appendix A below. **2/30000**
artist fields match the `"<Title> Lyrics - <Artist>"` scraper-swap pattern
(`Die a Happy Man Lyrics - Thomas Rhett`, `Forbes Lyrics - Borgore`) --
negligible, per the patch's own threshold. **No normalization, no dedup
re-run.** Neither song_id collides with the previously-found exact-text
duplicate groups.

## Full-corpus re-run (all patches applied)

Ran `preprocess` + `embed --embedders bge-m3` on the full 30000-song corpus
again with all patched code:

| | before patch | after patch |
|---|---|---|
| songs passed/rejected | 30000 / 0 | 30000 / 0 (unchanged) |
| chunks generated / kept | 208411 / 181471 | 208411 / 181471 (unchanged -- invariants still hold exactly) |
| bge-m3: count / norm_ok / dropped_no_signal | 181471 / True / (n/a) | 181471 / True / 0 |
| bge-m3: elapsed / peak VRAM | 860.3s / 1388 MiB | 848.0s / 1388 MiB (run-to-run variance only) |
| preprocess: length-related count | 10289, framed as "WARNING" | 10289 `force_split events`, framed as INFO (unchanged number, item 5) |
| bge-m3: real tokenizer-overflow WARNING | did not exist | **0** (real check added; max_seq_length=8192, max observed=832) |

Both §4.7 invariants (`songs passed + rejected == total`,
`chunks kept + rejected == pre-filter total`) still hold exactly at full
scale after all patch changes.

## Not in scope for this patch (unchanged)

- e5-base/tfidf/fasttext-avg full-scale (30k) runs -- deferred to eval phase
  as agreed; still only validated at dev scale (2951 chunks), now with the
  item-3/4 fixes applied and verified there.
- Cross-song exact-duplicate merging -- decision from SPEC-02 stands.
- `core/fusion.py` / online query path -- SPEC-03.

## Appendix A: artist-field Genius-scraper pollution check

(notes refactor: merged in from the former standalone
`notes/spec02-patch-item7-artist-pollution.md`, referenced from Item 7 above.)

### A.1 Method

Checked all 30000 `artist` fields in `data/full/raw.jsonl` against the exact
scraper-artifact pattern described in the patch: the whole Genius page-title
string (`"<Title> Lyrics - <Artist>"`) landing in the `artist` field instead
of just `<Artist>`.

```python
scraper_re = re.compile(r"(?i)^.*\blyrics\b\s*-\s*.+$")
```

### A.2 Result

**2 / 30000** artist fields match the pattern:

| song_id | artist (as scraped) | title |
|---|---|---|
| `4c8bcd11a105387d` | `Die a Happy Man Lyrics - Thomas Rhett` | `Die a Happy Man` |
| `d7cfaa3940192b5b` | `Forbes Lyrics - Borgore` | `Forbes` |

For context, two broader/looser checks were also run (not the pattern being
decided on, just to make sure nothing bigger was being missed by being too
strict):

- `artist` contains the bare word "lyrics" anywhere: 4/30000 — the 2 above,
  plus 2 unrelated legitimate artist names that happen to contain the word
  ("Everybody Hear's Lyrics Song", "Lyrics Of Two").
- `artist` contains the bare word "genius" anywhere: 149/30000 — almost all
  of these are real curator/channel pseudo-artists (`Rap Genius`, `Country
  Genius`, `Genius English Translations`, ...) or, in one case, a genuine
  artist name (`Perfume Genius`). This is a *different* kind of metadata
  noise (curator pages, not a scraper field-swap bug) and most of the
  translation-page instances are already caught by `detect_is_translation()`
  in `runners/preprocess.py` via the same "translation(s)" substring match
  documented in SPEC-02 §4.5. It is not the pattern item 7 asked about and
  is left out of scope here.

### A.3 Decision

2/30000 (0.007%) is well within "a handful" — the patch's own stated
threshold was that if only a handful turn up, they are left as they are and
the count is recorded, so **no normalization
of the artist field and no dedup re-run** is warranted. Left as-is,
documented here per the instruction to record the count either way.

Both `song_id`s were cross-checked against the exact-text-duplicate groups
found by `scripts/dedup_check_full.py`: neither appears in that list, so
this pollution is unrelated to the previously-found duplicate-text songs
and introduces no new song_id/dedup risk.
