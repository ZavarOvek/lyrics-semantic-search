# SPEC-02-PATCH item 1: language-confidence filter -- decision: explicitly cut

## SPEC-03 §0.2 addendum

Two clarifications added on re-review, without changing the underlying
decision:

1. **Scope of this decision.** Everything below is scoped specifically to
   the current English-only corpus (`mrYou/lyrics-dataset` as ingested for
   SPEC-02). If a Ukrainian (or any other non-English) branch is added later
   -- e.g. via `GeniusScraperSource` pointed at a different catalogue --
   language-ID would become genuinely necessary again: the whole argument
   for cutting it rests on "every surviving song is already
   near-certainly English", which stops holding the moment a second
   language is deliberately mixed into the corpus.

2. **Evidentiary framing was overstated.** The original "0/500 and 0/30000
   below 0.5 confidence" framing reads as strong evidence of a
   well-calibrated, discriminating classifier. It is not: 29961/30000
   (99.87%) of songs scored *exactly* 1.0. A classifier that saturates at
   its ceiling for effectively the whole corpus is not finely discriminating
   between "confidently English" and "less confidently English" -- its
   softmax is pinned at the boundary, so the metric carries much less
   information than "0 below 0.5" suggests on its own. The conclusion itself
   (cut the filter, it would reject 0 songs at any legitimate threshold)
   still holds -- it's supported by the hand-inspection of all 13 sub-0.9
   songs, not by the confidence distribution's shape -- but the confidence
   numbers alone should not be read as proof of a meaningfully discriminating
   signal.

## Question

SPEC-02 §4.5/4.6 called for a song-level language-ID sanity check (`fasttext
lid.176`, or a fallback like `lingua`/`py3langid`), producing a
`song_low_lang_confidence` reject reason. The original implementation
skipped this silently, relying only on the title/artist-based
`is_translation` heuristic. This patch item requires either implementing it
for real, or explicitly cutting it with a documented, evidence-based
reason -- not leaving it silently absent.

## What was tried

Installed `py3langid` (pure-Python, MIT-licensed fork of `langid.py`, model
bundled in the wheel -- no separate large model download needed, unlike
`fasttext`'s `lid.176.bin` which would have required asking permission for
a ~126MB download per SPEC-00's file-download rule).

Configured `LanguageIdentifier.from_pickled_model(MODEL_FILE,
norm_probs=True)` to get a proper normalized confidence in [0, 1] rather
than langid's default raw (unnormalized, unbounded-negative) log-likelihood
score, then ran it over `text_raw` for every song in both corpora as a
calibration exercise (`/tmp/langid_calib.py`, `/tmp/langid_calib_full.py`,
`/tmp/langid_check_09.py` -- one-off scripts, not committed, reproducible
from this description).

## Result

**Dev corpus (500 songs):** confidence >= 0.685 on every single song. 0
songs below 0.5. p50 = p25 = p10 = p5 = p1 = **1.0000**.

**Full corpus (30000 songs):** confidence >= 0.503 on every single song. 0
songs below 0.5. 29961/30000 (99.87%) classified at exactly 1.0 confidence.
Only 13 songs fell below 0.9.

All 13 sub-0.9 songs were inspected by hand. Every one is genuine,
legitimate English-language lyrics -- no garbled text, no encoding
corruption, no mixed scripts. The lowest-confidence song in the entire 30k
corpus (`The Blaze -- "Queens"`, conf=0.503) is ordinary English lyrics.
The classifier is just mildly less certain on short, heavily repetitive,
or all-caps stylized text -- not on genuinely bad data.

## Conclusion: cut, not implemented

Song-level language-ID confidence provides **zero discriminative signal**
on this corpus, for a structural reason, not a tuning problem: by the time
a song has enough words to pass the existing `MIN_SONG_WORDS=10` filter, it
already has enough signal for `py3langid` to classify it with near-total
confidence. The genuinely degenerate inputs a confidence check would be
meant to catch (empty pages, instrumental stubs, single-word noise) are
*already* caught earlier by the empty-text / instrumental-only /
too-short-word-count filters, which run first in `_song_reject_reason`.
There is no gap between "too short to filter on word count" and "long
enough that language-ID is unreliable" for this dataset -- confirmed
empirically at both 500- and 30000-song scale, not assumed.

Implementing `song_low_lang_confidence` as specified would add a reject
reason to the enum, a dependency, and ~20s of processing time on the full
corpus (measured), for a filter that -- per the calibration above -- would
reject **0 songs** at any threshold conservative enough to avoid false
positives on legitimate short/stylized/all-caps lyrics (the lowest
legitimate-song score observed, 0.503, would need the threshold set so low
it stops meaning anything).

This matches and confirms, with real numbers instead of assertion, the
reasoning already present in the original SPEC-02: pollution in this
dataset is semantic (wrong song matched to a title, mislabeled covers,
credits-only pages) rather than linguistic. `is_translation` (title/artist
pattern matching) already handles the one linguistic case that does occur
at meaningful volume in this corpus -- explicit translation pages -- more
precisely and far more cheaply than a general-purpose language-ID model
would.

**Action taken:** none in `runners/preprocess.py` -- no
`song_low_lang_confidence` reason added, no py3langid call added to the
pipeline. `py3langid` remains a dev-only dependency (added to
`requirements.txt` for reproducibility of this calibration, not imported by
any runner).
