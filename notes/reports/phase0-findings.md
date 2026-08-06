# Phase 0 findings

Corpus: 500 songs, seed=42, sampled from `mrYou/lyrics-dataset` (30k songs).

Dataset choice note: first candidate `brunokreiner/genius-lyrics` is listed and
readable via `dataset_info()`, but the actual CSV file returns
`401 Unauthorized` / `RepositoryNotFoundError` on download — file access seems
broken on the HF side despite public repo metadata. Skipped, used the second
candidate from the SPEC-01 list instead.

## Section tags: structure and variants

- 217 / 500 songs (43%) have **no bracketed section tags at all** — just raw
  text. Preprocessing can't rely on tags being present.
- Where tags exist, common types: `Verse`, `Chorus`, `Bridge`, `Hook`,
  `Outro`, `Pre-Chorus`, `Intro`, `Post-Chorus`, `Refrain`, `Interlude`,
  `Instrumental` — reasonably standardized.
- But also noise inside brackets: garbled placeholders (`[?]`, `[??]`),
  and speaker/producer credits used *instead of* a section type
  (e.g. `[Zaytoven]`, `[Blacky Bxans:]`, `[Lord Christy K:]`).
- Some songs use **plain-text, non-bracketed labels** for sections
  (`VERSE 1`, `Verse:`, `INTRO HOOK` as bare lines) — these are invisible to
  a `\[.*?\]` regex and survive into chunks as noise.

## Near-empty / garbage chunks

No full garbage *pages* (non-lyrics content) turned up in this sample, but
naive "strip tags + split on blank lines" produces **77 / 3054 chunks (2.5%)
under 15 characters** — mostly ad-libs, producer credits and stray section
labels of the kind described above. These carry no semantic content and
should be filtered by a minimum-length
threshold before embedding in SPEC-02, with the reason logged to
`rejects.jsonl` per SPEC-00 §3.3.

## Duplicates

Zero exact-text duplicates and zero duplicate `(title, artist)` pairs inside
this 500-song sample. Doesn't prove the full 30k corpus is dedup-clean —
just that collisions are rare enough not to show up at this sample size.
Needs an explicit dedup pass in SPEC-02 against the full corpus.

## Language admixture

`language` / `language_cld3` / `language_ft` metadata columns say `en` for
**100%** of the sample — the labels look pre-filtered/unreliable as a
contamination signal. Despite that, **6 / 500 (1.2%)** are explicit
"*... English Translation*" pages (Korean, Spanish, Japanese originals,
credited to "Genius English Translations"). Translated lyrics have
different phrasing statistics than native English lyrics and may embed
differently. Language metadata alone will not catch this class — title/artist
pattern matching (`contains 'Translation'`) is a cheap additional signal
worth adding in SPEC-02.

## Length distribution

- Full lyrics: 169–5397 chars, median 1209.
- Chunks (after tag-strip + blank-line split): 2–2841 chars, median 163,
  p95 674.
- **7 chunks exceed ~2000 chars (~500+ tokens)** — all from songs with *no
  blank lines at all* between verses, so the whole song collapses into one
  giant chunk instead of several. With `max_length=512` (SPEC-00 §3.2) these
  get silently truncated by the tokenizer rather than failing loudly.
  Blank-line splitting alone is not a robust chunking strategy for SPEC-02;
  needs a length-based fallback split for tag-less/blank-line-less songs.

## Other

- Genre balance in the sample is even: rock 131, country 124, rap 123,
  pop 122 — no genre skew to correct for at this scale.
- Rap/rock lyrics carry explicit language and slurs frequently. Not a
  processing problem, but a reminder to pick clean examples for any public
  README/demo screenshots (SPEC-00 §3.1: corpus itself never goes to git,
  this is about quoting examples in docs).

## Acceptance criteria results (SPEC-01 §5)

- `.gitignore` in place from first commit, `data/` untracked — OK.
- Single command (`python phase0.py "query"`) builds corpus + index on first
  run, searches on every run, no manual steps — OK.
- Manual eval, 20 queries (10 thematic, 10 paraphrased quotes from corpus
  lyrics): **18/20** returned a meaningful result in top-5 (threshold was
  14/20). All 10 paraphrased-quote queries recovered the source song at
  rank 1. 2 thematic queries ("losing a parent", "breakup and moving on")
  gave only tangentially related results.
- Query latency after model warm-up: 24–51 ms, well under the 200 ms bar.
  First call in a freshly loaded process was 476 ms (cuDNN kernel
  compilation) — added an explicit warm-up `encode(["warmup"])` call in
  `load_model()` to keep steady-state latency representative.
- Peak VRAM during embedding of 3054 chunks (bge-m3, fp16, batch 16):
  1363 MiB — comfortable margin under the 4 GB target.
