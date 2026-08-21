# EVAL-THEMATIC: theme filtering, 70 → 54

The thematic track's 70-theme draft went through two rounds of manual
review before the judgement pool was built. This is a record of what
survived and why, since the survival statistics are a property of the
query set that later analysis will have to account for, and by the time
labelling is done they are only recoverable from tool logs. Companion to
`eval-auto-report.md`, which covers the automatic track; the thematic
track's own full report (pool, human labelling, LLM judge, metrics)
follows §6-9 and is not this file.

## Pass one: 70 themes, one screen each

`scripts/thematic_filter.py`, file order, no shuffling -- this is corpus
selection, not a relevance judgement, so blinding does not apply. Each
theme was shown with its top-10 from `data/probe_themes.txt` (the
`full_hybrid_faiss.yaml` probe, bge-m3 + BM25) and marked `keep`, `drop`
(deletion candidate), or `unsure`.

First pass: 51 marked `keep` outright, 19 flagged for a second opinion --
7 already leaning `drop`, 12 genuinely `unsure`. The number that matters
is what survived *both* passes, below.

## Pass two: the flagged 19, re-probed through tfidf

§4's guard against discarding on a single retriever's blind spot: the 19
flagged themes were re-probed through `configs/full_hybrid_numpy_tfidf.yaml`
(hybrid tfidf + BM25 -- deliberately unlike bge-m3, so a theme that only a
lexical arm can reach does not get discarded on the strength of the dense
arm alone) into `data/probe_tfidf.txt`, then reviewed a second time with
`--only flagged`.

The second probe was not cosmetic: **122 of the 190 candidate songs shown
across the 19 themes had not appeared in the first probe at all** (range
4-8 new per theme). The two configurations are surfacing materially
different material, which is the whole point of running both before a
theme is cut.

Outcome of the 19:

| pass-one verdict | count | pass-two outcome |
|---|---|---|
| `drop` (deletion candidate) | 7 | all 7 reaffirmed `drop` -- t45, t50, t51, t52, t53, t54, t63 |
| `unsure` | 12 | 3 revived to `keep` -- t06, t64, t68 |
| `unsure` | ↳ | 6 decided `drop` on the tfidf look -- t39, t44, t55, t61, t65, t69 |
| `unsure` | ↳ | 3 still `unsure` -- t27, t31, t67 |

None of the 7 pass-one drop candidates were revived; the tfidf probe gave
them a second look and found nothing to change the verdict. Of the 12
genuinely undecided themes, a quarter resolved to keep, half resolved to
drop, and a quarter remain undecided after two configurations and two
human passes.

## Final split

| mark | count |
|---|---|
| `keep` | **54** |
| `drop` | 13 |
| `unsure` | 3 |
| total | 70 |

The 3 `unsure` themes (t27 "missing someone who died", t31 "falling out
with a friend", t67 "the pressure of everyone depending on you") are not a
judgement call left to the pooling step -- `thematic_pool.load_themes` now
excludes `mark: unsure` explicitly, on the reasoning that an unresolved
"not sure" after two passes is a verdict, not an omission, and pooling it
would grade the system against a theme nobody signed off as answerable.
See the docstring in `scripts/thematic_pool.py` for the full argument,
including why this had to be a separate condition from `keep: false`
rather than folded into it (`keep: null` is ambiguous between "never
reviewed" and "reviewed twice, still undecided", and only `mark`
disambiguates the two).

### Survivability by tier

| tier | kept | total | rate |
|---|---|---|---|
| short (1-2 words) | 27 | 28 | 96% |
| medium (3-6 words) | 19 | 27 | 70% |
| expanded (7+ words) | 8 | 15 | 53% |

**Hypothesis, not a finding: this is corpus coverage, not query
difficulty.** The natural reading of "expanded survived worst" is that
multi-clause themes are harder for the retrieval systems to satisfy. That
is not what pass two's re-probing tested, and it is probably not the
mechanism. A more likely explanation is compositional: an expanded theme
states several conditions at once ("a mother watching her child grow up
too fast", "someone who returns to the town they grew up in and doesn't
recognise anything"), and each additional condition narrows the set of
30,000 songs that could possibly satisfy all of them simultaneously,
independent of how well any retriever ranks what is there. A short theme
("revenge", "summer nights") only has to be *about* one thing; an expanded
one has to be about several things at once in the same song. If the
corpus simply contains fewer songs meeting a four- or five-part
description than a one-part one, no retriever can find what was never
written -- pass two's job was exactly to rule out "the retriever missed
it," and having done so, what is left standing is scarcity, not a ranking
failure. This bears on how the tier slice should be read below, and
nowhere else; it does not change any code or any threshold.

## What this does not settle

The tier slice is a **reported direction, not a measurement with
arm-to-arm resolution.** `expanded` now has 8 themes. At `n=8`, per-arm
confidence intervals inside that slice will be wide enough that no arm
should be declared a winner within it -- the slice says whether systems
tend to do better or worse on expanded themes as a group, nothing finer.
Short (27) and medium (19) support more, though still modest, resolution.

## Pool built from the 54

`scripts/thematic_pool.py` over the 54 `keep` themes, all nine arms
(4 dense + 4 hybrid + 1 lexical, per `scripts/eval_matrix.py`):

- **1,270 judgements** across 54 themes, 14-32 candidates per theme,
  **mean 23.5**. This replaces the `≈21.3` figure carried through
  filtering as an indicative estimate from a 3-theme smoke pool;
  `scripts/thematic_filter.py`'s `POOLED_PER_THEME` has been updated to
  the real figure for any future filtering run.
- Verified against the source themes file by exact identity, not by
  inspection: the pool's 54 theme ids equal the file's 54 `mark: keep`
  ids exactly (set equality, not just count), and none of the 13 `drop`
  or 3 `unsure` ids appear.
- Blinding verified by field set, not by spot check: every theme record
  in `pool.jsonl` carries exactly `{candidates, text, theme_id, tier,
  warmup}`; every candidate is a bare song id, no arm, rank or score
  attached. `scripts/thematic_pool.py`'s `POOL_FIELDS` guard makes this
  structural rather than a convention that could silently drift.

## Next

Human labelling of the 1,270-judgement pool, per §6. The LLM judge runs
strictly after the human pass is complete, never before or alongside it.
