# EVAL-AUTO completion report

The automatic, known-item track: generate one paraphrase and one
description per sampled song with an LLM, gate the generated queries
against verbatim leakage, then run all nine retrieval arms over the
surviving set.

This is one of the two eval tracks. The thematic track is separate, is
judged by a human, and is not covered here. **No overall winner is
declared in this report** — see "What this does not settle".

## The query set

500 songs were sampled with deliberate over-sampling of two small strata
(SPEC decision 7, recorded in `notes/STATUS.md`). 498 reached generation;
each produced two queries, for 996 candidates.

Generation ran through the `eval-query-generator-batch` subagent at a batch
size of 10, the size chosen by the batching experiment in the previous
step. Every generated query then passed the gate in
`scripts/eval_gate_queries.py`: a query fails if it shares a term with its
own source fragment that occurs in **≤ 100 of the corpus's 181,471 chunks**
(idf ≥ 8.4937). A query that reuses a rare word from the fragment it
describes is not a paraphrase of the song's meaning; it is a partial
verbatim quote, and it hands the item to lexical retrieval for free.
Failing queries were regenerated, to a maximum of three attempts.

Final set: **985 queries over 498 songs**, zero duplicate query texts.

| stratum | paraphrase | description | total |
|---|---|---|---|
| `main` | 392 | 398 | 790 |
| `structure` | 48 | 49 | 97 |
| `translation` | 48 | 50 | 98 |
| | 488 | 497 | **985** |

### Attrition, and why it is small

The owner asked for two numbers specifically: the final drop rate after
three attempts, and whether the same songs fail repeatedly. A drop above
10% was named in advance as a finding worth reporting.

**Final attrition is 11 of 996, 1.10%.** Ten songs lost their paraphrase,
one lost its description, and no song lost both — so every one of the 498
songs is still represented by at least one query. Per stratum the loss is
1.5% (`main`), 4.0% (`structure`) and 4.0% (`translation`); the two small
strata are on 100 candidates each, so those figures are one or two queries.

**Failure is a property of one word, not of the attempt.** Conditional
failure rate rises sharply with each retry:

| attempt | queries tried | failed | rate |
|---|---|---|---|
| 1 | 996 | 38 | 3.8% |
| 2 | 38 | 17 | 44.7% |
| 3 | 17 | 11 | 64.7% |

A query that failed once fails again at roughly twelve times the base
rate, and one that failed twice at roughly seventeen times. The mechanism
is visible directly in the scored output: **9 of the 11 finally-dropped
queries leaked the identical term on all three attempts** — `haired`,
`subject`, `acts`, `touches`, `readers`, `friendship`, `backing`,
`plugged`, `craft`. The remaining two changed term and failed anyway.

These are ordinary English words that happen to be scarce *in song
lyrics* (df 17–80 out of 181,471 chunks). The generator cannot see the
corpus term distribution, so no prompt closes the gap, and for several of
them no synonym of comparable corpus frequency exists — a bass that is not
plugged into an amp is not describable without `plugged`. The retry loop
is therefore worth having for the first retry and close to worthless for
the third; the 3.8% → 44.7% jump is where its value is.

### Two process facts belonging to the record

**The gate was scoring queries it should not have.** A retry reply carries
only the query type that failed, but the original loop scored every type
present in the record. Regenerating a sibling that had already passed
would have counted as an attempt against it and diluted the repeat-failure
rate — which is the number the whole accounting exists to produce. Fixed in
`scripts/eval_gate_queries.py` (commit `0175abd`) before the figures above
were taken.

**One retry batch was blocked by the API content filter.** The block was
on the outgoing message, not on a model reply: four retry groups
concatenated into one dispatch put 36 lyrics fragments in a single prompt,
and the same fragments had gone out without incident in earlier rounds when
sent one group per message. Mitigation was to dispatch one group per
message and, for a single fragment containing a slur, to mask it in the
prompt. That is safe for the measurement: the fragment text only conditions
generation, while the gate scores the generated query against the sample's
own stored text, which was never altered. No song was dropped for this
reason, so nothing merged into the 1.10% above.

## The GPU control

Nine arms run back to back on a shared 4 GB laptop card. If the card heats
and clocks down over the sequence, every later arm is measured under worse
conditions than every earlier one, and the ranking picks up a drift term
that has nothing to do with retrieval. A single before/after pair would
prove that drift happened without saying where, so `scripts/eval_matrix.py`
samples `nvidia-smi` before, mid and after **every** arm and writes
`gpu.jsonl` alongside the tables.

**Result: no drift.** Temperature over the whole 70-minute sequence stayed
in a 39–42 °C band; the first sample was 39 °C and the last 41 °C. SM clock
never approached the 2100 MHz ceiling and power draw stayed near 15 W
throughout. Baseline utilization of ~19% is the desktop compositor, not
this workload.

The reason the card stays cold is worth stating, because it also explains
the timings: this workload barely uses the GPU. Encoding 790 short queries
one at a time is negligible next to retrieval, and retrieval in the hybrid
and lexical arms is CPU-bound BM25.

One further point on what the control does and does not license. Thermal
state can only move the timing column. The metric columns are deterministic
given the on-disk artifacts — the same index, the same query, the same
ranking — so the scores were never at risk from drift. The control was
still worth running: it is what makes the timings comparable, and finding
drift would have meant re-running the sequence in a different order to
separate it out.

| arm | `--stratum main` | whole set |
|---|---|---|
| `dense-bge-m3` | 46.3 s | 52.3 s |
| `dense-e5-base` | 36.6 s | 40.1 s |
| `dense-fasttext-avg` | 10.1 s | 10.4 s |
| `dense-tfidf` | 18.6 s | 21.8 s |
| `hybrid-bge-m3` | 367.4 s | 466.2 s |
| `hybrid-e5-base` | 378.1 s | 435.3 s |
| `hybrid-fasttext-avg` | 334.8 s | 420.6 s |
| `hybrid-tfidf` | 367.7 s | 460.1 s |
| `lexical-bm25` | 336.2 s | 406.8 s |

Every arm carrying a lexical branch costs about 335 s more than its dense
counterpart, regardless of which embedder sits on the other side — roughly
0.42 s per query for BM25 over 30,000 whole songs, against 12–59 ms for a
dense branch. The two CPU-only arms (`fasttext-avg`, `tfidf`) act as the
control the driver's docstring describes: their timings are flat across the
sequence too, so nothing in this column is thermal.

## Headline: stratum `main`, n = 790

Nine arms, 985-query set restricted to `main`. Bracketed figures are
bootstrap 95% intervals.

| arm | recall@1 | recall@5 | recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|
| `dense-bge-m3` | **0.228** [0.199, 0.257] | **0.311** [0.280, 0.343] | **0.352** [0.319, 0.385] | **0.271** [0.243, 0.300] | **0.286** [0.258, 0.315] |
| `dense-e5-base` | 0.147 [0.123, 0.171] | 0.246 [0.216, 0.276] | 0.282 [0.251, 0.314] | 0.191 [0.167, 0.216] | 0.210 [0.185, 0.236] |
| `dense-fasttext-avg` | 0.015 [0.008, 0.024] | 0.019 [0.010, 0.029] | 0.028 [0.016, 0.039] | 0.019 [0.010, 0.028] | 0.020 [0.012, 0.030] |
| `dense-tfidf` | 0.001 [0.000, 0.004] | 0.013 [0.005, 0.022] | 0.023 [0.013, 0.034] | 0.008 [0.004, 0.012] | 0.010 [0.005, 0.015] |
| `hybrid-bge-m3` | 0.084 [0.065, 0.104] | 0.278 [0.248, 0.310] | 0.330 [0.297, 0.363] | 0.167 [0.148, 0.188] | 0.202 [0.181, 0.225] |
| `hybrid-e5-base` | 0.054 [0.039, 0.071] | 0.208 [0.180, 0.235] | 0.253 [0.223, 0.284] | 0.121 [0.104, 0.139] | 0.149 [0.129, 0.168] |
| `hybrid-fasttext-avg` | 0.011 [0.005, 0.019] | 0.029 [0.018, 0.042] | 0.043 [0.029, 0.058] | 0.022 [0.014, 0.031] | 0.025 [0.017, 0.035] |
| `hybrid-tfidf` | 0.011 [0.005, 0.019] | 0.029 [0.018, 0.042] | 0.042 [0.028, 0.057] | 0.022 [0.014, 0.031] | 0.025 [0.016, 0.035] |
| `lexical-bm25` | 0.010 [0.004, 0.018] | 0.032 [0.020, 0.044] | 0.047 [0.033, 0.062] | 0.020 [0.013, 0.028] | 0.025 [0.017, 0.035] |

Three things in this table are worth more than the ordering.

### 1. Hybrid is worse than dense, and the mechanism is RRF, not the lexical branch

`hybrid-bge-m3` scores 0.084 recall@1 against `dense-bge-m3`'s 0.228, on
intervals that do not come close to overlapping. The same holds for
`e5-base` (0.054 against 0.147). This is not a bug and it is not a claim
that BM25 damages retrieval in general; it is what RRF does when one branch
is at chance on the task.

RRF scores a document as the sum of `1/(k + rank)` over the branches that
returned it, with `k = 60` and `top_k = 50` per branch. A document the
dense branch ranks first and BM25 misses entirely scores `1/61 = 0.0164`.
A document BM25 ranks first that the dense branch has anywhere in its top
50 scores at least `1/61 + 1/110 = 0.0255`. So **any BM25 top hit that
dense merely includes outranks a dense top hit that BM25 missed.** On this
query set BM25 alone is at 0.010 recall@1 — essentially chance — so what
the fusion promotes into rank 1 is mostly noise.

Recall@10 tells the other half of the story: 0.330 against 0.352, a 6%
relative loss where recall@1 lost 63%. Fusion is not throwing the right
answer out of the window; it is reshuffling within it. Any arm whose
consumer reads the top result pays the full price. That distinction matters
for the eventual product decision and is the reason both `@1` and `@10` are
reported rather than one summary number.

### 2. The lexical arms are at chance, by construction

`lexical-bm25`, `dense-tfidf`, and both of their hybrids sit between 0.001
and 0.015 recall@1. That is the gate working as designed, not a broken arm.
The gate removes every query that shares a rare term with its source
fragment, which removes exactly the signal a bag-of-words matcher has. A
query set that left those terms in would report BM25 far higher and the
number would mean nothing — this is the same failure mode already recorded
against `tests/golden/spec03_eval_queries.jsonl`, which is built from song
titles.

So these rows should be read as "lexical retrieval contributes nothing to
*this* task", not as a general measurement of BM25. The task is
known-item retrieval from a paraphrase that was explicitly stripped of
shared rare vocabulary. Verifying that the harness itself was sound was
done the other way round: `dense-bge-m3` was smoke-tested first and
returned 0.228, so the near-zero lexical rows are a property of the query
set, not of the runner.

### 3. Description queries are much harder than paraphrases

Within `main`:

| arm | type | n | recall@1 | recall@10 | MRR |
|---|---|---|---|---|---|
| `dense-bge-m3` | paraphrase | 392 | 0.413 [0.365, 0.462] | 0.564 [0.515, 0.615] | 0.468 [0.423, 0.514] |
| `dense-bge-m3` | description | 398 | 0.045 [0.025, 0.068] | 0.143 [0.111, 0.178] | 0.078 [0.057, 0.100] |
| `dense-e5-base` | paraphrase | 392 | 0.263 [0.222, 0.306] | 0.431 [0.383, 0.480] | 0.319 [0.278, 0.362] |
| `dense-e5-base` | description | 398 | 0.033 [0.018, 0.050] | 0.136 [0.103, 0.171] | 0.065 [0.047, 0.086] |
| `dense-fasttext-avg` | paraphrase | 392 | 0.023 [0.010, 0.038] | 0.038 [0.020, 0.059] | 0.028 [0.013, 0.044] |
| `dense-fasttext-avg` | description | 398 | 0.008 [0.000, 0.018] | 0.018 [0.005, 0.033] | 0.010 [0.002, 0.019] |
| `dense-tfidf` | paraphrase | 392 | 0.003 [0.000, 0.008] | 0.046 [0.026, 0.066] | 0.015 [0.009, 0.023] |
| `dense-tfidf` | description | 398 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.001 [0.000, 0.001] |
| `hybrid-bge-m3` | paraphrase | 392 | 0.145 [0.112, 0.181] | 0.538 [0.490, 0.587] | 0.281 [0.248, 0.315] |
| `hybrid-bge-m3` | description | 398 | 0.023 [0.010, 0.038] | 0.126 [0.093, 0.161] | 0.055 [0.040, 0.073] |
| `hybrid-e5-base` | paraphrase | 392 | 0.094 [0.066, 0.125] | 0.403 [0.357, 0.452] | 0.199 [0.170, 0.230] |
| `hybrid-e5-base` | description | 398 | 0.015 [0.005, 0.028] | 0.106 [0.075, 0.136] | 0.044 [0.030, 0.059] |
| `hybrid-fasttext-avg` | paraphrase | 392 | 0.013 [0.003, 0.026] | 0.066 [0.043, 0.092] | 0.031 [0.019, 0.045] |
| `hybrid-fasttext-avg` | description | 398 | 0.010 [0.003, 0.020] | 0.020 [0.008, 0.035] | 0.014 [0.005, 0.025] |
| `hybrid-tfidf` | paraphrase | 392 | 0.023 [0.010, 0.038] | 0.069 [0.046, 0.094] | 0.040 [0.025, 0.057] |
| `hybrid-tfidf` | description | 398 | 0.000 [0.000, 0.000] | 0.015 [0.005, 0.028] | 0.004 [0.002, 0.008] |
| `lexical-bm25` | paraphrase | 392 | 0.015 [0.005, 0.028] | 0.071 [0.046, 0.097] | 0.031 [0.019, 0.046] |
| `lexical-bm25` | description | 398 | 0.005 [0.000, 0.013] | 0.023 [0.010, 0.038] | 0.010 [0.003, 0.019] |

`dense-bge-m3` gets 0.413 recall@1 on paraphrases and 0.045 on
descriptions — a factor of nine, and the same gap appears in every arm that
scores above chance at all. The two types are not interchangeable and the
headline number is an average over a bimodal population, which is why they
are reported apart.

The gap is expected rather than surprising. A paraphrase restates one
specific fragment, so a single chunk is a close semantic neighbour of it. A
description characterises the whole song in general terms — subject,
mood, situation — and a great many songs in a 30,000-song corpus share
those. Known-item retrieval from a general description is close to the
thematic task, where "the correct answer" is a set rather than one item;
scoring it as known-item retrieval undercounts it by construction. The
thematic track exists precisely to measure that properly, and these
`description` rows should not be read as the final word on it.

## Over-sampled strata, reported apart

`structure` (songs whose section splitting was `none` or `plain_label`) and
`translation` were sampled at 50 songs each against `main`'s 400, so a row
pooling all three would describe no real population. They are read from the
whole-set run's `stratum` slice and never merged into the headline.

| arm | stratum | n | recall@1 | recall@10 | MRR |
|---|---|---|---|---|---|
| `dense-bge-m3` | structure | 97 | 0.237 [0.155, 0.330] | 0.371 [0.278, 0.474] | 0.275 [0.194, 0.361] |
| `dense-bge-m3` | translation | 98 | 0.286 [0.194, 0.378] | 0.429 [0.327, 0.531] | 0.331 [0.246, 0.421] |
| `dense-e5-base` | structure | 97 | 0.144 [0.082, 0.216] | 0.268 [0.186, 0.361] | 0.176 [0.110, 0.250] |
| `dense-e5-base` | translation | 98 | 0.153 [0.082, 0.224] | 0.276 [0.194, 0.367] | 0.191 [0.122, 0.265] |
| `dense-fasttext-avg` | structure | 97 | 0.031 [0.000, 0.072] | 0.062 [0.021, 0.113] | 0.040 [0.009, 0.080] |
| `dense-fasttext-avg` | translation | 98 | 0.031 [0.000, 0.071] | 0.041 [0.010, 0.082] | 0.034 [0.004, 0.075] |
| `dense-tfidf` | structure | 97 | 0.010 [0.000, 0.031] | 0.010 [0.000, 0.031] | 0.012 [0.001, 0.034] |
| `dense-tfidf` | translation | 98 | 0.010 [0.000, 0.031] | 0.020 [0.000, 0.051] | 0.013 [0.000, 0.035] |
| `hybrid-bge-m3` | structure | 97 | 0.124 [0.062, 0.196] | 0.351 [0.258, 0.443] | 0.211 [0.147, 0.280] |
| `hybrid-bge-m3` | translation | 98 | 0.092 [0.041, 0.153] | 0.429 [0.327, 0.531] | 0.206 [0.148, 0.270] |
| `hybrid-e5-base` | structure | 97 | 0.103 [0.052, 0.165] | 0.258 [0.175, 0.351] | 0.162 [0.103, 0.228] |
| `hybrid-e5-base` | translation | 98 | 0.051 [0.010, 0.102] | 0.255 [0.173, 0.337] | 0.111 [0.065, 0.161] |
| `hybrid-fasttext-avg` | structure | 97 | 0.052 [0.010, 0.103] | 0.093 [0.041, 0.155] | 0.064 [0.024, 0.113] |
| `hybrid-fasttext-avg` | translation | 98 | 0.010 [0.000, 0.031] | 0.051 [0.010, 0.102] | 0.022 [0.004, 0.049] |
| `hybrid-tfidf` | structure | 97 | 0.010 [0.000, 0.031] | 0.082 [0.031, 0.144] | 0.027 [0.009, 0.053] |
| `hybrid-tfidf` | translation | 98 | 0.010 [0.000, 0.031] | 0.051 [0.010, 0.102] | 0.020 [0.003, 0.045] |
| `lexical-bm25` | structure | 97 | 0.052 [0.010, 0.103] | 0.103 [0.041, 0.165] | 0.067 [0.026, 0.115] |
| `lexical-bm25` | translation | 98 | 0.020 [0.000, 0.051] | 0.031 [0.000, 0.071] | 0.024 [0.001, 0.055] |

**Neither stratum collapses.** For `dense-bge-m3`, `structure` is at 0.237
and `translation` at 0.286 against `main`'s 0.228; both intervals contain
the `main` value. At n ≈ 97 those intervals are wide enough that only a
gross failure would have been visible, so this is a floor check, not a
measurement — but the floor is the thing the over-sampling was there to
check. Two design decisions taken earlier survive it: songs whose sections
could not be split are not thereby unfindable, and translations, which are
flagged rather than filtered, are not dragging the corpus down.

The one asymmetry worth flagging is that `hybrid-bge-m3` loses more on
`translation` (0.286 → 0.092) than on `main` (0.228 → 0.084) in relative
terms while its recall@10 on `translation` holds at 0.429, identical to the
dense arm. Same mechanism as §1, and on 98 queries not worth reading
further into.

## What this does not settle

**No winner is declared.** The automatic track separates the arms cleanly —
`dense-bge-m3` is ahead of `dense-e5-base` on every metric with
non-overlapping intervals, and both are far ahead of everything else — but
this track measures one thing: recovering a known item from a
model-written paraphrase of one of its fragments, with shared rare
vocabulary deliberately removed. That is a legitimate task and a hard one,
and it is not the task the search is for.

Specifically:

- **The thematic track has not run.** It is judged by a human, it scores
  relevance in three grades rather than known-item hit or miss, and it is
  where the `description` rows above have a fair test. Averaging the two
  tracks is ruled out by an earlier decision; they answer different
  questions.
- **The queries are model-generated.** A retriever that shares an
  inductive bias with the generator is advantaged in a way the gate does
  not catch — the gate removes shared *terms*, not shared notions of what
  a fragment is about. This is a known limit of the track, and the reason
  the thematic track's judge is a human.
- **The whole-song chunking arm is out of scope**, per the earlier
  decision. Only `sections` is measured here.
- **The hybrid result argues for tuning, not for deletion.** RRF with
  equal branch weighting is what these numbers indict. A weighted fusion,
  or routing by query type, was not tested and is not ruled out by
  anything above.

## Reproducing

```
./.venv/Scripts/python.exe scripts/eval_gate_queries.py \
    --raw-dir data/eval/auto/gen --scored data/eval/auto/scored.jsonl \
    --retry data/eval/auto/retry.txt --out data/eval/auto/queries.jsonl \
    --max-attempts 3

./.venv/Scripts/python.exe scripts/eval_matrix.py \
    --eval-set data/eval/auto/queries.jsonl --out-dir data/eval/auto/results
```

The matrix driver writes two result tables per arm (`<arm>.main.json`,
`<arm>.all.json`), a per-run log, `gpu.jsonl` and `timings.json`. All of it
lands under `data/`, which is not in git.

Three configs were added for this run, since the matrix named arms that had
no config: `configs/full_hybrid_faiss_e5base.yaml`,
`configs/full_hybrid_faiss_fasttext.yaml` and `configs/full_lexical.yaml`.
The last one is a single config rather than four because `build_retriever`
resolves no embedder in lexical mode and BM25 scores `text_raw` from
`raw.jsonl`, which is shared across arms; its `embedder` and `index` keys
are inert and present only because the schema validates them either way.
