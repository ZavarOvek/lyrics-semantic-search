# Equal-weight RRF sorts by agreement, not by confidence

**Status:** established during EVAL-AUTO, on the nine-arm matrix
(`notes/reports/eval-auto-report.md`). Recorded separately from
`retrieval-design.md` §1 because that item decides *RRF instead of a
weighted score sum* and its reasoning is untouched. This note is about a
failure mode that RRF has on its own terms, once the branch weights are
equal.

## The mechanism

With 1-based ranks, `k = 60` and `top_k = 50` per branch — the values in
every `configs/full_hybrid_*.yaml` — a document's fused score is
`1/(k + rank)` summed over the branches that returned it.

| what returned the document | best possible | worst possible |
|---|---|---|
| one branch only | `1/61` = 0.016393 | `1/110` = 0.009091 |
| both branches | `2/61` = 0.032787 | `2/110` = 0.018182 |

**The worst two-branch score exceeds the best one-branch score**
(0.018182 > 0.016393). Not by a little, and not in some corner case: a
document sitting at rank 50 in *both* lists outranks a document sitting at
rank 1 in *one*. Rank cannot compensate for absence at all.

The general condition is
```
2/(k + top_k) > 1/(k + 1)   ⟺   top_k < k + 2
```
so with `k = 60` the regime holds for any `top_k ≤ 61`. At `top_k = 50` the
config sits well inside it.

That is the whole finding, and it is worth stating as a slogan: **under
equal-weight RRF with `top_k < k + 2`, the primary sort key is
co-occurrence and rank is only a tie-break within it.** The fusion is not
combining two opinions about how good a document is; it is partitioning
documents into "both branches saw it" and "one branch saw it", then
ordering within those blocks.

*(The eval report states a weaker special case of this — that a BM25 top
hit which dense merely includes outranks a dense top hit BM25 missed. That
is true but not the sharpest form. Reports are append-only, so the
correction is recorded here rather than edited into it.)*

## Why that promotes noise

The partition is only meaningful if both branches are informative. When one
branch is at chance, its top-50 is an arbitrary draw from the corpus, so
co-occurrence stops being evidence of anything and becomes a lottery: the
fused rank 1 goes to whichever of the strong branch's candidates the weak
branch happened to also return.

The matrix shows exactly that signature. On stratum `main`, n = 790:

| | recall@1 | recall@10 |
|---|---|---|
| `dense-bge-m3` | 0.228 | 0.352 |
| `hybrid-bge-m3` | 0.084 | 0.330 |
| `lexical-bm25` alone | 0.010 | 0.047 |

Recall@1 loses 63% relative; recall@10 loses 6%. The same shape appears for
`e5-base` (0.147 → 0.054 at @1, 0.282 → 0.253 at @10). **Fusion is not
ejecting the right answer from the result window — it is shuffling it
inside the window**, which is precisely what a co-occurrence-first sort
does. Any consumer reading the top hit pays the full price; a consumer
reading ten pays almost nothing.

## The caveat that matters more than the mechanism

The obvious response is to fit branch weights. **Do not fit them on this
track.**

The automatic track's lexical branch is degenerate *by construction of the
eval set, not by property of BM25*. The query gate removes every term that
a query shares with its source fragment and that occurs in ≤ 100 of 181,471
chunks — which is to say it removes exactly the evidence a bag-of-words
matcher runs on. See `corpus-relative-rarity.md` for why that gate is
structural rather than optional.

So any weight fitted against this track drives the lexical weight toward
zero, and that answer is *correct for this track and meaningless for the
product*. It would encode the eval set's construction into the retrieval
config, then present it as a measurement. This is the same class of error
as `tests/golden/spec03_eval_queries.jsonl`, which is built from song
titles and therefore systematically favours BM25 — the identical mistake
with the sign reversed.

Consequences, in order of confidence:

1. **The nine-arm numbers do not license deleting the hybrid mode.** They
   license saying that equal-weight RRF is wrong for a top-1 consumer when
   one branch is uninformative for the query type at hand.
2. **Weights, if ever introduced, must be fitted somewhere else** — the
   thematic track, or real query logs. Not here.
3. **Routing by query type is untested and not ruled out.** The paraphrase
   / description split in the report is large enough (0.413 vs 0.045
   recall@1 for `dense-bge-m3`) that one fusion policy for both is unlikely
   to be right for both.
4. **If fusion is kept unweighted, `top_k ≥ k + 2` would at least end the
   hard partition** and let rank compete with co-occurrence. This is
   arithmetic, not a recommendation: it has not been measured, and raising
   `top_k` changes the candidate pool as well.

## Where this lives in code

`lyrics_search/core/fusion.py` (`reciprocal_rank_fusion`) — the formula and
its deterministic tie-break; `k` is a parameter, branch weights do not
exist. `lyrics_search/retrievers/hybrid.py` calls it. `rrf_k` and `top_k`
come from the `retrieval:` block of each config. Nothing here proposes a
code change; the note exists so the next person to look at
`hybrid-*` scoring below `dense-*` does not read it as a bug in the fusion
implementation, and does not "fix" it by tuning against a degenerate
branch.
