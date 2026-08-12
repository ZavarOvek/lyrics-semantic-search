# Rarity is corpus-relative, so the IDF gate is a component and not a safety net

**Status:** decided during SUBAGENTS.md, on evidence from the 10-song
generation pilot. Recorded separately from `retrieval-design.md` because
it constrains how the *eval set* is built, not how retrieval works.

## The question

The known-item eval track generates queries by asking a model to
paraphrase a lyrics fragment. If the paraphrase keeps the fragment's
distinctive wording, BM25 finds the song by exact string match, and the
track measures the generation procedure instead of the retrieval task.

The generator prompt therefore forbids rare and distinctive words. The
question was whether that prohibition is sufficient on its own -- in
which case the IDF-weighted overlap check is a cheap safety net -- or
whether a mechanical gate plus regeneration is structurally required.

## The evidence

Pilot batch 1, 10 songs, 20 generated queries. One paraphrase failed the
gate, and the term that failed it was **`amid`**.

| term | document frequency | IDF |
|---|---|---|
| `amid` | 9 chunks out of 181 471 | 10.81 |
| `weeds` | 59 | 9.02 |
| `the` | 112 237 | 1.48 |

`amid` is an ordinary English preposition. Nothing about it is archaic,
dialectal, technical or distinctive in the sense the prompt describes.
Any competent judgement of "is this word unusual?" made against English
as a whole returns *no*.

Against *this* corpus it returns *yes*, emphatically: nine chunks. A
query containing `amid` hands BM25 a key that narrows 181 471 chunks to
nine before any other evidence is considered.

**`weeds` is the second case, found in a different session** during the
EVAL-AUTO mass generation (first 57 songs). It is if anything a plainer
word than `amid` -- concrete, common, no register markings at all -- and
it occurs in 59 chunks. One such case is an anecdote; two found
independently, by the same mechanical gate, on different fragments and
different runs, make it a property of the corpus rather than a quirk of
one generation.

The pattern both share is worth naming: the failing term is never an
exotic word. Exotic words are exactly the ones the prompt successfully
suppresses, because the model recognises them as distinctive. What
survives the prompt and fails the gate is ordinary vocabulary that
happens to be scarce in song lyrics specifically -- prepositions of
formal register, concrete nouns from outside the usual lyrical subject
matter. That class is invisible from inside the model and visible only
by counting.

## The decision

**The IDF gate is a required component of generation, not a check on it.**
Every generated query is scored against its source fragment and
regenerated if it fails; a run without the gate is not a valid eval set.

The reasoning is that the failure is not fixable by prompting, even in
principle. The model would need the term-frequency distribution of this
particular corpus to answer the question correctly, and it has no access
to that distribution. Sharpening the prompt can only improve the model's
estimate of *general English* rarity, which is the wrong quantity. The
gap between "rare in English" and "rare in 181 471 lyrics chunks" is
exactly where `amid` sits, and the gap does not close with better
instructions.

Keeping the prohibition in the prompt is still worthwhile: it lowers the
failure rate, which lowers the regeneration cost and -- more importantly
-- lowers how often regeneration has to fight the source material. It
just cannot be relied on as the mechanism.

## Consequences for reporting

Two numbers must be reported separately after mass generation, because
they answer different questions and the second one is the one that can
invalidate the sample:

1. **Final loss rate** -- the share of songs that failed the gate on all
   three attempts and left the set.
2. **Whether the same songs fail repeatedly.** Attempts are not
   independent. A song whose lyrics are built around distinctive rare
   vocabulary is hard to paraphrase without overlap on attempt one, and
   for the same reason on attempts two and three. If repeat failures
   concentrate on the same songs, the loss is not random attrition --
   it is systematic removal of exactly the songs where dense retrieval
   should have the largest advantage over lexical, which biases the
   benchmark toward the lexical baseline.

Per-attempt failure rate is not a substitute for either. Pilot batch 1
showed 1 in 10 on a single attempt, which says nothing directly about
what three attempts leave behind.

## Second leak channel: the title

The same pilot surfaced a leak the fragment-based gate cannot see. Batch
2's description query for one song contained `anthem` (IDF 8.93), which
does not occur in the fragment at all -- it came from the song's title,
which was passed to the generator as context.

Titles are indexed as part of the song's text, so a title word is a
lexical anchor exactly like a lyrics word, while registering as zero
fragment overlap. Hence the gate scores query terms against the title
and artist as well as against the fragment, and the two are reported
separately rather than summed.
