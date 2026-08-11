# SPEC-03 §9 / SPEC-03-PATCH §5: dense vs lexical vs hybrid retriever
# comparison on the dev corpus, real bge-m3 (already-built
# data/dev/sections/embeddings/bge-m3).
#
# BOOTSTRAP SANITY SET ONLY -- NOT the eval-phase benchmark. See
# tests/golden/spec03_eval_queries.jsonl's own leading _meta record and
# notes/reports/spec03-patch-report.md §5 for the full caveat. Short version: the
# query is each song's own *title*, and a title typically recurs verbatim
# in the lyrics (usually the hook) -- so BM25 gets a free exact-string-match
# advantage that dense embedding similarity does not. Any dense-vs-lexical
# *ordering* this script prints is therefore an artifact of how these
# queries were constructed, not a property of the retrievers themselves.
# Only the fact that all three retrievers clear a low floor is meaningful
# here; do not carry this eval set or its numbers forward into the real
# eval phase.
#
# Eval set: tests/golden/spec03_eval_queries.jsonl -- 20 songs sampled
# evenly (every len(songs)//20-th) across data/dev/raw.jsonl, using each
# song's own title as the query text and that song's song_id as the sole
# relevant judgment. Cheap-to-construct (no manual lyrics reading/judging
# needed) -- a floor/sanity check on Recall@10, not a rigorous benchmark.
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

from lyrics_search.embedders.bge_m3 import BGEM3Embedder
from lyrics_search.indexes.numpy_index import NumpyIndex
from lyrics_search.retrievers.dense import DenseRetriever
from lyrics_search.retrievers.hybrid import HybridRetriever
from lyrics_search.retrievers.lexical import LexicalRetriever
from lyrics_search.retrievers.loading import load_chunk_lookup, load_song_corpus

RETURN_N = 10

chunk_lookup = load_chunk_lookup("data/dev/sections/chunks.jsonl")
song_corpus = load_song_corpus("data/dev/raw.jsonl")

import numpy as np  # noqa: E402 — sys.path is prepared above, on purpose

vectors = np.load("data/dev/sections/embeddings/bge-m3/vectors.npy").astype(np.float32)
chunk_ids = json.loads(
    Path("data/dev/sections/embeddings/bge-m3/chunk_ids.json").read_text(encoding="utf-8")
)
index = NumpyIndex()
index.build(vectors, chunk_ids)

embedder = BGEM3Embedder()

dense = DenseRetriever(embedder, index, chunk_lookup)
lexical = LexicalRetriever(song_corpus, chunk_lookup)
hybrid = HybridRetriever(dense, lexical)

queries = []
with open("tests/golden/spec03_eval_queries.jsonl", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if "_meta" in rec:
            continue  # leading caveat record, not a query
        queries.append(rec)

retrievers = {"dense": dense, "lexical": lexical, "hybrid": hybrid}
hits_per_mode = dict.fromkeys(retrievers, 0)

print(f"{len(queries)} eval queries\n")
for q in queries:
    relevant = set(q["relevant_song_ids"])
    row = []
    for name, retriever in retrievers.items():
        result = retriever.search(q["query"], top_k=50)
        top_ids = {h.song_id for h in result.hits[:RETURN_N]}
        hit = bool(relevant & top_ids)
        hits_per_mode[name] += int(hit)
        row.append(f"{name}={'HIT' if hit else 'miss'}")
    print(f"{q['query']!r:50s} " + "  ".join(row))

print()
for name in retrievers:
    recall = hits_per_mode[name] / len(queries)
    print(f"Recall@{RETURN_N} [{name}]: {hits_per_mode[name]}/{len(queries)} = {recall:.2f}")
