# SPEC-03 §7/§9 / SPEC-03-PATCH §2 / SPEC-04 §0.3: first-pass-vs-repeat
# latency on the full corpus (~181k chunks), real bge-m3 + NumpyIndex, run
# via `.venv`'s own Python (CUDA + faiss both available there -- see
# notes/reports/spec03-patch-report.md §2 for why the interpreter matters).
#
# Deliberately still benchmarks NumpyIndex, not FaissIndex: this isolates
# the retrieval-layer's own search() cost from HNSW's approximate-search
# behavior, and NumpyIndex's brute-force dot product at full-corpus scale
# is the more conservative (slower) of the two, so it's the right one to
# hold the <100ms target to.
#
# Naming (SPEC-04 §0.3): "first pass" / "repeats" below, not "cold"/"warm" --
# both are measured *after* the embedder's own warmup() call (below, part of
# model load), so neither is a true cold start. The real cold start (Phase 0
# measured ~476ms for one unwarmed encode call) is folded into "embedder
# load" instead -- exactly what making warmup() mandatory was for: pay the
# one-time JIT/cudnn-autotune cost once at load time, never on a real query.
# encode() and index.search() are additionally timed separately below, to
# attribute repeat-query cost between "the retrieval infrastructure this
# project built" (search) and "the pretrained model's own inference cost"
# (encode) -- SPEC-03-PATCH §2's explicit ask.
import json
import time
import sys

sys.path.insert(0, ".")

import numpy as np

from lyrics_search.indexes.numpy_index import NumpyIndex
from lyrics_search.embedders.bge_m3 import BGEM3Embedder
from lyrics_search.retrievers.dense import DenseRetriever
from lyrics_search.retrievers.loading import load_chunk_lookup, warmup

t0 = time.time()
chunk_lookup = load_chunk_lookup("data/full/sections/chunks.jsonl")
print(f"chunk_lookup load: {time.time() - t0:.2f}s ({len(chunk_lookup)} chunks)")

t0 = time.time()
vectors = np.load("data/full/sections/embeddings/bge-m3/vectors.npy").astype(np.float32)
chunk_ids = json.load(open("data/full/sections/embeddings/bge-m3/chunk_ids.json", encoding="utf-8"))
index = NumpyIndex()
index.build(vectors, chunk_ids)
print(f"index build: {time.time() - t0:.2f}s ({vectors.shape[0]} vectors, dim={vectors.shape[1]})")

t0 = time.time()
embedder = BGEM3Embedder()
warmup(embedder)  # absorbs the true cold-start cost here, once
model_load_s = time.time() - t0
print(f"embedder load (incl. warmup -- true cold start absorbed here): {model_load_s:.2f}s")

retriever = DenseRetriever(embedder, index, chunk_lookup)

queries = [
    "heartbreak and lost love",
    "partying all night in the city",
    "faith and prayer to god",
    "money power and fame",
    "missing someone who is gone",
]

REPEATS = 10
first_pass_times, repeat_times_all = [], []
encode_times_all, search_times_all = [], []
for q in queries:
    t0 = time.time()
    retriever.search(q, top_k=50)
    first_pass_s = time.time() - t0
    first_pass_times.append(first_pass_s)

    repeat_times = []
    for _ in range(REPEATS):
        t0 = time.time()
        retriever.search(q, top_k=50)
        repeat_times.append(time.time() - t0)
    repeat_times_all.extend(repeat_times)

    # Separately time encode() vs index.search() (SPEC-03-PATCH §2): same
    # two calls DenseRetriever.search() makes internally, just timed apart
    # so repeat-query cost can be attributed between "the pretrained model's
    # own inference cost" (encode) and "the retrieval infrastructure this
    # project built" (search).
    encode_times, search_times = [], []
    for _ in range(REPEATS):
        t0 = time.time()
        vec = embedder.encode([q], is_query=True)[0]
        encode_times.append(time.time() - t0)

        t0 = time.time()
        index.search(vec, top_k=50)
        search_times.append(time.time() - t0)
    encode_times_all.extend(encode_times)
    search_times_all.extend(search_times)

    print(
        f"{q!r:40s} first_pass={first_pass_s * 1000:6.1f}ms  "
        f"repeat_avg={sum(repeat_times) / len(repeat_times) * 1000:6.1f}ms  "
        f"repeat_min={min(repeat_times) * 1000:6.1f}ms  repeat_max={max(repeat_times) * 1000:6.1f}ms  "
        f"encode_avg={sum(encode_times) / len(encode_times) * 1000:6.1f}ms  "
        f"search_avg={sum(search_times) / len(search_times) * 1000:6.1f}ms"
    )

print()
print(f"Total load time (index+embedder, one-time, incl. true cold start): {model_load_s:.2f}s")
print(
    f"Mean first-pass query (first call per query, model already loaded+warmed): {sum(first_pass_times) / len(first_pass_times) * 1000:.1f}ms"
)
print(
    f"Mean repeat query ({REPEATS} repeats x {len(queries)} queries): {sum(repeat_times_all) / len(repeat_times_all) * 1000:.1f}ms"
)
print(f"  of which mean encode(): {sum(encode_times_all) / len(encode_times_all) * 1000:.1f}ms")
print(
    f"  of which mean index.search(): {sum(search_times_all) / len(search_times_all) * 1000:.1f}ms"
)
print(
    f"<100ms repeat-query target: {'PASS' if sum(repeat_times_all) / len(repeat_times_all) * 1000 < 100 else 'FAIL'}"
)
