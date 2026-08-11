import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from lyrics_search.embedders.bge_m3 import BGEM3Embedder
from lyrics_search.indexes.faiss_index import FaissIndex
from lyrics_search.indexes.numpy_index import NumpyIndex

vectors = np.load("data/full/sections/embeddings/bge-m3/vectors.npy").astype(np.float32)
chunk_ids = json.loads(
    Path("data/full/sections/embeddings/bge-m3/chunk_ids.json").read_text(encoding="utf-8")
)
print(f"Loaded {vectors.shape} vectors")

t0 = time.time()
npidx = NumpyIndex()
npidx.build(vectors, chunk_ids)
print(f"NumpyIndex build: {time.time() - t0:.2f}s")

t0 = time.time()
fidx = FaissIndex()
fidx.build(vectors, chunk_ids)
print(f"FaissIndex build: {time.time() - t0:.2f}s")

embedder = BGEM3Embedder()
queries = [
    "heartbreak and lost love",
    "partying all night in the city",
    "faith and prayer to god",
    "money power and fame",
    "missing someone who is gone",
]

chunk_text = {}
with open("data/full/sections/chunks.jsonl", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        chunk_text[rec["chunk_id"]] = rec["text"]

overlaps = []
np_times, f_times = [], []
for q in queries:
    qv = embedder.encode([q], is_query=True)[0]
    t0 = time.time()
    np_res = npidx.search(qv, 10)
    np_times.append(time.time() - t0)

    t0 = time.time()
    f_res = fidx.search(qv, 10)
    f_times.append(time.time() - t0)

    np_ids = {cid for cid, _ in np_res}
    f_ids = {cid for cid, _ in f_res}
    overlap = len(np_ids & f_ids)
    overlaps.append(overlap)
    print(f"\nQuery: {q!r}  top10 overlap={overlap}/10")
    print(
        f"  numpy top1: {np_res[0][0]} score={np_res[0][1]:.4f} text={chunk_text[np_res[0][0]][:60]!r}"
    )
    print(
        f"  faiss top1: {f_res[0][0]} score={f_res[0][1]:.4f} text={chunk_text[f_res[0][0]][:60]!r}"
    )

print(f"\nMean top-10 overlap: {sum(overlaps) / len(overlaps):.2f}/10")
print(f"Mean numpy search time: {sum(np_times) / len(np_times) * 1000:.1f}ms")
print(f"Mean faiss search time: {sum(f_times) / len(f_times) * 1000:.1f}ms")
