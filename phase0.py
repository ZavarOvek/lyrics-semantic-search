"""Phase 0 vertical slice: corpus -> search -> top-5 in console.

SPEC-01: deliberately a single dirty script. No protocols, no config,
no cache, no tests. Run: python phase0.py "some query text"

Source dataset: mrYou/lyrics-dataset (30k songs, English-tagged).
brunokreiner/genius-lyrics (first choice per SPEC-01) returned 401 on
file download despite public metadata -- skipped, recorded in notes.
"""

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

DEV_DIR = Path("data/dev")
CORPUS_PATH = DEV_DIR / "corpus.jsonl"
CHUNKS_PATH = DEV_DIR / "chunks.jsonl"
EMB_PATH = DEV_DIR / "embeddings.npy"

MODEL_NAME = "BAAI/bge-m3"
SEED = 42
SAMPLE_SIZE = 500
BATCH_SIZE = 16
MAX_LEN = 512

SECTION_TAG_RE = re.compile(r"\[.*?\]")


def build_corpus() -> None:
    """Download source dataset, sample SAMPLE_SIZE songs, save as JSONL."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    print("Downloading source dataset (mrYou/lyrics-dataset)...")
    path = hf_hub_download(
        repo_id="mrYou/lyrics-dataset",
        filename="data/train-00000-of-00001.parquet",
        repo_type="dataset",
    )
    df = pd.read_parquet(path)
    sample = df.sample(n=SAMPLE_SIZE, random_state=SEED).reset_index(drop=True)

    DEV_DIR.mkdir(parents=True, exist_ok=True)
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        for row in sample.itertuples():
            f.write(json.dumps({
                "id": int(row.id),
                "title": row.title,
                "artist": row.artist,
                "lyrics": row.lyrics,
            }, ensure_ascii=False) + "\n")
    print(f"Saved {len(sample)} songs to {CORPUS_PATH}")


def chunk_lyrics(lyrics: str) -> list[str]:
    """Strip section tags like [Verse 1], split into paragraphs by blank lines."""
    text = SECTION_TAG_RE.sub("", lyrics)
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


def load_model(device: str) -> SentenceTransformer:
    try:
        model = SentenceTransformer(MODEL_NAME, device=device)
    except torch.cuda.OutOfMemoryError:
        sys.exit(f"Not enough VRAM to load {MODEL_NAME} on {device}. "
                  f"Close other GPU processes or run on CPU.")
    if device == "cuda":
        model = model.half()
    model.encode(["warmup"], convert_to_numpy=True)  # trigger cudnn kernel compilation
    return model


def build_index(model: SentenceTransformer, device: str) -> None:
    """Chunk corpus, embed with bge-m3 fp16, save chunks + normalized vectors."""
    songs = [json.loads(line) for line in open(CORPUS_PATH, encoding="utf-8")]

    chunks = []
    for song in songs:
        for i, chunk_text in enumerate(chunk_lyrics(song["lyrics"])):
            chunks.append({
                "song_id": song["id"],
                "chunk_id": i,
                "title": song["title"],
                "artist": song["artist"],
                "text": chunk_text,
            })
    print(f"{len(songs)} songs -> {len(chunks)} chunks")

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    texts = [c["text"] for c in chunks]
    t0 = time.time()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    try:
        vectors = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    except torch.cuda.OutOfMemoryError:
        sys.exit("Out of VRAM during embedding. Lower BATCH_SIZE or MAX_LEN in phase0.py.")
    print(f"Embedded {len(texts)} chunks in {time.time() - t0:.1f}s")
    if device == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"Peak VRAM during embedding: {peak_mb:.0f} MiB")

    np.save(EMB_PATH, vectors.astype(np.float16))
    print(f"Saved vectors {vectors.shape} to {EMB_PATH}")


def search(model: SentenceTransformer, query: str, top_k: int = 5) -> None:
    chunks = [json.loads(line) for line in open(CHUNKS_PATH, encoding="utf-8")]
    vectors = np.load(EMB_PATH).astype(np.float32)

    t0 = time.time()
    q_vec = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
    scores = vectors @ q_vec
    elapsed_ms = (time.time() - t0) * 1000

    best_per_song = {}
    for score, chunk in zip(scores, chunks):
        key = chunk["song_id"]
        if key not in best_per_song or score > best_per_song[key][0]:
            best_per_song[key] = (score, chunk)

    ranked = sorted(best_per_song.values(), key=lambda x: -x[0])[:top_k]

    print(f"\nQuery: {query!r} ({elapsed_ms:.0f} ms)\n")
    for rank, (score, chunk) in enumerate(ranked, 1):
        print(f"{rank}. {chunk['title']} - {chunk['artist']} (score {score:.3f})")
        print(f'   "{chunk["text"][:200]}"\n')


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit('Usage: python phase0.py "query text"')
    query = " ".join(sys.argv[1:])

    if not CORPUS_PATH.exists():
        build_corpus()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)

    if not CHUNKS_PATH.exists() or not EMB_PATH.exists():
        build_index(model, device)

    search(model, query)


if __name__ == "__main__":
    main()
