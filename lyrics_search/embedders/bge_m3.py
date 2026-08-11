"""SPEC-02 §5: BAAI/bge-m3 embedder.

Ports load_model()/build_index() encode logic from phase0.py: fp16 on
CUDA, warmup call on load to avoid the ~450ms cuDNN cold-start hit on the
first real query (see notes/phase0-findings.md).
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-m3"
DIM = 1024


class BGEM3Embedder:
    name = "bge-m3"
    dim = DIM

    def __init__(self, device: str | None = None, batch_size: int = 16):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        try:
            self._model = SentenceTransformer(MODEL_NAME, device=self.device)
        except torch.cuda.OutOfMemoryError:
            sys.exit(f"Not enough VRAM to load {MODEL_NAME} on {self.device}.")
        if self.device == "cuda":
            self._model = self._model.half()
        self._model.encode(["warmup"], convert_to_numpy=True)  # cudnn kernel warmup
        # SPEC-02-PATCH item 5: real max_length from the real tokenizer,
        # for runners/embed.py's post-cascade overflow check.
        self.max_seq_length = self._model.max_seq_length

    def token_lengths(self, texts: Sequence[str], *, is_query: bool) -> list[int]:
        """Real (untruncated) token counts for `texts`, as this embedder
        would actually tokenize them -- same prefixing as encode()."""
        enc = self._model.tokenizer(list(texts), truncation=False, padding=False)
        return [len(ids) for ids in enc["input_ids"]]

    def encode(self, texts: Sequence[str], *, is_query: bool) -> np.ndarray:
        # bge-m3 does not use query/passage prefixes; is_query is accepted
        # only for Embedder-protocol compatibility with e5.
        try:
            vecs = self._model.encode(
                list(texts),
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > 256,
            )
        except torch.cuda.OutOfMemoryError:
            sys.exit("Out of VRAM during bge-m3 embedding. Lower batch_size.")
        return vecs.astype(np.float32)
