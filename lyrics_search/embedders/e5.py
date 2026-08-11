"""SPEC-02 §5: intfloat/multilingual-e5-base embedder.

E5 models require "query: "/"passage: " prefixes per the model card --
without them, retrieval quality degrades significantly. is_query in the
Embedder protocol exists specifically so this prefix choice can live here
rather than being pushed onto every caller.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-base"
DIM = 768


class E5Embedder:
    name = "e5-base"
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
        self._model.encode(["query: warmup"], convert_to_numpy=True)  # cudnn warmup
        # SPEC-02-PATCH item 5: real max_length from the real tokenizer,
        # for runners/embed.py's post-cascade overflow check.
        self.max_seq_length = self._model.max_seq_length

    def _prefix(self, texts: Sequence[str], *, is_query: bool) -> list[str]:
        prefix = "query: " if is_query else "passage: "
        return [prefix + t for t in texts]

    def token_lengths(self, texts: Sequence[str], *, is_query: bool) -> list[int]:
        """Real (untruncated) token counts, including the query:/passage:
        prefix -- same prefixing as encode()."""
        prefixed = self._prefix(texts, is_query=is_query)
        enc = self._model.tokenizer(prefixed, truncation=False, padding=False)
        return [len(ids) for ids in enc["input_ids"]]

    def encode(self, texts: Sequence[str], *, is_query: bool) -> np.ndarray:
        prefixed = self._prefix(texts, is_query=is_query)
        try:
            vecs = self._model.encode(
                prefixed,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > 256,
            )
        except torch.cuda.OutOfMemoryError:
            sys.exit("Out of VRAM during e5 embedding. Lower batch_size.")
        return vecs.astype(np.float32)
