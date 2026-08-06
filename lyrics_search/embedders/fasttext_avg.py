"""SPEC-02 §5: FastText-average embedder (cc.en.300, TF-IDF-weighted).

The embedder this project exists to test (per SPEC-02 §5, "implement
properly, not as a token gesture"): static pretrained word vectors,
averaged over a chunk's words and weighted by each word's TF-IDF weight
in the fitted corpus. Falls back to uniform averaging for
out-of-vocabulary words.

Vector file: cc.en.300.vec, fastText English word vectors, 300-dim,
text format (https://fasttext.cc/docs/en/crawl-vectors.html). This file
is large (~1.3GB gzip-compressed / ~4.5GB uncompressed, ~2M words) and is
NOT downloaded automatically by this module -- the caller supplies a
local path via `vectors_path`, which runners/embed.py only does after
explicit user permission (see SPEC-00 file-download rule).

SPEC-02-PATCH item 3: unlike the word vectors themselves (loaded fresh
from `vectors_path` every time, not part of "fit"), the per-corpus IDF
weighting *is* fitted state -- a query weighted against a different
corpus's IDF would not be comparable. save_fit()/load_fit() persist just
that small dict (not the multi-GB word-vector table) via joblib.
encode() now raises if neither fit() nor load_fit() was called, instead
of the previous silent fallback to a flat default_idf=1.0 weighting
(SPEC-00 §3.2: fail loudly, don't silently degrade).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from lyrics_search.core.text import tokenize_words

DIM = 300


class FastTextAvgEmbedder:
    name = "fasttext-avg"
    dim = DIM

    def __init__(self, vectors_path: str, limit: int | None = 500_000):
        from gensim.models import KeyedVectors

        self._kv = KeyedVectors.load_word2vec_format(vectors_path, limit=limit)
        self._idf: dict[str, float] = {}
        self._default_idf = 1.0
        self._fitted = False

    def fit(self, corpus_texts: Sequence[str]) -> None:
        vectorizer = TfidfVectorizer(tokenizer=tokenize_words, lowercase=False, token_pattern=None)
        vectorizer.fit(corpus_texts)
        idf_values = vectorizer.idf_
        vocab = vectorizer.vocabulary_
        self._idf = {word: float(idf_values[idx]) for word, idx in vocab.items()}
        self._default_idf = float(np.mean(idf_values))
        self._fitted = True

    def save_fit(self, path: Path | str) -> None:
        if not self._fitted:
            raise RuntimeError("Cannot save_fit(): fit(corpus_texts) was never called.")
        import joblib

        joblib.dump({"idf": self._idf, "default_idf": self._default_idf}, path)

    def load_fit(self, path: Path | str) -> None:
        import joblib

        state = joblib.load(path)
        self._idf = state["idf"]
        self._default_idf = state["default_idf"]
        self._fitted = True

    def _embed_one(self, text: str) -> np.ndarray:
        words = tokenize_words(text)
        vecs, weights = [], []
        for w in words:
            if w not in self._kv:
                continue
            vecs.append(self._kv[w])
            weights.append(self._idf.get(w, self._default_idf))
        if not vecs:
            return np.zeros(self.dim, dtype=np.float32)
        vecs_arr = np.asarray(vecs, dtype=np.float32)
        weights_arr = np.asarray(weights, dtype=np.float32)
        avg = (vecs_arr * weights_arr[:, None]).sum(axis=0) / weights_arr.sum()
        norm = np.linalg.norm(avg)
        return avg / norm if norm > 0 else avg

    def encode(self, texts: Sequence[str], *, is_query: bool) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError(
                "FastTextAvgEmbedder.fit(corpus_texts) or load_fit(path) must be "
                "called before encode() -- encode() never fits implicitly, and no "
                "longer silently falls back to a flat IDF weighting."
            )
        return np.stack([self._embed_one(t) for t in texts]).astype(np.float32)
