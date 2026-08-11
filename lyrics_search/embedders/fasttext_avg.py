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

PRE-EVAL-OPT §1: `vectors_path` may instead name a `.kv` artifact --
gensim's native format, produced once by
scripts/convert_fasttext_vectors.py and loaded with `mmap='r'` so the
vector matrix is mapped from its .npy sidecar rather than parsed. Parsing
the text file costs ~178s against ~0.2s mapped, and both eval tracks pay
that on every run. The format is chosen by file extension, the same way
vector_store.py picks a reader from `vectors.npy` vs `vectors.npz`.

`limit` cannot be applied to a `.kv` artifact: gensim's loader has no such
parameter, so the vocabulary is fixed when the artifact is built. Rather
than ignore the argument, a disagreement between it and the artifact is
raised -- silently loading a different vocabulary than the config asked
for would change which words are in scope, and so change every embedding
built from it.

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

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from lyrics_search.core.text import tokenize_words

DIM = 300
BINARY_SUFFIX = ".kv"  # gensim native format; anything else is parsed as text


class FastTextAvgEmbedder:
    name = "fasttext-avg"
    dim = DIM

    def __init__(self, vectors_path: str, limit: int | None = 500_000):
        from gensim.models import KeyedVectors

        path = Path(vectors_path)
        if not path.exists():
            raise FileNotFoundError(
                f"{self.name}: missing word vectors {path}. Supply the fastText "
                f".vec file, or build the fast-loading binary with "
                f"scripts/convert_fasttext_vectors.py."
            )
        if path.suffix == BINARY_SUFFIX:
            self._kv = KeyedVectors.load(str(path), mmap="r")
            vocab_size = len(self._kv.index_to_key)
            if limit is not None and vocab_size != limit:
                raise ValueError(
                    f"{self.name}: {path} holds {vocab_size} words but limit={limit} "
                    f"was requested. A .kv artifact's vocabulary is fixed at build "
                    f"time and cannot be re-limited on load -- rebuild it with "
                    f"scripts/convert_fasttext_vectors.py --limit {limit}, or drop "
                    f"the limit from the config."
                )
        else:
            self._kv = KeyedVectors.load_word2vec_format(str(path), limit=limit)
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
