"""SPEC-02 §5: TF-IDF baseline embedder (no neural net, no GPU).

Unlike the neural embedders, TF-IDF needs corpus statistics before it can
encode anything: call fit(corpus_texts) once, on the chunk texts being
indexed, before the first encode(). `dim` is only meaningful after fit()
-- it equals the fitted vocabulary size -- and is 0 beforehand.

EVAL-PREP §1: encode() returns a `scipy.sparse` CSR matrix, not a dense
array, and the vocabulary is uncapped. Measured on the full corpus
(181,471 chunks): the uncapped vocabulary is 76,649 terms at ~28 non-zeros
per chunk -- 0.14% density, 51.8 GiB dense against 39.3 MiB sparse. The
former `max_features=20_000` cap existed only to keep the dense matrix
from being absurd (13.5 GiB rather than 51.8); against sparse storage it
saves 0.8 MiB while discarding 56,000 terms. Because `max_features` keeps
the top terms by document frequency, it cut precisely the rare words that
distinguish one song among thirty thousand -- the one case where a lexical
baseline has a structural advantage. Removing it makes the baseline
honest.

Two consequences, recorded rather than worked around: `FaissIndex` takes
no sparse input, so `tfidf` runs with the `numpy` index only -- a property
of the method, not a gap in coverage; and `tfidf` is expected to be the
*fastest* arm, since ~5.2 million non-zeros is far less arithmetic than
bge-m3's dense 181,471 x 1024.

SPEC-02-PATCH item 3: the fitted vocabulary/IDF *is* this embedder's model
-- a query encoded against a different fit would land in a different,
incomparable vector space (different dim, different feature indices even
at the same dim). save_fit()/load_fit() persist and restore that exact
state via joblib so a separate query-time process can reproduce the same
embedding without ever calling fit() again. encode() refuses to run
without either fit() or load_fit() having been called first -- no silent
fallback, per SPEC-00 §3.2 (fail loudly).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfEmbedder:
    name = "tfidf"

    def __init__(self, max_features: int | None = None):
        # Uncapped by default (see module docstring). The parameter stays
        # so a caller can deliberately cap the vocabulary -- e.g. to
        # reproduce an older index -- but nothing in the pipeline does.
        self._vectorizer = TfidfVectorizer(max_features=max_features, norm="l2")
        self._fitted = False
        self.dim = 0

    def fit(self, corpus_texts: Sequence[str]) -> None:
        self._vectorizer.fit(corpus_texts)
        self._fitted = True
        self.dim = len(self._vectorizer.vocabulary_)

    def save_fit(self, path: Path | str) -> None:
        if not self._fitted:
            raise RuntimeError("Cannot save_fit(): fit(corpus_texts) was never called.")
        import joblib

        joblib.dump(self._vectorizer, path)

    def load_fit(self, path: Path | str) -> None:
        import joblib

        self._vectorizer = joblib.load(path)
        self._fitted = True
        self.dim = len(self._vectorizer.vocabulary_)

    def encode(self, texts: Sequence[str], *, is_query: bool) -> sparse.csr_matrix:
        if not self._fitted:
            raise RuntimeError(
                "TfidfEmbedder.fit(corpus_texts) or load_fit(path) must be called "
                "before encode() -- encode() never fits implicitly."
            )
        # TfidfVectorizer already L2-normalises each row (norm="l2"), so the
        # sparse rows are unit vectors and cosine similarity stays a plain
        # dot product -- exactly as it was for the dense return.
        return self._vectorizer.transform(list(texts)).astype("float32").tocsr()
