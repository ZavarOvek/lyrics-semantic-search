"""SPEC-02 §2: contracts shared across the whole pipeline.

Kept deliberately free of I/O and of heavy dependencies (no torch import
here) so that anything importing contracts stays cheap and testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol, Sequence

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class RawSong:
    song_id: str  # sha1(f"{norm(artist)}|{norm(title)}")[:16]
    artist: str
    title: str
    text_raw: str
    source: str
    is_translation: bool = False


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # f"{song_id}:{position}"
    song_id: str
    section: str | None  # canonical section type, or None
    text: str
    position: int
    split_by: str  # "bracket_tag" | "plain_label" | "blank_line" | "none"
    force_split: bool = False  # SPEC-02-PATCH item 6: level-4 length packing also applied


@dataclass(frozen=True)
class Hit:
    song_id: str
    score: float
    best_chunk: Chunk


@dataclass(frozen=True)
class SearchResult:
    """SPEC-03 §2: a query result that distinguishes "genuinely no
    matches" from "the query itself carried no usable signal" (OOV for
    tfidf/fasttext-avg, or literally empty after normalization).

    status:
      "ok"                       -- query had signal, hits reflect a real search
      "query_out_of_vocabulary"  -- query vector had ~zero norm, index was
                                     never even queried (no meaningless
                                     all-tied-at-zero scores)
      "empty_query"              -- query string itself was empty/whitespace
                                     after normalization, caught before encoding

    query_norm: L2 norm of the query vector for dense retrievers. For
    LexicalRetriever (BM25, no native vector/norm) this is instead the count
    of query tokens that matched the BM25 vocabulary, as a float -- 0.0 means
    OOV, consistent in spirit with the dense case (zero => no signal).

    branch_statuses: for HybridRetriever only -- records each underlying
    branch's own status (e.g. {"dense": "ok", "lexical":
    "query_out_of_vocabulary"}) so a partial degradation is visible instead
    of being silently absorbed into an overall "ok". None for non-hybrid
    retrievers.
    """

    hits: list[Hit]
    status: str
    query_norm: float
    branch_statuses: dict[str, str] | None = None


class Source(Protocol):
    def iter_songs(self) -> Iterator[RawSong]: ...


class Embedder(Protocol):
    """Turns texts into unit-norm vectors, one row per input text.

    EVAL-PREP §1: `encode()` may return either a dense `np.ndarray` of
    shape (len(texts), dim) or a `scipy.sparse` matrix of the same shape.
    That choice is a property of the representation, not of the caller:
    `tfidf` is ~0.14% dense on the full corpus, where a dense float32
    matrix would be 51.8 GiB against 39.3 MiB sparse. Every other embedder
    here returns dense.

    Consumers must therefore not assume `np.ndarray`. `core/vectors.py`
    holds the dense/sparse branch for row norms, row selection and
    dot-product scoring; `vector_store.py` holds it for the on-disk
    format. Cosine similarity is identical either way -- rows are
    L2-normalised in both representations, so scoring is a dot product.

    One consequence is stated rather than hidden: `FaissIndex` accepts
    dense input only, so a sparse embedder runs with the `numpy` index
    alone.
    """

    name: str
    dim: int

    def encode(self, texts: Sequence[str], *, is_query: bool) -> np.ndarray | sparse.spmatrix: ...


class Retriever(Protocol):
    def search(self, query: str, top_k: int) -> SearchResult: ...
