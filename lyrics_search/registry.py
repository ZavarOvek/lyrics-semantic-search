"""SPEC-03 §5 / SPEC-03-PATCH §1: registry -- name -> factory, so a config's
embedder/index selection (a plain string, or a name + params mapping) can
build a real object without a hardcoded if/elif chain living in every
runner.

Factories take `**params` and forward them verbatim to the real class's
`__init__` -- so each embedder's own constructor signature *is* its
config-params schema, with no separate translation layer to keep in sync.
Each factory does its own import *inside* the function body, not at module
level -- so importing lyrics_search.registry itself never pulls in
torch/sentence-transformers/faiss/gensim; that cost is only paid when a
specific name is actually resolved (@register still runs the decorator at
import time, but the decorator only stores the function object, it doesn't
call it).

SPEC-03-PATCH §1: fasttext-avg is now registered. Its constructor requires
`vectors_path` (a large local file, only ever supplied after explicit user
permission per SPEC-00's download-permission rule) -- `required_params`
declares that at registration time so config.py can fail loudly at
*config-load* time on a missing param, without importing gensim or touching
the multi-GB vector file just to check.
"""

from __future__ import annotations

from collections.abc import Callable

_EMBEDDERS: dict[str, Callable[..., object]] = {}
_EMBEDDER_REQUIRED_PARAMS: dict[str, tuple[str, ...]] = {}
_INDEXES: dict[str, Callable[..., object]] = {}


def register_embedder(name: str, required_params: tuple[str, ...] = ()):
    def decorator(factory: Callable[..., object]) -> Callable[..., object]:
        _EMBEDDERS[name] = factory
        _EMBEDDER_REQUIRED_PARAMS[name] = required_params
        return factory

    return decorator


def register_index(name: str):
    def decorator(factory: Callable[..., object]) -> Callable[..., object]:
        _INDEXES[name] = factory
        return factory

    return decorator


def resolve_embedder(name: str, params: dict | None = None) -> object:
    if name not in _EMBEDDERS:
        raise KeyError(f"Unknown embedder {name!r}. Registered: {registered_embedder_names()}")
    return _EMBEDDERS[name](**(params or {}))


def resolve_index(name: str) -> object:
    if name not in _INDEXES:
        raise KeyError(f"Unknown index {name!r}. Registered: {registered_index_names()}")
    return _INDEXES[name]()


def registered_embedder_names() -> list[str]:
    return sorted(_EMBEDDERS)


def registered_index_names() -> list[str]:
    return sorted(_INDEXES)


def required_embedder_params(name: str) -> tuple[str, ...]:
    return _EMBEDDER_REQUIRED_PARAMS.get(name, ())


@register_embedder("bge-m3")
def _make_bge_m3(**params) -> object:
    from lyrics_search.embedders.bge_m3 import BGEM3Embedder

    return BGEM3Embedder(**params)


@register_embedder("e5-base")
def _make_e5_base(**params) -> object:
    from lyrics_search.embedders.e5 import E5Embedder

    return E5Embedder(**params)


@register_embedder("tfidf")
def _make_tfidf(**params) -> object:
    from lyrics_search.embedders.tfidf import TfidfEmbedder

    return TfidfEmbedder(**params)


@register_embedder("fasttext-avg", required_params=("vectors_path",))
def _make_fasttext_avg(**params) -> object:
    from lyrics_search.embedders.fasttext_avg import FastTextAvgEmbedder

    return FastTextAvgEmbedder(**params)


@register_index("numpy")
def _make_numpy_index() -> object:
    from lyrics_search.indexes.numpy_index import NumpyIndex

    return NumpyIndex()


@register_index("faiss")
def _make_faiss_index() -> object:
    from lyrics_search.indexes.faiss_index import FaissIndex

    return FaissIndex()
