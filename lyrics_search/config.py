"""SPEC-03 §5 / SPEC-03-PATCH §1: experiment config -- one YAML file fully
describes one run.

An `ExperimentConfig` picks which corpus, embedder, and index to use (by
name, resolved later through `registry.py`) plus retrieval-mode settings.
Validation happens in two layers: pydantic checks shape/types, and
`load_config()` additionally cross-checks `embedder.name`/`index` against
the registry's currently-registered names, and `embedder.params` against
that embedder's declared `required_params`, so a typo, an unregistered
name, or a missing required param (e.g. fasttext-avg's `vectors_path`)
fails loudly at config-load time ("at startup") rather than later when the
registry itself is resolved deep inside a build/search run.

`embedder` accepts two YAML shapes, normalized to the same `EmbedderConfig`:
    embedder: bge-m3                              # plain name, params={}
    embedder:                                     # name + params
      name: fasttext-avg
      params:
        vectors_path: data/models/fasttext/cc.en.300.vec
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from lyrics_search.core.fusion import RRF_DEFAULT_K


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["dense", "lexical", "hybrid"] = "dense"
    rrf_k: int = RRF_DEFAULT_K
    top_k: int = 50
    return_n: int = 10


class EmbedderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    corpus: Literal["dev", "full", "demo"] = "dev"
    # EVAL-PREP §3. Also a path component under the corpus dir (see
    # runners/build.py), so the two arms' artifacts sit side by side
    # instead of evicting each other.
    chunking: Literal["sections", "whole_song"] = "sections"
    embedder: EmbedderConfig = Field(default_factory=lambda: EmbedderConfig(name="bge-m3"))
    index: str = "numpy"
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)

    @field_validator("embedder", mode="before")
    @classmethod
    def _coerce_embedder(cls, v: Any) -> Any:
        # Plain-string shorthand (`embedder: bge-m3`) -> {"name": "bge-m3"}.
        return {"name": v} if isinstance(v, str) else v


def load_config(path: Path | str) -> ExperimentConfig:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    config = ExperimentConfig.model_validate(raw)
    _validate_registry_names(config)
    return config


def _validate_registry_names(config: ExperimentConfig) -> None:
    # Imported lazily so importing config.py itself never pulls in the
    # registry's (eventually torch/faiss/gensim-touching) factory modules.
    from lyrics_search import registry

    known_embedders = registry.registered_embedder_names()
    if config.embedder.name not in known_embedders:
        raise ValueError(
            f"Unknown embedder {config.embedder.name!r} in config. "
            f"Registered embedders: {known_embedders}"
        )
    missing_params = [
        p for p in registry.required_embedder_params(config.embedder.name)
        if p not in config.embedder.params
    ]
    if missing_params:
        raise ValueError(
            f"Embedder {config.embedder.name!r} is missing required params "
            f"{missing_params} under `embedder.params` in config."
        )
    known_indexes = registry.registered_index_names()
    if config.index not in known_indexes:
        raise ValueError(
            f"Unknown index {config.index!r} in config. "
            f"Registered indexes: {known_indexes}"
        )
