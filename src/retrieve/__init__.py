"""Hybrid retrieval and RRF fusion.

Three retrievers over one corpus. `vector_search` and `bm25_search` stay
independently callable so the ablation can show what fusion actually buys.
"""

from src.index import SearchResult

from .fusion import reciprocal_rank_fusion
# NOTE: the rerank *function* is deliberately not re-exported: `from .rerank
# import rerank` here would shadow the `src.retrieve.rerank` submodule with
# the function (the same module/attribute collision that forced serve's
# app.py -> api.py rename), breaking attribute-path imports and patching.
# Callers use hybrid_rerank_search, or import from src.retrieve.rerank.
from .retrievers import (
    RETRIEVERS,
    MissingIndexError,
    bm25_search,
    hybrid_rerank_search,
    hybrid_search,
    load_bm25,
    vector_search,
)

__all__ = [
    "RETRIEVERS",
    "MissingIndexError",
    "SearchResult",
    "bm25_search",
    "hybrid_rerank_search",
    "hybrid_search",
    "load_bm25",
    "reciprocal_rank_fusion",
    "vector_search",
]
