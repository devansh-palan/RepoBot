"""Hybrid retrieval and RRF fusion.

Three retrievers over one corpus. `vector_search` and `bm25_search` stay
independently callable so the ablation can show what fusion actually buys.
"""

from src.index import SearchResult

from .fusion import reciprocal_rank_fusion
from .rerank import rerank
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
    "rerank",
    "vector_search",
]
