"""Embeddings, the Chroma vector store, and the BM25 keyword index.

Both indexes are built from the same chunks and the same document text, so the
ablation compares retrieval methods rather than corpora. `store.search` and
`BM25Index.search` are each callable on their own; fusion lives in src/retrieve.
"""

from .bm25 import BM25Index, bm25_path_for
from .cache import cache_key, cache_path_for, load_cache, save_cache
from .embedder import EmbedStats, document_text, embed_chunks, embed_query, embed_texts, load_model
from .models import SearchResult
from .pipeline import IndexStats, index_repo
from .store import (
    chunk_id,
    collection_name,
    get_collection,
    prune_to,
    search,
    upsert_chunks,
)
from .tokenize import split_identifier, tokenize

__all__ = [
    "BM25Index",
    "EmbedStats",
    "IndexStats",
    "SearchResult",
    "bm25_path_for",
    "cache_key",
    "cache_path_for",
    "chunk_id",
    "collection_name",
    "document_text",
    "embed_chunks",
    "embed_query",
    "embed_texts",
    "get_collection",
    "index_repo",
    "load_cache",
    "load_model",
    "prune_to",
    "save_cache",
    "search",
    "split_identifier",
    "tokenize",
    "upsert_chunks",
]
