"""Repo path in, both indexes out — the whole index stage in one call."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from src.ingest import Chunk, ingest_repo

from .bm25 import BM25Index, bm25_path_for
from .cache import cache_path_for
from .embedder import embed_chunks
from .store import chunk_id, collection_name, get_collection, prune_to, upsert_chunks


@dataclass(frozen=True)
class IndexStats:
    """What one indexing run did. Printed by the CLI, asserted on by tests."""

    collection: str
    chunks: int
    embedded: int  # distinct texts sent to the model
    cached: int    # chunks served from the embedding cache
    pruned: int    # stale rows deleted from the collection
    keywords: int  # chunks in the BM25 index

    def __str__(self) -> str:
        return (
            f"{self.collection}: {self.chunks} chunks "
            f"({self.embedded} embedded, {self.cached} cached, {self.pruned} pruned; "
            f"{self.keywords} in bm25)"
        )


def _deduplicate(chunks: list[Chunk]) -> list[Chunk]:
    """Keep one chunk per id.

    Chunk spans are disjoint by construction, so this should never drop
    anything — it is here so that a future chunker bug degrades into a slightly
    smaller index rather than a silent, order-dependent overwrite inside Chroma.
    """
    seen: dict[str, Chunk] = {}
    for chunk in chunks:
        seen.setdefault(chunk_id(chunk), chunk)
    return list(seen.values())


def index_repo(
    repo_path: str | Path,
    persist_dir: str | None = None,
    cache_dir: str | Path | None = None,
    bm25_dir: str | Path | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> IndexStats:
    """Chunk a repository, then build both the vector and the keyword index.

    One ingest feeds both. That is deliberate: if each index chunked the repo
    separately they could drift, and the ablation would be comparing corpora
    rather than retrieval methods.

    Safe and cheap to re-run: unchanged chunks hit the embedding cache, changed
    ones overwrite their old row by id, and deleted ones are pruned.
    """
    root = Path(repo_path).resolve()
    name = collection_name(root)

    # Progress is coarse on purpose — stage boundaries, not percentages. The
    # serve layer streams these lines to a user waiting on a first index, where
    # embedding dominates and can take minutes on CPU.
    def progress(message: str) -> None:
        if on_progress:
            on_progress(message)

    progress("reading and chunking source files")
    chunks = _deduplicate(ingest_repo(root))
    progress(f"chunked into {len(chunks)} pieces of code")

    cache_file = (Path(cache_dir) / f"{name}.npz") if cache_dir else cache_path_for(name)
    bm25_file = (Path(bm25_dir) / f"{name}.pkl") if bm25_dir else bm25_path_for(name)
    progress("embedding chunks (the slow part on a first run)")
    vectors, stats = embed_chunks(chunks, cache_file)
    progress(f"embedded {stats.computed} chunks, {stats.cached} from cache")

    collection = get_collection(root, persist_dir)
    upsert_chunks(collection, chunks, vectors)
    progress("building the keyword index")

    # Pruning to an empty chunk list would delete the whole collection. That is
    # right when a repo really has no code and catastrophic when ingest broke —
    # a missing grammar wheel or a bad path silently destroys a good index. A
    # stale index with a loud warning is the better failure; delete the
    # collection by hand if the repo is genuinely empty now.
    if not chunks and collection.count():
        warnings.warn(
            f"ingest produced no chunks for {root}; keeping the "
            f"{collection.count()} chunks already indexed rather than wiping them",
            stacklevel=2,
        )
        pruned = 0
        keywords = len(BM25Index.load(bm25_file) or [])
    else:
        pruned = prune_to(collection, (chunk_id(c) for c in chunks))
        # Rebuilt wholesale rather than updated: BM25's IDF and average document
        # length depend on the entire corpus, so an incremental update would
        # leave every stored score subtly stale.
        keyword_index = BM25Index.build(chunks)
        keyword_index.save(bm25_file)
        keywords = len(keyword_index)

    return IndexStats(
        collection=name,
        chunks=len(chunks),
        embedded=stats.computed,
        cached=stats.cached,
        pruned=pruned,
        keywords=keywords,
    )
