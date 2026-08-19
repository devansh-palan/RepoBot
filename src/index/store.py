"""The persistent Chroma collection: writing chunks in, querying chunks out.

This module is the **vector-only** retriever. It is deliberately usable on its
own, without BM25 or fusion, because the ablation study needs to measure it in
isolation.
"""

from __future__ import annotations

import hashlib
import re
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src import config
from src.ingest import Chunk

from .embedder import embed_query
from .models import SearchResult

if TYPE_CHECKING:
    from chromadb.api import ClientAPI
    from chromadb.api.models.Collection import Collection

# Chroma names must be 3-512 chars of [a-zA-Z0-9._-], starting and ending
# alphanumeric. Anything else in a directory name is folded to an underscore.
_UNSAFE = re.compile(r"[^a-zA-Z0-9_-]+")

# Collection metadata key recording which model produced the stored vectors.
_MODEL_STAMP = "embedding_model"


def collection_name(repo_path: str | Path) -> str:
    """A stable, Chroma-legal collection name for a repository.

    The path hash is part of the name so two checkouts that happen to share a
    directory name do not silently write into each other's collection.
    """
    resolved = Path(repo_path).resolve()
    slug = _UNSAFE.sub("_", resolved.name).strip("_-") or "repo"
    digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:8]
    return f"repo_{slug[:40]}_{digest}"


@lru_cache(maxsize=None)
def get_client(persist_dir: str | None = None) -> ClientAPI:
    """Open the persistent Chroma client, once per process and directory."""
    import chromadb

    path = Path(persist_dir) if persist_dir else config.CHROMA_DIR
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path))


def get_collection(repo_path: str | Path, persist_dir: str | None = None) -> Collection:
    """Get or create the collection for a repo, configured for cosine distance.

    `embedding_function=None` matters: without it Chroma would install and run
    its own default embedder, and we always supply vectors ourselves.

    A collection built by a different embedding model is dropped and rebuilt.
    Two models can share a dimension without sharing a vector space, so Chroma
    would accept the mixture happily and simply return worse results forever —
    a silent failure, and the expensive kind to notice.
    """
    client = get_client(persist_dir)
    name = collection_name(repo_path)
    options = {
        "name": name,
        "embedding_function": None,
        "metadata": {_MODEL_STAMP: config.EMBEDDING_MODEL},
        "configuration": {"hnsw": {"space": "cosine"}},
    }

    collection = client.get_or_create_collection(**options)
    # Chroma keeps the metadata from creation time and ignores it on re-get,
    # so this reads what actually built the collection, not what we asked for.
    stamped = (collection.metadata or {}).get(_MODEL_STAMP)
    if stamped != config.EMBEDDING_MODEL:
        warnings.warn(
            f"{name} was built with {stamped!r}; its vectors are unusable under "
            f"{config.EMBEDDING_MODEL!r} and have been dropped. Re-index to rebuild.",
            stacklevel=2,
        )
        client.delete_collection(name)
        collection = client.get_or_create_collection(**options)
    return collection


def chunk_id(chunk: Chunk) -> str:
    """Stable id for a chunk, so re-indexing updates rather than duplicates."""
    return f"{chunk.file_path}:{chunk.start_line}-{chunk.end_line}#{chunk.part}"


def _metadata(chunk: Chunk) -> dict[str, str | int]:
    """Everything needed to rebuild a Chunk. Chroma allows only scalar values."""
    return {
        "language": chunk.language,
        "file_path": chunk.file_path,
        "symbol": chunk.symbol,
        "kind": chunk.kind,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "part": chunk.part,
        "part_count": chunk.part_count,
    }


def _to_chunk(document: str, metadata: dict[str, Any]) -> Chunk:
    """Rebuild a Chunk from a stored document and its metadata."""
    return Chunk(
        code=document,
        language=str(metadata["language"]),
        file_path=str(metadata["file_path"]),
        symbol=str(metadata["symbol"]),
        kind=str(metadata["kind"]),
        start_line=int(metadata["start_line"]),
        end_line=int(metadata["end_line"]),
        part=int(metadata["part"]),
        part_count=int(metadata["part_count"]),
    )


def upsert_chunks(
    collection: Collection,
    chunks: Sequence[Chunk],
    vectors: np.ndarray,
) -> int:
    """Write chunks and their vectors. Returns the number of rows written.

    Upsert rather than add so a re-index of a modified file overwrites the old
    rows in place instead of failing on duplicate ids.
    """
    if len(chunks) != len(vectors):
        raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")

    size = config.CHROMA_BATCH_SIZE
    for start in range(0, len(chunks), size):
        batch = chunks[start : start + size]
        collection.upsert(
            ids=[chunk_id(c) for c in batch],
            embeddings=vectors[start : start + size].tolist(),
            documents=[c.code for c in batch],
            metadatas=[_metadata(c) for c in batch],
        )
    return len(chunks)


def prune_to(collection: Collection, keep_ids: Iterable[str]) -> int:
    """Delete stored chunks that are no longer produced by ingest.

    Without this, deleting a function would leave its vector in the collection
    forever and the agent would keep citing code that does not exist.
    """
    keep = set(keep_ids)
    stale = [i for i in collection.get(include=[])["ids"] if i not in keep]

    size = config.CHROMA_BATCH_SIZE
    for start in range(0, len(stale), size):
        collection.delete(ids=stale[start : start + size])
    return len(stale)


def search(
    collection: Collection,
    question: str,
    k: int = config.VECTOR_TOP_K,
) -> list[SearchResult]:
    """Vector-only retrieval: embed the question, return the k nearest chunks.

    Chroma reports cosine *distance*; similarity is 1 - distance.
    """
    stored = collection.count()
    if k <= 0 or stored == 0:
        return []

    response = collection.query(
        query_embeddings=[embed_query(question).tolist()],
        n_results=min(k, stored),
        include=["documents", "metadatas", "distances"],
    )

    # Chroma nests one list per query; we only ever send one.
    documents = response["documents"][0]
    metadatas = response["metadatas"][0]
    distances = response["distances"][0]

    return [
        SearchResult(
            chunk=_to_chunk(doc, meta),
            score=1.0 - float(dist),
            rank=rank,
            contributions={"vector": rank},
        )
        for rank, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), start=1)
    ]
