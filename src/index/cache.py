"""Content-addressed embedding cache.

Embedding is the slow part of indexing, and most of a repo is unchanged between
runs. Keying on a hash of the exact text that was embedded — rather than on a
file path or mtime — means an unchanged function is a cache hit even if its file
was touched, renamed, or moved, and a changed function is a guaranteed miss.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

from src import config


def cache_key(text: str, model_name: str | None = None) -> str:
    """Hash the embedded text together with the model that produced the vector.

    The model name is part of the key because vectors from different models are
    not interchangeable — swapping EMBEDDING_MODEL must invalidate everything,
    not silently mix two vector spaces in one collection.

    Defaults to None rather than to config.EMBEDDING_MODEL directly: a default
    argument is bound once at import, which would make the ablation's in-process
    model swaps write both models' vectors under the same key.
    """
    model_name = model_name or config.EMBEDDING_MODEL
    digest = hashlib.sha256()
    digest.update(model_name.encode("utf-8"))
    digest.update(b"\0")  # separator, so model+text can never alias
    digest.update(text.encode("utf-8"))
    return digest.hexdigest()


def cache_path_for(name: str) -> Path:
    """Path of the cache file for a named index (usually the collection name)."""
    return config.EMBEDDING_CACHE_DIR / f"{name}.npz"


def load_cache(path: str | Path) -> dict[str, np.ndarray]:
    """Read a cache file, returning an empty cache if it is missing or corrupt.

    A corrupt cache is a performance problem, not a correctness one: throwing it
    away costs one re-embed, whereas raising would block indexing entirely.
    """
    path = Path(path)
    if not path.exists():
        return {}

    try:
        with np.load(path, allow_pickle=False) as data:
            keys = data["keys"]
            vectors = data["vectors"]
    except (OSError, ValueError, KeyError):
        return {}

    if len(keys) != len(vectors):
        return {}
    return {str(key): vectors[i] for i, key in enumerate(keys)}


def save_cache(path: str | Path, cache: dict[str, np.ndarray]) -> None:
    """Write the cache atomically, so an interrupted run cannot corrupt it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # The npz format stacks every vector into one matrix, so entries must share
    # a shape — but after an embedding-model swap the loaded cache holds the
    # old model's dimension next to the new one's, and stacking them crashed a
    # real re-index. Keep only the active model's dimension: this is a cache,
    # not an archive, and a swap-back simply re-embeds.
    keys = [k for k in cache if cache[k].shape == (config.EMBEDDING_DIM,)]
    if keys:
        vectors = np.stack([cache[key] for key in keys]).astype(np.float32)
    else:
        vectors = np.empty((0, config.EMBEDDING_DIM), dtype=np.float32)

    tmp = path.with_name(path.name + ".tmp")
    np.savez(tmp, keys=np.array(keys, dtype=np.str_), vectors=vectors)
    # np.savez appends .npz unless the target already ends in it.
    written = tmp if tmp.exists() else tmp.with_name(tmp.name + ".npz")
    os.replace(written, path)
