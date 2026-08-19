"""Turning chunks and questions into vectors.

One model serves both sides. Vectors are L2-normalised at encode time so that a
dot product is a cosine similarity, which is what the Chroma collection is
configured to use.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from src import config
from src.ingest import Chunk

from .cache import cache_key, load_cache, save_cache

if TYPE_CHECKING:  # importing torch costs seconds; keep it out of module import
    from sentence_transformers import SentenceTransformer


class EmbedStats(NamedTuple):
    """How much work an embed_chunks call actually did."""

    # cached + computed need not equal total: `computed` counts distinct texts
    # sent to the model, and several chunks can share one text.
    total: int      # chunks asked for
    computed: int   # distinct texts sent to the model
    cached: int     # chunks whose vector was already on disk

    @property
    def hit_rate(self) -> float:
        return self.cached / self.total if self.total else 0.0


def load_model(name: str | None = None) -> SentenceTransformer:
    """Load the configured sentence-transformer, once per model per process."""
    return _load_model(name or config.EMBEDDING_MODEL)


@lru_cache(maxsize=2)
def _load_model(name: str) -> SentenceTransformer:
    """Cached on the resolved name, so `load_model()` and `load_model(same)` share.

    torch is imported lazily here because it costs seconds, and `import
    src.index` should stay cheap for callers that only need the store or cache.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


def document_text(chunk: Chunk) -> str:
    """The exact text embedded for a chunk: a short context line, then the code.

    Naming the file, language, and symbol gives the embedder signal that raw
    code often lacks — a question like "where do we parse gitignore" matches the
    path and symbol long before it matches any identifier in the body.
    """
    return f"{chunk.file_path}\n{chunk.language} {chunk.kind} {chunk.symbol}\n\n{chunk.code}"


def embed_texts(texts: Sequence[str]) -> np.ndarray:
    """Encode raw texts into an (n, EMBEDDING_DIM) float32 matrix."""
    if not texts:
        return np.empty((0, config.EMBEDDING_DIM), dtype=np.float32)

    vectors = load_model().encode(
        list(texts),
        batch_size=config.EMBEDDING_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)


def embed_query(question: str) -> np.ndarray:
    """Encode a question into a single vector, applying the query-side prefix."""
    return embed_texts([config.EMBEDDING_QUERY_PREFIX + question])[0]


def embed_chunks(
    chunks: Sequence[Chunk],
    cache_path: str | Path | None = None,
) -> tuple[np.ndarray, EmbedStats]:
    """Embed chunks in order, reusing cached vectors and caching new ones.

    Deduplicates within the batch as well as against the cache: repeated
    boilerplate (identical `__init__.py` headers, generated accessors) is common
    in real repos and only needs encoding once.
    """
    texts = [document_text(chunk) for chunk in chunks]
    keys = [cache_key(text) for text in texts]

    cache = load_cache(cache_path) if cache_path else {}
    already_had = set(cache)

    # dict, not set, so the order sent to the model is deterministic.
    missing = {key: text for key, text in zip(keys, texts) if key not in cache}
    if missing:
        fresh = embed_texts(list(missing.values()))
        cache.update(zip(missing, fresh))
        if cache_path:
            save_cache(cache_path, cache)

    stats = EmbedStats(
        total=len(chunks),
        computed=len(missing),
        cached=sum(1 for key in keys if key in already_had),
    )
    if not chunks:
        return np.empty((0, config.EMBEDDING_DIM), dtype=np.float32), stats
    return np.stack([cache[key] for key in keys]), stats
