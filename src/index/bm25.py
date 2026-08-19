"""The BM25 keyword index — the second half of hybrid retrieval.

Built from the same chunks and the same document text as the vector index, so
that any difference the ablation measures comes from the retrieval method and
not from what each index was fed.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from src import config
from src.ingest import Chunk

from .embedder import document_text
from .models import SearchResult
from .tokenize import tokenize


def bm25_path_for(name: str) -> Path:
    """Path of the pickled index for a named collection."""
    return config.BM25_DIR / f"{name}.pkl"


@dataclass
class BM25Index:
    """An in-memory BM25 index over a fixed list of chunks."""

    chunks: list[Chunk]
    tokens: list[list[str]]

    def __post_init__(self) -> None:
        # Rebuilt rather than stored: it is only IDF arithmetic over `tokens`,
        # and pickling a third-party object would tie the on-disk format to the
        # installed rank_bm25 version.
        self._bm25 = (
            BM25Okapi(self.tokens, k1=config.BM25_K1, b=config.BM25_B) if self.tokens else None
        )

    def __len__(self) -> int:
        return len(self.chunks)

    @classmethod
    def build(cls, chunks: list[Chunk]) -> BM25Index:
        """Tokenize every chunk and build the index."""
        return cls(chunks=list(chunks), tokens=[tokenize(document_text(c)) for c in chunks])

    def save(self, path: str | Path) -> None:
        """Persist chunks and their tokens.

        NOTE: pickle executes arbitrary code on load. This file is written by
        us into our own data directory, so that is acceptable here; a service
        loading indexes from untrusted input would need a safe format instead.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"chunks": self.chunks, "tokens": self.tokens}

        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
        tmp.replace(path)  # atomic; a killed run cannot leave a half-written index

    @classmethod
    def load(cls, path: str | Path) -> BM25Index | None:
        """Read an index back, or None if it is missing or unreadable.

        Same reasoning as the embedding cache: a damaged index is a rebuild, not
        a crash.
        """
        path = Path(path)
        if not path.exists():
            return None
        try:
            payload = pickle.loads(path.read_bytes())
            return cls(chunks=payload["chunks"], tokens=payload["tokens"])
        except (OSError, pickle.UnpicklingError, KeyError, EOFError, AttributeError):
            return None

    def search(self, question: str, k: int = config.BM25_TOP_K) -> list[SearchResult]:
        """Keyword-only retrieval: the k highest-scoring chunks for a question."""
        if k <= 0 or self._bm25 is None:
            return []

        query = tokenize(question)
        if not query:
            return []

        scores = self._bm25.get_scores(query)

        # A zero score means no query term is present at all. Those are not
        # weak matches, they are non-matches, and letting them fill out the
        # top-k would hand RRF a ranking of pure noise.
        ranked = np.argsort(-scores, kind="stable")[: min(k, len(scores))]
        return [
            SearchResult(
                chunk=self.chunks[i],
                score=float(scores[i]),
                rank=rank,
                contributions={"bm25": rank},
            )
            for rank, i in enumerate(ranked, start=1)
            if scores[i] > 0
        ]
