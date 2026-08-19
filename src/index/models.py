"""The result type every retriever returns.

Lives here rather than in store.py so that bm25.py does not have to import the
Chroma layer to describe its own output, and so the ablation can treat all three
retrievers identically.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.ingest import Chunk


@dataclass(frozen=True)
class SearchResult:
    """One retrieved chunk and its position in a ranking.

    `score` is retriever-specific and **not comparable across retrievers**:
    cosine similarity for vector search, a BM25 score for keyword search, an RRF
    score after fusion. Rank is the portable quantity — which is exactly why
    RRF fuses on ranks rather than on scores.
    """

    chunk: Chunk
    score: float
    rank: int  # 1-based, so it can feed RRF directly

    # Which retrievers found this chunk and at what rank: {"vector": 3,
    # "bm25": 1}. A single retriever records itself, so fused and unfused
    # results have the same shape and one metrics function handles both.
    contributions: dict[str, int] = field(default_factory=dict)

    def explain(self) -> str:
        """Human-readable provenance, e.g. `vector #3 + bm25 #1`."""
        if not self.contributions:
            return "-"
        return " + ".join(f"{name} #{rank}" for name, rank in sorted(self.contributions.items()))

    def __str__(self) -> str:
        return f"{self.rank:>2}. {self.score:.4f}  {self.chunk.header()}  [{self.explain()}]"
