"""Cross-encoder reranking over the hybrid candidates.

The bi-encoder scores question and chunk from two separate embeddings; a
cross-encoder reads the concatenated pair and scores the actual interaction.
That is much more accurate and much slower, which dictates the architecture:
the cheap retrievers fetch a wide pool, the cross-encoder reorders only that
pool, and only the top of the reordered list reaches the model.
"""

from __future__ import annotations

from functools import lru_cache

from src import config
from src.index import SearchResult


def load_reranker():
    """The cross-encoder, cached per process (resolved at call time so the
    ablation can swap models by changing config)."""
    return _load_reranker(config.RERANKER_MODEL)


@lru_cache(maxsize=2)
def _load_reranker(model_name: str):
    from sentence_transformers import CrossEncoder  # lazy: pulls in torch

    return CrossEncoder(model_name)


def rerank(
    question: str,
    candidates: list[SearchResult],
    top_k: int = config.FINAL_TOP_K,
) -> list[SearchResult]:
    """Reorder retrieval candidates by cross-encoder score, keep the best.

    The document side matches the embedder's `document_text` framing — path
    and symbol first — so the reranker sees the same signals retrieval scored.
    Contributions are preserved: the fused rank a chunk arrived with is still
    part of its explanation.
    """
    if not candidates:
        return []

    model = load_reranker()
    pairs = [
        (question, f"{r.chunk.file_path}\n{r.chunk.kind} {r.chunk.symbol}\n{r.chunk.code}")
        for r in candidates
    ]
    scores = model.predict(pairs)

    # Ties broken by the incoming rank so equal scores keep retrieval's order.
    reordered = sorted(
        zip(scores, candidates), key=lambda pair: (-float(pair[0]), pair[1].rank)
    )
    return [
        SearchResult(
            chunk=result.chunk,
            score=float(score),
            rank=rank,
            contributions=result.contributions,
        )
        for rank, (score, result) in enumerate(reordered[:top_k], start=1)
    ]
