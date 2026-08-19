"""Reciprocal Rank Fusion.

    score(chunk) = sum over retrievers of  1 / (RRF_K + rank_in_that_retriever)

Fusing on *ranks* rather than scores is the whole idea. A cosine similarity of
0.72 and a BM25 score of 8.3 have no common scale, and normalising them means
inventing one. Ranks are already comparable, so RRF needs no calibration, no
tuning per corpus, and no assumption about either retriever's score
distribution — it only asks "how near the top did each one put this?".

A chunk found by both retrievers beats a chunk found deeply by one, which is
exactly the behaviour hybrid retrieval is for.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from src import config
from src.index import SearchResult, chunk_id


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[SearchResult]],
    k: int = config.RRF_K,
    top_k: int = config.FINAL_TOP_K,
    weights: Mapping[str, float] | None = None,
) -> list[SearchResult]:
    """Merge several ranked lists into one.

    `rankings` maps a retriever name to its results, best first. Each result's
    own `rank` field is used, so a caller may pass a truncated list without the
    arithmetic silently shifting.

    `weights` scales each retriever's vote (missing names default to 1.0). A
    weight of 0.0 removes that retriever entirely — its chunks must not pad the
    tail of the fused list with zero scores.
    """
    scores: dict[str, float] = {}
    contributions: dict[str, dict[str, int]] = {}
    chunks: dict[str, SearchResult] = {}

    for retriever, results in rankings.items():
        weight = 1.0 if weights is None else weights.get(retriever, 1.0)
        if weight == 0.0:
            continue
        for result in results:
            key = chunk_id(result.chunk)
            scores[key] = scores.get(key, 0.0) + weight / (k + result.rank)
            contributions.setdefault(key, {})[retriever] = result.rank
            chunks.setdefault(key, result)

    # Sorted by score, then by id: ties are common when every retriever agrees,
    # and an arbitrary tiebreak would make results differ run to run.
    ordered = sorted(scores, key=lambda key: (-scores[key], key))

    return [
        SearchResult(
            chunk=chunks[key].chunk,
            score=scores[key],
            rank=rank,
            contributions=contributions[key],
        )
        for rank, key in enumerate(ordered[:top_k], start=1)
    ]
