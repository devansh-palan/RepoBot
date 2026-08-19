"""CLI for retrieval: `python -m src.retrieve <repo> -q "..." --explain`.

`--explain` prints both input rankings and the RRF arithmetic that produced the
fused order, which is the fastest way to see why hybrid beat either half.
"""

from __future__ import annotations

import argparse

from src import config
from src.index import SearchResult

from .fusion import reciprocal_rank_fusion
from .retrievers import bm25_search, hybrid_search, vector_search


def _show(title: str, results: list[SearchResult]) -> None:
    print(f"\n{title}")
    if not results:
        print("   (nothing)")
    for result in results:
        print(f"  {result.rank:>2}. {result.score:8.4f}  {result.chunk.header()}")


def _explain(vector: list[SearchResult], bm25: list[SearchResult], k: int) -> None:
    """Print the fusion with its arithmetic spelled out term by term."""
    fused = reciprocal_rank_fusion({"vector": vector, "bm25": bm25}, top_k=k)

    print(f"\nRRF fusion  (k={config.RRF_K}, top {k})")
    print(f"  {'#':>2}  {'score':>8}  {'= sum of 1/(k + rank)':<40}  chunk")
    for result in fused:
        terms = " + ".join(
            f"1/({config.RRF_K}+{rank}) [{name} #{rank}]"
            for name, rank in sorted(result.contributions.items())
        )
        both = "  <- both" if len(result.contributions) > 1 else ""
        print(f"  {result.rank:>2}  {result.score:8.5f}  {terms:<40}  {result.chunk.header()}{both}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m src.retrieve", description=__doc__)
    parser.add_argument("repo", help="path to an already-indexed repository")
    parser.add_argument("--query", "-q", required=True, help="the question")
    parser.add_argument("-k", type=int, default=config.FINAL_TOP_K, help="results to show")
    parser.add_argument(
        "--retriever",
        choices=("vector", "bm25", "hybrid", "all"),
        default="hybrid",
    )
    parser.add_argument("--explain", action="store_true", help="show the RRF arithmetic")
    args = parser.parse_args()

    print(f"query: {args.query}")

    if args.explain or args.retriever == "all":
        vector = vector_search(args.repo, args.query, config.VECTOR_TOP_K)
        bm25 = bm25_search(args.repo, args.query, config.BM25_TOP_K)
        _show(f"vector-only (top {args.k} of {len(vector)} candidates)", vector[: args.k])
        _show(f"bm25-only   (top {args.k} of {len(bm25)} candidates)", bm25[: args.k])
        _explain(vector, bm25, args.k)
        return

    results = {"vector": vector_search, "bm25": bm25_search, "hybrid": hybrid_search}[
        args.retriever
    ](args.repo, args.query, args.k)
    _show(f"{args.retriever} (top {args.k})", results)


if __name__ == "__main__":
    main()
