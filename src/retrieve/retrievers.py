"""The three retrievers, each callable on its own.

Separate entry points are a requirement, not a convenience: the ablation has to
measure vector-only and bm25-only against hybrid, and it can only do that if
none of them is reachable exclusively through the others.
"""

from __future__ import annotations

import re
from pathlib import Path

from src import config
from src.index import BM25Index, SearchResult, bm25_path_for, collection_name, get_collection
from src.index import search as vector_query

from .fusion import reciprocal_rank_fusion


class MissingIndexError(RuntimeError):
    """Raised when a repo has no BM25 index on disk yet."""


def vector_search(
    repo_path: str | Path,
    question: str,
    k: int = config.VECTOR_TOP_K,
    persist_dir: str | None = None,
) -> list[SearchResult]:
    """Dense retrieval alone: nearest neighbours by cosine similarity."""
    return vector_query(get_collection(repo_path, persist_dir), question, k)


def _bm25_path(repo_path: str | Path, bm25_dir: str | Path | None) -> Path:
    name = collection_name(repo_path)
    return (Path(bm25_dir) / f"{name}.pkl") if bm25_dir else bm25_path_for(name)


# Loaded indexes, keyed by (path, mtime, size). Without this every request
# unpickles the index AND rebuilds BM25Okapi from scratch — hundreds of ms of
# pure overhead per question in the server, which holds no other per-repo
# state. Keying on mtime+size means re-indexing invalidates naturally; stale
# entries for old mtimes are dropped so a long-lived server does not accumulate
# one dead index per re-index.
_BM25_CACHE: dict[tuple[str, float, int], BM25Index] = {}


def load_bm25(repo_path: str | Path, bm25_dir: str | Path | None = None) -> BM25Index:
    """Read the keyword index for a repo, or explain that it needs building."""
    path = _bm25_path(repo_path, bm25_dir)
    try:
        stat = path.stat()
    except OSError:
        raise MissingIndexError(
            f"no BM25 index at {path}; run index_repo({repo_path!r}) first"
        ) from None

    key = (str(path.resolve()), stat.st_mtime, stat.st_size)
    cached = _BM25_CACHE.get(key)
    if cached is not None:
        return cached

    index = BM25Index.load(path)
    if index is None:
        raise MissingIndexError(f"no BM25 index at {path}; run index_repo({repo_path!r}) first")

    for old in [k for k in _BM25_CACHE if k[0] == key[0]]:
        del _BM25_CACHE[old]
    _BM25_CACHE[key] = index
    return index


def bm25_search(
    repo_path: str | Path,
    question: str,
    k: int = config.BM25_TOP_K,
    bm25_dir: str | Path | None = None,
) -> list[SearchResult]:
    """Keyword retrieval alone: BM25 over code-aware tokens."""
    return load_bm25(repo_path, bm25_dir).search(question, k)


# A query that names a code identifier: backticked code, snake_case, camelCase,
# a dotted call, or a dunder. "Where is parse_jwt defined" matches; "how are
# redirects resolved" does not. This decides BM25's weight in the fusion, so a
# false positive costs a little semantic ranking and a false negative costs the
# exact-match boost — both degrade softly rather than failing.
_IDENTIFIER = re.compile(
    r"`[^`]+`"                           # `anything in backticks`
    r"|\b[A-Za-z][a-z0-9]*_[A-Za-z0-9_]+\b"  # snake_case
    r"|\b\w*[a-z][A-Z]\w*\b"             # camelCase / PascalCase: parseJwt, HttpClient
    r"|\b[A-Z]{2,}[a-z]\w+\b"            # acronym-prefixed: HTTPAdapter (but not URLs)
    r"|\b\w+\.\w+\("                     # dotted call: requests.get(
    r"|__\w+__"                          # dunder
)


def mentions_identifier(question: str) -> bool:
    """Whether the query quotes a code identifier, verbatim.

    When it does, the user has handed us the exact token to look up and BM25 is
    the right instrument; when it does not, BM25 mostly matches question words
    against docstrings and pollutes the fusion.
    """
    return bool(_IDENTIFIER.search(question))


def hybrid_search(
    repo_path: str | Path,
    question: str,
    k: int = config.FINAL_TOP_K,
    persist_dir: str | None = None,
    bm25_dir: str | Path | None = None,
) -> list[SearchResult]:
    """Run both retrievers and fuse their rankings with query-weighted RRF.

    Each retriever contributes its own top-k (VECTOR_TOP_K / BM25_TOP_K), which
    are deliberately larger than the k returned here: fusion can only promote a
    chunk that at least one retriever surfaced, so the candidate pools need
    headroom for the other's misses.

    BM25's weight depends on the query (see mentions_identifier and the
    RRF_BM25_WEIGHT_* comment in config). At weight 0.0 the BM25 search is
    skipped entirely — running it would cost time and change nothing.
    """
    bm25_weight = (
        config.RRF_BM25_WEIGHT_LEXICAL
        if mentions_identifier(question)
        else config.RRF_BM25_WEIGHT_SEMANTIC
    )

    if bm25_weight <= 0.0 and not _bm25_path(repo_path, bm25_dir).is_file():
        # Both indexes are built together, so a missing BM25 index means the
        # repo was never indexed. Without this check a semantic query against
        # an unindexed repo would quietly search an empty vector collection
        # instead of saying "run index_repo first".
        raise MissingIndexError(
            f"no index for {repo_path}; run index_repo({str(repo_path)!r}) first"
        )

    rankings = {"vector": vector_search(repo_path, question, config.VECTOR_TOP_K, persist_dir)}
    if bm25_weight > 0.0:
        rankings["bm25"] = bm25_search(repo_path, question, config.BM25_TOP_K, bm25_dir)

    return reciprocal_rank_fusion(rankings, top_k=k, weights={"bm25": bm25_weight})


def hybrid_rerank_search(
    repo_path: str | Path,
    question: str,
    k: int = config.FINAL_TOP_K,
    persist_dir: str | None = None,
    bm25_dir: str | Path | None = None,
) -> list[SearchResult]:
    """Hybrid retrieval widened to RERANK_CANDIDATES, then cross-encoder ranked.

    The wide pool is the point: the reranker can only promote what retrieval
    surfaced, and its job is to correct fusion's ordering mistakes.
    """
    from .rerank import rerank  # lazy: pulls in torch

    pool = hybrid_search(
        repo_path, question, max(k, config.RERANK_CANDIDATES), persist_dir, bm25_dir
    )
    return rerank(question, pool, top_k=k)


#: Name -> retriever, so the ablation can loop over all of them uniformly.
RETRIEVERS = {
    "vector": vector_search,
    "bm25": bm25_search,
    "hybrid": hybrid_search,
    "hybrid_rerank": hybrid_rerank_search,
}
