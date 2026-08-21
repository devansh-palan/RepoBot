"""Code-aware tokenization, BM25, and RRF fusion.

The fusion tests use hand-built rankings so the arithmetic is checked exactly,
independently of what either retriever happens to return today.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src import config
from src.index import BM25Index, SearchResult, chunk_id, index_repo, split_identifier, tokenize
from src.ingest import Chunk, ingest_repo
from src.retrieve import (
    MissingIndexError,
    bm25_search,
    hybrid_search,
    load_bm25,
    reciprocal_rank_fusion,
    vector_search,
)

SAMPLE = '''\
"""Auth helpers."""


def parse_jwt(token):
    """Decode a JSON web token and return its claims."""
    return {}


def parseHTTPResponse(raw):
    """Turn a raw response into a status and a body."""
    return 200, raw


class RetryingHttpClient:
    """Issues requests and retries with exponential backoff on failure."""

    def fetch(self, url):
        return url
'''


@pytest.fixture(scope="module")
def repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("retrieve_repo")
    (root / "auth.py").write_text(SAMPLE, encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def indexed(repo: Path, tmp_path_factory: pytest.TempPathFactory):
    """Index the sample repo into throwaway directories; yields the kwargs."""
    base = tmp_path_factory.mktemp("indexes")
    dirs = {"persist_dir": str(base / "chroma"), "bm25_dir": base / "bm25"}
    index_repo(repo, persist_dir=dirs["persist_dir"], cache_dir=base / "cache",
               bm25_dir=dirs["bm25_dir"])
    return dirs


def _chunk(symbol: str, start: int = 1) -> Chunk:
    """A minimal chunk, for fusion tests that never touch a real index."""
    return Chunk(
        code=f"def {symbol}(): pass",
        language="Python",
        file_path=f"{symbol}.py",
        symbol=symbol,
        kind="function",
        start_line=start,
        end_line=start,
    )


def _ranking(*symbols: str) -> list[SearchResult]:
    """Turn symbol names into a ranked list, best first."""
    return [
        SearchResult(chunk=_chunk(s), score=0.0, rank=rank)
        for rank, s in enumerate(symbols, start=1)
    ]


# --------------------------------------------------------------------------
# Tokenization
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("identifier", "expected"),
    [
        ("parse_jwt", ["parse", "jwt"]),
        ("parseJwt", ["parse", "jwt"]),
        ("ParseJWT", ["parse", "jwt"]),
        ("parse", ["parse"]),
        ("PARSE", ["parse"]),
        ("parseHTTPResponse", ["parse", "http", "response"]),
        ("HTTPServer", ["http", "server"]),
        ("__init__", ["init"]),
        ("sha256", ["sha256"]),
        ("MAX_CHUNK_LINES", ["max", "chunk", "lines"]),
        ("", []),
    ],
)
def test_split_identifier(identifier: str, expected: list[str]) -> None:
    assert split_identifier(identifier) == expected


def test_snake_and_camel_produce_the_same_parts() -> None:
    """The whole point: naming convention must not change what matches."""
    assert set(tokenize("parse_jwt")) == set(tokenize("parseJwt"))


def test_multi_part_identifiers_also_emit_a_joined_term() -> None:
    """So an exact hit outranks a chunk that merely mentions both parts."""
    assert tokenize("parse_jwt") == ["parsejwt", "parse", "jwt"]
    assert "parsejwt" not in tokenize("parse the jwt")


def test_tokenize_drops_punctuation_and_stopwords() -> None:
    tokens = tokenize("how do we handle the retry_count = 3;")
    assert "how" not in tokens and "the" not in tokens
    assert "retrycount" in tokens
    assert {"retry", "count", "3", "handle"} <= set(tokens)


def test_tokenizing_a_question_reaches_code_identifiers() -> None:
    """A natural-language query must share terms with camelCase source."""
    query = set(tokenize("where do we parse the JWT token"))
    code = set(tokenize("def parseJwtToken(raw): ..."))
    assert {"parse", "jwt", "token"} <= query & code


# --------------------------------------------------------------------------
# BM25
# --------------------------------------------------------------------------


def test_bm25_finds_a_snake_case_symbol_from_a_spaced_query(repo: Path) -> None:
    index = BM25Index.build(ingest_repo(repo))
    top = index.search("parse jwt", k=1)[0]
    assert top.chunk.symbol == "parse_jwt"


def test_bm25_finds_a_camel_case_symbol_from_a_spaced_query(repo: Path) -> None:
    index = BM25Index.build(ingest_repo(repo))
    top = index.search("parse http response", k=1)[0]
    assert top.chunk.symbol == "parseHTTPResponse"


def test_bm25_drops_zero_scoring_chunks(repo: Path) -> None:
    """Non-matches must not pad the ranking; RRF would treat them as evidence."""
    results = BM25Index.build(ingest_repo(repo)).search("kubernetes helm chart", k=10)
    assert results == []


def test_bm25_ranks_and_scores_descend(repo: Path) -> None:
    results = BM25Index.build(ingest_repo(repo)).search("retry http request", k=5)
    assert [r.rank for r in results] == list(range(1, len(results) + 1))
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)
    assert all(r.contributions == {"bm25": r.rank} for r in results)


def test_bm25_round_trips_through_disk(repo: Path, tmp_path: Path) -> None:
    built = BM25Index.build(ingest_repo(repo))
    built.save(tmp_path / "i.pkl")

    loaded = BM25Index.load(tmp_path / "i.pkl")
    assert loaded is not None
    assert len(loaded) == len(built)
    assert [r.chunk.symbol for r in loaded.search("parse jwt", 3)] == [
        r.chunk.symbol for r in built.search("parse jwt", 3)
    ]


def test_missing_and_corrupt_bm25_indexes_return_none(tmp_path: Path) -> None:
    assert BM25Index.load(tmp_path / "nope.pkl") is None
    corrupt = tmp_path / "bad.pkl"
    corrupt.write_bytes(b"not a pickle")
    assert BM25Index.load(corrupt) is None


def test_empty_bm25_index_is_not_an_error() -> None:
    assert BM25Index.build([]).search("anything") == []


def test_loading_a_missing_index_explains_itself(tmp_path: Path) -> None:
    with pytest.raises(MissingIndexError, match="index_repo"):
        load_bm25(tmp_path, bm25_dir=tmp_path / "absent")


def test_a_loaded_index_is_cached_until_the_file_changes(
    repo: Path, tmp_path: Path
) -> None:
    """The server answers many questions per repo; unpickling and rebuilding
    BM25Okapi per request is pure overhead. Re-indexing (new mtime/size) must
    still invalidate, or a re-indexed repo would serve stale results forever."""
    import time

    BM25Index.build(ingest_repo(repo)).save(tmp_path / "bm25" / f"{_bm25_name(repo)}.pkl")

    first = load_bm25(repo, bm25_dir=tmp_path / "bm25")
    assert load_bm25(repo, bm25_dir=tmp_path / "bm25") is first, "second load is the cache"

    time.sleep(0.01)  # ensure a distinct mtime even on coarse filesystems
    BM25Index.build(ingest_repo(repo)).save(tmp_path / "bm25" / f"{_bm25_name(repo)}.pkl")
    assert load_bm25(repo, bm25_dir=tmp_path / "bm25") is not first, "rebuild must invalidate"


def _bm25_name(repo: Path) -> str:
    from src.index import collection_name

    return collection_name(repo)


# --------------------------------------------------------------------------
# RRF fusion
# --------------------------------------------------------------------------


def test_rrf_arithmetic_is_exact() -> None:
    """A chunk at vector #1 and bm25 #2 scores 1/61 + 1/62."""
    fused = reciprocal_rank_fusion(
        {"vector": _ranking("a", "b"), "bm25": _ranking("b", "a")},
        k=60,
    )
    by_symbol = {r.chunk.symbol: r for r in fused}
    assert by_symbol["a"].score == pytest.approx(1 / 61 + 1 / 62)
    assert by_symbol["b"].score == pytest.approx(1 / 61 + 1 / 62)


def test_agreement_beats_depth() -> None:
    """The behaviour hybrid retrieval exists for.

    `agreed` is only 3rd in both lists; `deep` is 1st in one and absent from the
    other. Two mid-table votes outweigh one top vote.
    """
    fused = reciprocal_rank_fusion(
        {
            "vector": _ranking("deep", "x", "agreed"),
            "bm25": _ranking("y", "z", "agreed"),
        },
        k=60,
    )
    assert fused[0].chunk.symbol == "agreed"
    assert fused[0].score == pytest.approx(2 / 63)
    assert fused[1].score == pytest.approx(1 / 61)


def test_rrf_records_where_each_result_came_from() -> None:
    fused = reciprocal_rank_fusion(
        {"vector": _ranking("a", "b"), "bm25": _ranking("b")},
        k=60,
    )
    by_symbol = {r.chunk.symbol: r for r in fused}
    assert by_symbol["b"].contributions == {"vector": 2, "bm25": 1}
    assert by_symbol["a"].contributions == {"vector": 1}
    assert by_symbol["b"].explain() == "bm25 #1 + vector #2"


def test_rrf_ranks_are_contiguous_and_scores_descend() -> None:
    fused = reciprocal_rank_fusion(
        {"vector": _ranking("a", "b", "c"), "bm25": _ranking("c", "d")},
        k=60,
    )
    assert [r.rank for r in fused] == list(range(1, len(fused) + 1))
    assert [r.score for r in fused] == sorted((r.score for r in fused), reverse=True)


def test_rrf_deduplicates_across_retrievers() -> None:
    fused = reciprocal_rank_fusion(
        {"vector": _ranking("a", "b"), "bm25": _ranking("a", "b")},
        k=60,
    )
    assert len(fused) == 2
    assert {chunk_id(r.chunk) for r in fused} == {chunk_id(_chunk("a")), chunk_id(_chunk("b"))}


def test_rrf_is_deterministic_on_ties() -> None:
    """Every chunk here scores identically; the order must still be stable."""
    args = {"vector": _ranking("b", "a"), "bm25": _ranking("a", "b")}
    assert [r.chunk.symbol for r in reciprocal_rank_fusion(args, k=60)] == [
        r.chunk.symbol for r in reciprocal_rank_fusion(args, k=60)
    ]


def test_rrf_respects_top_k() -> None:
    fused = reciprocal_rank_fusion({"vector": _ranking("a", "b", "c", "d")}, top_k=2)
    assert len(fused) == 2


def test_rrf_of_nothing_is_nothing() -> None:
    assert reciprocal_rank_fusion({"vector": [], "bm25": []}) == []


def test_larger_k_flattens_the_curve() -> None:
    """RRF_K controls how much a top rank outweighs a lower one."""
    small = reciprocal_rank_fusion({"v": _ranking("a", "b")}, k=1)
    large = reciprocal_rank_fusion({"v": _ranking("a", "b")}, k=1000)
    assert small[0].score / small[1].score > large[0].score / large[1].score


# --------------------------------------------------------------------------
# The three retrievers together
# --------------------------------------------------------------------------


def test_all_three_retrievers_are_callable_and_agree_on_shape(
    repo: Path, indexed: dict
) -> None:
    """The ablation needs each one independently, returning the same type."""
    vector = vector_search(repo, "decode a web token", 5, persist_dir=indexed["persist_dir"])
    bm25 = bm25_search(repo, "parse jwt", 5, bm25_dir=indexed["bm25_dir"])
    # The identifier in the query gives BM25 full weight, so both contribute.
    hybrid = hybrid_search(repo, "how does parse_jwt decode a web token", 5, **indexed)

    for results in (vector, bm25, hybrid):
        assert results, "every retriever should find something"
        assert all(isinstance(r, SearchResult) for r in results)
        assert [r.rank for r in results] == list(range(1, len(results) + 1))

    assert all(set(r.contributions) == {"vector"} for r in vector)
    assert all(set(r.contributions) == {"bm25"} for r in bm25)
    assert any(len(r.contributions) == 2 for r in hybrid), "fusion should agree somewhere"


def test_hybrid_returns_at_most_k(repo: Path, indexed: dict) -> None:
    assert len(hybrid_search(repo, "http retry", 2, **indexed)) == 2


# --------------------------------------------------------------------------
# Query-aware fusion weight
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "where is parse_jwt defined",          # snake_case
        "what does parseHTTPResponse return",  # camelCase
        "explain `RetryingHttpClient`",        # backticks
        "what happens in requests.get()",      # dotted call
        "where is __init__ overridden",        # dunder
    ],
)
def test_identifier_bearing_questions_are_detected(question: str) -> None:
    from src.retrieve.retrievers import mentions_identifier

    assert mentions_identifier(question)


@pytest.mark.parametrize(
    "question",
    [
        "how are redirects resolved",
        "what happens when a request times out",
        "where is the retry logic",
    ],
)
def test_natural_language_questions_are_not(question: str) -> None:
    from src.retrieve.retrievers import mentions_identifier

    assert not mentions_identifier(question)


def test_a_semantic_query_lets_bm25_sit_out(repo: Path, indexed: dict) -> None:
    """Measured: on natural-language queries BM25 matches question words against
    docstrings and drags fused MRR below pure vector. At weight 0 hybrid must
    be exactly the vector ranking, not vector plus a tail of BM25 leftovers."""
    hybrid = hybrid_search(repo, "decode a web token", 5, **indexed)
    vector = vector_search(repo, "decode a web token", 5, persist_dir=indexed["persist_dir"])

    assert [chunk_id(r.chunk) for r in hybrid] == [chunk_id(r.chunk) for r in vector]
    assert all(set(r.contributions) == {"vector"} for r in hybrid)


def test_a_lexical_query_gets_bm25_at_full_weight(repo: Path, indexed: dict) -> None:
    hybrid = hybrid_search(repo, "where is parse_jwt defined", 5, **indexed)
    assert any("bm25" in r.contributions for r in hybrid)


# --------------------------------------------------------------------------
# Cross-encoder reranking (model mocked — an 80MB download has no place in CI)
# --------------------------------------------------------------------------


def test_rerank_reorders_by_cross_encoder_score(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.retrieve.rerank as rerank_module

    class Scripted:
        def predict(self, pairs):
            # Score by how late the chunk's symbol is in the alphabet, so the
            # incoming order is provably discarded.
            return [ord(p[1].split()[-2][0]) for p in pairs]

    monkeypatch.setattr(rerank_module, "load_reranker", lambda: Scripted())
    results = rerank_module.rerank("q", _ranking("alpha", "beta", "gamma"), top_k=2)

    assert [r.chunk.symbol for r in results] == ["gamma", "beta"]
    assert [r.rank for r in results] == [1, 2]


def test_rerank_of_nothing_is_nothing() -> None:
    from src.retrieve.rerank import rerank

    assert rerank("q", [], top_k=5) == []


def test_the_reranked_retriever_is_separately_callable() -> None:
    """Registered like the others so the ablation can measure it — measurement
    is exactly how it was found to lose to plain hybrid and kept off the
    default path (see RERANKER_MODEL in config)."""
    from src.retrieve import RETRIEVERS, hybrid_rerank_search

    assert RETRIEVERS["hybrid_rerank"] is hybrid_rerank_search
    assert RETRIEVERS["hybrid"] is hybrid_search, "the default stays un-reranked"


def test_fusion_weights_scale_votes() -> None:
    """Two retrievers disagree; the weighted one wins the fused ranking."""
    a, b = _ranking("alpha", "beta"), _ranking("beta", "alpha")
    fused = reciprocal_rank_fusion({"a": a, "b": b}, k=1, top_k=2, weights={"b": 3.0})
    assert fused[0].chunk.symbol == "beta"


def test_a_zero_weight_retriever_contributes_nothing() -> None:
    """Its chunks must not pad the tail of the fused list with zero scores."""
    a, b = _ranking("alpha"), _ranking("alpha", "beta", "gamma")
    fused = reciprocal_rank_fusion({"a": a, "b": b}, k=1, top_k=5, weights={"b": 0.0})
    assert [r.chunk.symbol for r in fused] == ["alpha"]
    assert all("b" not in r.contributions for r in fused)


def test_hybrid_covers_a_chunk_either_retriever_alone_would_rank_lower(
    repo: Path, indexed: dict
) -> None:
    """Fusion may only promote what a retriever surfaced — never invent a chunk."""
    vector = {chunk_id(r.chunk) for r in vector_search(repo, "parse jwt", config.VECTOR_TOP_K,
                                                       persist_dir=indexed["persist_dir"])}
    bm25 = {chunk_id(r.chunk) for r in bm25_search(repo, "parse jwt", config.BM25_TOP_K,
                                                   bm25_dir=indexed["bm25_dir"])}
    hybrid = {chunk_id(r.chunk) for r in hybrid_search(repo, "parse jwt", 20, **indexed)}
    assert hybrid <= vector | bm25


def test_indexing_builds_both_indexes(repo: Path, tmp_path: Path) -> None:
    stats = index_repo(
        repo,
        persist_dir=str(tmp_path / "chroma"),
        cache_dir=tmp_path / "cache",
        bm25_dir=tmp_path / "bm25",
    )
    assert stats.chunks > 0
    assert stats.keywords == stats.chunks, "both indexes must cover the same chunks"
