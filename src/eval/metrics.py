"""Retrieval metrics: recall@k and MRR.

Both answer different questions and the ablation needs both. Recall@k asks "how
much of the answer did we put in front of the model", which is what caps how
good a grounded answer can possibly be. MRR asks "how near the top was the first
useful chunk", which is what matters when the context budget is tight and the
tail gets truncated. A retriever can win on one and lose on the other.

Gold answers are labelled by (file, symbol) rather than by line range, because
line numbers shift on every edit to the file above them and would make the gold
set stale after a single upstream commit.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from src.index import SearchResult

from .gold import GoldAnswer


def matches(result: SearchResult, answer: GoldAnswer) -> bool:
    """Whether a retrieved chunk is the code a gold answer points at."""
    return result.chunk.file_path == answer.file and result.chunk.symbol == answer.symbol


def matched_answers(
    results: Sequence[SearchResult],
    answers: Sequence[GoldAnswer],
    k: int | None = None,
) -> set[GoldAnswer]:
    """The distinct gold answers covered by the top-k results.

    Distinct matters: one gold answer can be matched by several chunks — an
    oversized symbol split into parts, or a function with `@overload` stubs
    ahead of it. Counting chunks instead of answers would push recall above 1.0.
    """
    top = results if k is None else results[:k]
    return {answer for answer in answers for result in top if matches(result, answer)}


def recall_at_k(
    results: Sequence[SearchResult],
    answers: Sequence[GoldAnswer],
    k: int,
) -> float:
    """Fraction of a question's gold answers that appear in the top k.

    Undefined with no gold answers, so that case raises rather than silently
    contributing a 0 or a 1 that would skew the average.
    """
    if not answers:
        raise ValueError("recall is undefined for a question with no gold answers")
    if k <= 0:
        return 0.0
    return len(matched_answers(results, answers, k)) / len(set(answers))


def hit_rate_at_k(
    results: Sequence[SearchResult],
    answers: Sequence[GoldAnswer],
    k: int,
) -> float:
    """1.0 if *any* gold answer is in the top k, else 0.0.

    Reported alongside recall because they diverge on multi-answer questions:
    finding one of three relevant chunks is recall 0.33 but hit rate 1.0, and
    for an explanatory question one good chunk is often enough.
    """
    if not answers:
        raise ValueError("hit rate is undefined for a question with no gold answers")
    return 1.0 if matched_answers(results, answers, k) else 0.0


def reciprocal_rank(
    results: Sequence[SearchResult],
    answers: Sequence[GoldAnswer],
) -> float:
    """1 / (rank of the first relevant result), or 0.0 if none is relevant.

    Uses position in the passed list rather than `result.rank`, so a caller that
    slices or re-ranks gets the arithmetic it expects.
    """
    if not answers:
        raise ValueError("reciprocal rank is undefined for a question with no gold answers")
    for position, result in enumerate(results, start=1):
        if any(matches(result, answer) for answer in answers):
            return 1.0 / position
    return 0.0


def mean(values: Iterable[float]) -> float:
    """Average, treating an empty run as 0.0 rather than raising.

    An eval run over zero questions is a configuration mistake, but it should
    surface as a zero score in the report, not a traceback halfway through.
    """
    values = list(values)
    return sum(values) / len(values) if values else 0.0
