"""Retrieval metrics and the gold set.

Metric tests use hand-built rankings so the arithmetic is pinned exactly,
independently of what any retriever returns today. The gold-set tests check the
labels are real, which is the failure mode that would quietly corrupt every
number the eval produces.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval import (
    GoldAnswer,
    GoldQuestion,
    hit_rate_at_k,
    load_gold,
    matched_answers,
    matches,
    mean,
    recall_at_k,
    reciprocal_rank,
    summary,
    validate,
)
from src.eval.gold import KINDS, SPLITS, STYLES, GoldSet, RepoRef
from src.index import SearchResult
from src.ingest import Chunk


def _result(file: str, symbol: str, rank: int = 1) -> SearchResult:
    chunk = Chunk(
        code=f"def {symbol}(): pass",
        language="Python",
        file_path=file,
        symbol=symbol,
        kind="function",
        start_line=rank,
        end_line=rank,
    )
    return SearchResult(chunk=chunk, score=1.0 / rank, rank=rank)


def _ranking(*pairs: tuple[str, str]) -> list[SearchResult]:
    return [_result(f, s, rank) for rank, (f, s) in enumerate(pairs, start=1)]


A = GoldAnswer("a.py", "alpha")
B = GoldAnswer("b.py", "beta")
C = GoldAnswer("c.py", "gamma")


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------


def test_a_chunk_matches_only_on_both_file_and_symbol() -> None:
    assert matches(_result("a.py", "alpha"), A)
    assert not matches(_result("other.py", "alpha"), A), "same symbol, wrong file"
    assert not matches(_result("a.py", "other"), A), "same file, wrong symbol"


def test_one_answer_matched_by_several_chunks_counts_once() -> None:
    """`@overload` stubs and split symbols produce several chunks per label.

    Counting chunks instead of answers would push recall above 1.0.
    """
    results = _ranking(("a.py", "alpha"), ("a.py", "alpha"), ("a.py", "alpha"))
    assert matched_answers(results, [A]) == {A}
    assert recall_at_k(results, [A], k=3) == 1.0


# --------------------------------------------------------------------------
# recall@k
# --------------------------------------------------------------------------


def test_recall_counts_the_fraction_of_answers_found() -> None:
    results = _ranking(("a.py", "alpha"), ("z.py", "noise"), ("b.py", "beta"))
    assert recall_at_k(results, [A, B, C], k=3) == pytest.approx(2 / 3)


def test_recall_respects_the_cutoff() -> None:
    results = _ranking(("z.py", "noise"), ("a.py", "alpha"))
    assert recall_at_k(results, [A], k=1) == 0.0, "alpha is at rank 2, outside k=1"
    assert recall_at_k(results, [A], k=2) == 1.0


def test_recall_is_zero_when_nothing_relevant_is_retrieved() -> None:
    assert recall_at_k(_ranking(("z.py", "noise")), [A], k=5) == 0.0


def test_recall_of_an_empty_ranking_is_zero() -> None:
    assert recall_at_k([], [A], k=5) == 0.0


def test_recall_with_a_nonpositive_k_is_zero() -> None:
    assert recall_at_k(_ranking(("a.py", "alpha")), [A], k=0) == 0.0


def test_recall_without_gold_answers_raises() -> None:
    """Silently scoring 0 or 1 here would skew the average without a trace."""
    with pytest.raises(ValueError, match="no gold answers"):
        recall_at_k(_ranking(("a.py", "alpha")), [], k=5)


# --------------------------------------------------------------------------
# hit rate
# --------------------------------------------------------------------------


def test_hit_rate_diverges_from_recall_on_multi_answer_questions() -> None:
    """One of three found is recall 0.33 but hit rate 1.0 — both are worth knowing."""
    results = _ranking(("a.py", "alpha"))
    assert recall_at_k(results, [A, B, C], k=8) == pytest.approx(1 / 3)
    assert hit_rate_at_k(results, [A, B, C], k=8) == 1.0


def test_hit_rate_is_zero_when_nothing_matches() -> None:
    assert hit_rate_at_k(_ranking(("z.py", "noise")), [A], k=8) == 0.0


# --------------------------------------------------------------------------
# MRR
# --------------------------------------------------------------------------


def test_reciprocal_rank_uses_the_first_relevant_position() -> None:
    assert reciprocal_rank(_ranking(("a.py", "alpha")), [A]) == 1.0
    assert reciprocal_rank(_ranking(("z.py", "n"), ("a.py", "alpha")), [A]) == 0.5
    assert reciprocal_rank(
        _ranking(("z.py", "n"), ("y.py", "n"), ("a.py", "alpha")), [A]
    ) == pytest.approx(1 / 3)


def test_reciprocal_rank_takes_the_earliest_of_several_answers() -> None:
    results = _ranking(("z.py", "noise"), ("b.py", "beta"), ("a.py", "alpha"))
    assert reciprocal_rank(results, [A, B]) == 0.5, "beta at rank 2 is the first hit"


def test_reciprocal_rank_is_zero_when_nothing_is_relevant() -> None:
    assert reciprocal_rank(_ranking(("z.py", "noise")), [A]) == 0.0
    assert reciprocal_rank([], [A]) == 0.0


def test_reciprocal_rank_uses_list_position_not_the_stored_rank() -> None:
    """A caller that slices or re-ranks must get the arithmetic it expects."""
    sliced = _ranking(("z.py", "n"), ("a.py", "alpha"))[1:]
    assert sliced[0].rank == 2, "the stored rank is still 2"
    assert reciprocal_rank(sliced, [A]) == 1.0, "but it is first in the list given"


def test_mean_of_no_questions_is_zero_not_an_error() -> None:
    assert mean([]) == 0.0
    assert mean([1.0, 0.0, 0.5]) == pytest.approx(0.5)


# --------------------------------------------------------------------------
# The gold set itself
# --------------------------------------------------------------------------


def test_gold_set_shape() -> None:
    gold = load_gold()
    assert len(gold) == 80
    assert {q.kind for q in gold.questions} == KINDS
    assert {q.style for q in gold.questions} == STYLES
    assert {q.split for q in gold.questions} == SPLITS
    assert set(gold.repos) == {"requests", "click"}


def test_dev_and_test_splits_are_sized_for_a_real_holdout() -> None:
    """Test must be big enough, and must not be a copy of dev's shape."""
    gold = load_gold()
    assert len(gold.where(split="dev")) == 30
    assert len(gold.where(split="test")) == 50


def test_the_holdout_covers_what_dev_cannot() -> None:
    """A holdout that shares dev's blind spots cannot detect overfitting.

    dev is one repo and entirely natural-language; test adds a second repo and
    questions that name identifiers, which is where BM25 should earn its place.
    """
    gold = load_gold()
    assert gold.where(split="dev", style="lexical") == [], "dev is all semantic by construction"
    assert len(gold.where(split="test", style="lexical")) >= 20
    assert {q.repo for q in gold.where(split="dev")} == {"requests"}
    assert {q.repo for q in gold.where(split="test")} == {"requests", "click"}


def test_gold_ids_are_unique_and_every_question_has_an_answer() -> None:
    gold = load_gold()
    assert len({q.id for q in gold.questions}) == len(gold)
    assert all(q.answers for q in gold.questions)
    assert all(q.question.strip().endswith("?") for q in gold.questions)


def test_explanatory_questions_span_more_chunks_than_factual_ones() -> None:
    """If they did not, the two kinds would not be measuring different things."""
    gold = load_gold()
    factual = mean(len(q.answers) for q in gold.where(kind="factual"))
    explanatory = mean(len(q.answers) for q in gold.where(kind="explanatory"))
    assert factual < 1.2
    assert explanatory > 2.0


def test_lexical_questions_actually_name_an_identifier() -> None:
    """Otherwise the style label is a lie and the slice measures nothing."""
    gold = load_gold()
    for q in gold.where(style="lexical"):
        assert "`" in q.question, f"{q.id} is labelled lexical but names no identifier"


def test_every_gold_label_points_at_a_real_chunk() -> None:
    """The failure that would silently make every retriever look worse."""
    if not Path("data/repos/requests").exists() or not Path("data/repos/click").exists():
        pytest.skip("eval checkouts not present")
    assert validate(load_gold()) == []


def test_summary_reports_the_splits() -> None:
    text = summary(load_gold())
    assert "80 questions" in text
    assert "dev" in text and "test" in text


# --------------------------------------------------------------------------
# Loader validation
# --------------------------------------------------------------------------


def _write(tmp_path: Path, questions: list[dict]) -> Path:
    path = tmp_path / "gold.json"
    path.write_text(
        json.dumps(
            {"repos": {"r": {"path": "data/repos/requests", "commit": "c"}},
             "questions": questions},
        ),
        encoding="utf-8",
    )
    return path


def _q(**over) -> dict:
    base = {"id": "x", "repo": "r", "split": "dev", "kind": "factual",
            "style": "semantic", "question": "?", "answers": [{"file": "a", "symbol": "b"}]}
    return {**base, **over}


def test_loader_rejects_duplicate_ids(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate question id"):
        load_gold(_write(tmp_path, [_q(), _q()]))


@pytest.mark.parametrize(
    ("field", "value"),
    [("kind", "trivia"), ("style", "cryptic"), ("split", "holdout")],
)
def test_loader_rejects_unknown_enum_values(tmp_path: Path, field: str, value: str) -> None:
    with pytest.raises(ValueError, match=value):
        load_gold(_write(tmp_path, [_q(**{field: value})]))


def test_loader_rejects_an_unknown_repo(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown repo"):
        load_gold(_write(tmp_path, [_q(repo="nope")]))


def test_loader_rejects_a_question_with_no_answers(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one answer"):
        load_gold(_write(tmp_path, [_q(answers=[])]))


def test_validate_reports_a_bad_label_with_a_hint() -> None:
    """A wrong label must be loud, and the message should help fix it."""
    if not Path("data/repos/requests").exists():
        pytest.skip("requests checkout not present")

    real = load_gold()
    bogus = GoldQuestion(
        id="bogus", repo="requests", split="test", kind="factual", style="semantic",
        question="?", answers=(GoldAnswer("src/requests/utils.py", "no_such_symbol"),),
    )
    gold = GoldSet(repos={"requests": real.repos["requests"]}, questions=(bogus,))

    problems = validate(gold)
    assert len(problems) == 1
    assert "bogus" in problems[0] and "no_such_symbol" in problems[0]
    assert "symbols in that file" in problems[0], "should suggest what is really there"


def test_validate_reports_a_missing_checkout() -> None:
    gold = GoldSet(repos={"ghost": RepoRef("ghost", "data/repos/ghost", "abc")}, questions=())
    assert validate(gold) == ["ghost: no checkout at data/repos/ghost"]
