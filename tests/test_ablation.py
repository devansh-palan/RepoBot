"""The ablation runner: judging, aggregation, caching, and report shape.

No real model. The judge and the answering configs are scripted, so what is
under test is the scoring logic — which is the part that would quietly produce a
wrong headline number rather than an error.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agent import LLMResponse
from src.eval.ablation import (
    CONFIGS,
    ConfigResult,
    QuestionOutcome,
    breakdown_table,
    failure_report,
    judge_by_citation,
    judge_by_llm,
    markdown_table,
    run_config,
)
from src.eval.gold import GoldAnswer, GoldQuestion, GoldSet, RepoRef, load_gold


def _question(qid="q1", kind="factual", style="semantic", answers=None) -> GoldQuestion:
    return GoldQuestion(
        id=qid, repo="requests", split="dev", kind=kind, style=style,
        question="Where is the default User-Agent built?",
        answers=tuple(answers or [GoldAnswer("src/requests/utils.py", "default_user_agent")]),
    )


class JudgeProvider:
    name = model = "judge"

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.calls = 0

    def complete(self, system, user, model=None, max_tokens=0, on_text=None) -> LLMResponse:
        self.calls += 1
        text = self.replies.pop(0) if self.replies else "VERDICT: partial\nREASON: some"
        return LLMResponse(text=text, model=self.model, input_tokens=3, output_tokens=2)


# --------------------------------------------------------------------------
# Citation judging (factual)
# --------------------------------------------------------------------------


def test_citing_the_right_file_is_correct() -> None:
    answer = "It is built in `src/requests/utils.py:930-948`."
    assert judge_by_citation(answer, _question()) == (1.0, "correct")


def test_citing_a_different_window_of_the_same_function_still_counts() -> None:
    """The chunk boundary is ours, not the answer's; it should not decide grading."""
    answer = "See `src/requests/utils.py:931-940`."
    assert judge_by_citation(answer, _question())[1] == "correct"


def test_naming_the_symbol_without_citing_is_partial() -> None:
    """Right answer, unverifiable form — worth more than nothing, less than full."""
    answer = "The default_user_agent function builds it."
    assert judge_by_citation(answer, _question()) == (0.5, "partial")


def test_citing_the_wrong_file_is_incorrect() -> None:
    answer = "It is built in `src/requests/sessions.py:1-10`."
    assert judge_by_citation(answer, _question()) == (0.0, "incorrect")


def test_an_empty_answer_is_incorrect() -> None:
    assert judge_by_citation("", _question()) == (0.0, "incorrect")


# --------------------------------------------------------------------------
# LLM judging (explanatory)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("VERDICT: correct\nREASON: matches", 1.0),
        ("VERDICT: partial\nREASON: half", 0.5),
        ("VERDICT: incorrect\nREASON: wrong", 0.0),
        ("verdict: CORRECT\nreason: fine", 1.0),
    ],
)
def test_judge_verdicts_map_to_scores(reply: str, expected: float, tmp_path: Path) -> None:
    gold = load_gold()
    question = next(q for q in gold.questions if q.kind == "explanatory")
    score, _ = judge_by_llm("an answer", question, gold, JudgeProvider(reply))
    assert score == expected


def test_a_malformed_verdict_defaults_to_partial() -> None:
    """Neither full credit nor zero for a judge that failed to answer the format."""
    gold = load_gold()
    question = next(q for q in gold.questions if q.kind == "explanatory")
    score, verdict = judge_by_llm("x", question, gold, JudgeProvider("I think it's fine!"))
    assert (score, verdict) == (0.5, "partial")


def test_the_judge_is_shown_the_gold_code_not_the_question_alone() -> None:
    from src.eval.ablation import gold_code

    gold = load_gold()
    question = next(q for q in gold.questions if q.kind == "explanatory")
    reference = gold_code(question, gold)
    assert reference, "the judge needs code to grade against"
    for answer in question.answers:
        assert answer.symbol in reference


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def _outcome(**over) -> QuestionOutcome:
    base = dict(
        question_id="q", kind="factual", style="semantic", repo="requests",
        recall=1.0, reciprocal_rank=1.0, accuracy=1.0,
    )
    return QuestionOutcome(**{**base, **over})


def test_metrics_can_be_sliced_by_kind_and_style() -> None:
    result = ConfigResult(name="rag", outcomes=[
        _outcome(kind="factual", recall=1.0, accuracy=1.0),
        _outcome(kind="explanatory", recall=0.0, accuracy=0.0),
        _outcome(kind="explanatory", recall=0.5, accuracy=0.5),
    ])
    assert result.recall() == pytest.approx(0.5)
    assert result.recall(kind="factual") == 1.0
    assert result.recall(kind="explanatory") == pytest.approx(0.25)
    assert result.accuracy(kind="explanatory") == pytest.approx(0.25)


def test_a_retrieval_only_config_reports_no_accuracy() -> None:
    """It produces no answer, so a 0.0 would be a lie rather than a score."""
    result = ConfigResult(name="vector", outcomes=[_outcome(accuracy=None)])
    assert result.accuracy() is None
    assert not result.answers


def test_accuracy_ignores_unanswered_questions_rather_than_scoring_them_zero() -> None:
    result = ConfigResult(name="rag", outcomes=[_outcome(accuracy=1.0), _outcome(accuracy=None)])
    assert result.accuracy() == 1.0


# --------------------------------------------------------------------------
# Report rendering
# --------------------------------------------------------------------------


def test_the_table_marks_missing_accuracy_as_not_applicable() -> None:
    results = {"vector": ConfigResult("vector", [_outcome(accuracy=None)])}
    table = markdown_table(results, [_question()])
    assert "| vector-only |" in table
    assert "n/a" in table, "must not render a missing score as 0.00"
    assert "—" not in table, "ascii only; the report is printed to a Windows console"


def test_the_breakdown_covers_both_kinds_and_both_styles() -> None:
    results = {"rag": ConfigResult("rag", [
        _outcome(kind="factual", style="lexical"),
        _outcome(kind="explanatory", style="semantic"),
    ])}
    table = breakdown_table(results, load_gold())
    for label in ("factual", "explanatory", "semantic", "lexical"):
        assert label in table


def test_the_failure_report_lists_only_imperfect_answers() -> None:
    results = {"rag": ConfigResult("rag", [
        _outcome(question_id="good", accuracy=1.0),
        _outcome(question_id="bad", accuracy=0.0, verdict="incorrect", recall=0.0),
        _outcome(question_id="half", accuracy=0.5, verdict="partial"),
    ])}
    report = failure_report(results)
    assert "bad" in report and "half" in report
    assert "| good |" not in report


def test_the_failure_report_picks_the_best_answering_config() -> None:
    results = {
        "vector": ConfigResult("vector", [_outcome(accuracy=None)]),
        "rag": ConfigResult("rag", [_outcome(question_id="r", accuracy=0.0, verdict="incorrect")]),
        "agent": ConfigResult("agent", [_outcome(question_id="a", accuracy=0.5, verdict="partial")]),
    }
    report = failure_report(results)
    assert "agent + reflection" in report, "report failures of the config that scored best"


def test_the_failure_report_is_honest_when_nothing_answered() -> None:
    assert "no answering configs" in failure_report(
        {"vector": ConfigResult("vector", [_outcome(accuracy=None)])}
    )


# --------------------------------------------------------------------------
# Caching
# --------------------------------------------------------------------------


def test_a_cached_run_is_not_repeated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A full ablation is hours of local generation; a crash must not restart it."""
    gold = load_gold()
    question = next(q for q in gold.where(repo="requests") if q.kind == "factual")
    provider = JudgeProvider()

    calls = {"n": 0}

    def fake_answer(*args, **kwargs):
        calls["n"] += 1
        from src.agent import Answer

        return Answer(question=question.question, text="see `x.py:1-2`", results=[])

    monkeypatch.setattr("src.eval.ablation.answer_question", fake_answer)

    first = run_config("rag", gold, [question], provider, cache=tmp_path)
    second = run_config("rag", gold, [question], provider, cache=tmp_path)

    assert calls["n"] == 1, "the second run should be served from cache"
    assert first.outcomes[0].accuracy == second.outcomes[0].accuracy


def test_the_cache_is_keyed_by_model_so_a_swap_does_not_reuse_scores(tmp_path: Path) -> None:
    from src.eval.ablation import _cache_file

    class P:
        name = "p"
        model = "qwen2.5-coder:7b"

    class Q:
        name = "p"
        model = "qwen2.5-coder:3b"

    a = _cache_file(tmp_path, "rag", _question(), P())
    b = _cache_file(tmp_path, "rag", _question(), Q())
    assert a != b, "scores from different models must not be mixed"


def test_a_corrupt_cache_entry_is_ignored_rather_than_fatal(tmp_path: Path) -> None:
    from src.eval.ablation import _cache_file, _load

    class P:
        name = model = "p"

    path = _cache_file(tmp_path, "rag", _question(), P())
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    assert _load(tmp_path, "rag", _question(), P()) is None


def test_the_cache_is_keyed_by_the_settings_that_shape_outcomes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retuning retrieval or editing a prompt and re-running the ablation must
    not silently report the old system's cached scores as the new system's."""
    from src import config as cfg
    from src.eval.ablation import _cache_file

    class P:
        name = model = "p"

    before = _cache_file(tmp_path, "rag", _question(), P())
    monkeypatch.setattr(cfg, "RRF_K", cfg.RRF_K + 1)
    after = _cache_file(tmp_path, "rag", _question(), P())
    assert before != after


# --------------------------------------------------------------------------
# Configuration ladder
# --------------------------------------------------------------------------


def test_the_four_configs_isolate_one_change_each() -> None:
    """Each row must differ from the one before it in exactly one dimension."""
    ladder = list(CONFIGS.values())
    for before, after in zip(ladder, ladder[1:]):
        differences = sum(1 for a, b in zip(before, after) if a != b)
        assert differences == 1, f"{before} -> {after} changes {differences} things"


def test_retrieval_only_configs_do_not_answer() -> None:
    assert CONFIGS["vector"][1] is False
    assert CONFIGS["hybrid"][1] is False
    assert CONFIGS["rag"][1] is True and CONFIGS["rag"][2] is False
    assert CONFIGS["agent"][1] is True and CONFIGS["agent"][2] is True


def test_an_empty_slice_reports_not_measured_rather_than_zero() -> None:
    """A run that sampled no lexical questions once reported lexical R@5 = 0.000
    across every config, which reads as total failure instead of "not run"."""
    result = ConfigResult(name="rag", outcomes=[_outcome(style="semantic")])
    assert result.recall(style="semantic") == 1.0
    assert result.recall(style="lexical") is None
    assert result.mrr(style="lexical") is None

    table = breakdown_table({"rag": result}, load_gold())
    assert "n/a" in table


def test_the_balanced_sampler_covers_both_styles() -> None:
    """Sampling only the head of the pool draws from dev, which is all semantic."""
    import argparse

    from src.eval.__main__ import _select

    gold = load_gold()
    args = argparse.Namespace(split="all", limit=16, balanced=True)
    chosen = _select(gold, args)

    assert {q.kind for q in chosen} == {"factual", "explanatory"}
    assert {q.style for q in chosen} == {"semantic", "lexical"}
