"""The agent: tools, the reflection nodes, and the retry edge.

`should_retry` is a pure function of state, so the control flow — the part most
likely to loop forever or terminate early — is tested with plain dicts and no
model at all. The node tests use a scripted provider so a retry can be forced
deterministically rather than hoped for.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from src import config
from src.agent import (
    MAX_ATTEMPTS,
    Critique,
    EchoProvider,
    LLMResponse,
    build_graph,
    detect_language,
    read_file,
    run_agent,
    run_tests,
    search_code,
    should_retry,
)
from src.agent.graph import _citations, _merge, _query_from_citations, reflect
from src.index import SearchResult, index_repo
from src.ingest import Chunk

SAMPLE = '''"""Billing helpers."""


def compound_interest(principal, rate, years):
    """Grow a principal at a fixed annual rate."""
    return principal * (1 + rate) ** years


class RetryingHttpClient:
    """Issues requests and retries with exponential backoff on failure."""

    def fetch(self, url):
        return url
'''


@pytest.fixture(scope="module")
def repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("agent_graph_repo")
    (root / "billing.py").write_text(SAMPLE, encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def indexed(repo: Path, tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Index into throwaway dirs; yields the kwargs that point the agent there."""
    base = tmp_path_factory.mktemp("agent_graph_idx")
    dirs = {"persist_dir": str(base / "chroma"), "bm25_dir": str(base / "bm25")}
    index_repo(repo, cache_dir=base / "cache", **dirs)
    return {"repo_path": repo, **dirs}


def _result(path: str, symbol: str, start: int = 1, end: int = 5, rank: int = 1) -> SearchResult:
    chunk = Chunk(
        code=f"def {symbol}(): pass",
        language="Python",
        file_path=path,
        symbol=symbol,
        kind="function",
        start_line=start,
        end_line=end,
    )
    return SearchResult(chunk=chunk, score=1.0, rank=rank)


class ScriptedProvider:
    """Returns a fixed sequence of replies, so a retry can be forced exactly."""

    name = "scripted"
    model = "scripted-1"

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.prompts: list[tuple[str, str]] = []

    def complete(
        self,
        system: str,
        user: str,
        model: str | None = None,
        max_tokens: int = config.LLM_MAX_TOKENS,
        on_text: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        self.prompts.append((system, user))
        text = self.replies.pop(0) if self.replies else "GROUNDED: yes\nREASON: fine\nQUERY: NONE"
        if on_text is not None:
            on_text(text)
        return LLMResponse(text=text, model=self.model, input_tokens=5, output_tokens=3)


# --------------------------------------------------------------------------
# The conditional edge
# --------------------------------------------------------------------------


def test_a_grounded_answer_ends_the_loop() -> None:
    state = {"critique": Critique(grounded=True), "attempts": 1, "seen": [_result("a.py", "f")]}
    assert should_retry(state) == "end"


def test_an_ungrounded_answer_with_a_followup_query_searches_again() -> None:
    state = {
        "critique": Critique(grounded=False, reason="unsupported", followup_query="parse_jwt"),
        "attempts": 1,
        "seen": [_result("a.py", "f")],
    }
    assert should_retry(state) == "search"


def test_an_ungrounded_answer_with_no_better_query_rewrites_without_searching() -> None:
    """Re-running the original query returns the chunks already in `seen`, so a
    citation-format failure or a query-less critic rejection goes straight back
    to generate rather than paying for a redundant retrieval."""
    state = {
        "critique": Critique(grounded=False, reason="no citations", kind="no_citations"),
        "attempts": 1,
        "seen": [_result("a.py", "f")],
    }
    assert should_retry(state) == "rewrite"


def test_the_attempt_cap_stops_the_loop_even_when_never_grounded() -> None:
    """A critic that rejects everything must not spin forever."""
    state = {
        "critique": Critique(grounded=False, reason="still unsupported"),
        "attempts": MAX_ATTEMPTS,
        "seen": [_result("a.py", "f")],
    }
    assert should_retry(state) == "end"


def test_retrying_is_allowed_right_up_to_the_cap() -> None:
    for attempts in range(1, MAX_ATTEMPTS):
        state = {
            "critique": Critique(grounded=False, followup_query="something new"),
            "attempts": attempts,
            "seen": [1],
        }
        assert should_retry(state) == "search", f"should still retry after {attempts}"


def test_a_repo_with_nothing_retrievable_ends_rather_than_looping() -> None:
    """Retrying a search that returned nothing would return nothing again."""
    state = {"critique": Critique(grounded=False), "attempts": 1, "seen": [], "queries": []}
    assert should_retry(state) == "end"


# --------------------------------------------------------------------------
# Evidence accumulation
# --------------------------------------------------------------------------


def test_merge_keeps_earlier_chunks_and_appends_new_ones() -> None:
    """A bad retry must not throw away a good first attempt."""
    seen = [_result("a.py", "alpha"), _result("b.py", "beta")]
    fresh = [_result("b.py", "beta"), _result("c.py", "gamma")]

    merged = _merge(seen, fresh)
    assert [r.chunk.symbol for r in merged] == ["alpha", "beta", "gamma"]


def test_merge_of_nothing_new_is_a_no_op() -> None:
    seen = [_result("a.py", "alpha")]
    assert len(_merge(seen, list(seen))) == 1


# --------------------------------------------------------------------------
# Reflection
# --------------------------------------------------------------------------


def test_a_fabricated_citation_is_caught_without_asking_the_model() -> None:
    """Set membership, so the model cannot argue its way out of it."""
    provider = ScriptedProvider("GROUNDED: yes\nREASON: looks fine\nQUERY: NONE")
    state = {
        "question": "q",
        "answer": "See `src/invented.py:99-120`.",
        "seen": [_result("src/real.py", "real", 1, 5)],
        "attempts": 0,
    }

    out = reflect(state, provider)
    critique = out["critique"]

    assert not critique.grounded
    assert critique.unsupported == ("src/invented.py:99-120",)
    assert provider.prompts == [], "the mechanical check must short-circuit the critic"


def test_a_fabricated_citation_becomes_a_useful_next_query() -> None:
    """Retrying the original question would return the original chunks."""
    assert _query_from_citations(("src/foo/parser.py:10-20",)) == "parser"
    assert "cookies" in _query_from_citations(("src/requests/cookies.py:1-5",))


def test_the_critic_is_consulted_when_citations_all_check_out() -> None:
    provider = ScriptedProvider("GROUNDED: no\nREASON: the call is never shown\nQUERY: prepare_body")
    state = {
        "question": "q",
        "answer": "Handled in `src/real.py:1-5`.",
        "seen": [_result("src/real.py", "real", 1, 5)],
        "attempts": 0,
    }

    critique = reflect(state, provider)["critique"]
    assert len(provider.prompts) == 1
    assert not critique.grounded
    assert critique.followup_query == "prepare_body"


def test_a_grounded_verdict_carries_no_followup_query() -> None:
    provider = ScriptedProvider("GROUNDED: yes\nREASON: supported\nQUERY: something")
    state = {
        "question": "q",
        "answer": "See `src/real.py:1-5`.",
        "seen": [_result("src/real.py", "real", 1, 5)],
        "attempts": 0,
    }
    critique = reflect(state, provider)["critique"]
    assert critique.grounded
    assert critique.followup_query == "", "a grounded answer has nothing to search for"


def test_a_malformed_critique_is_not_read_as_a_rejection() -> None:
    """A small local model will not always obey the format.

    Treating a formatting slip as 'not grounded' would burn every retry on a
    parsing problem rather than a real one.
    """
    provider = ScriptedProvider("Sure! The answer looks well supported to me.")
    state = {
        "question": "q",
        "answer": "See `src/real.py:1-5`.",
        "seen": [_result("src/real.py", "real", 1, 5)],
        "attempts": 0,
    }
    assert reflect(state, provider)["critique"].grounded


def test_no_evidence_is_ungrounded_without_paying_for_a_critique() -> None:
    provider = ScriptedProvider()
    out = reflect({"question": "q", "answer": "anything", "seen": [], "attempts": 0}, provider)
    assert not out["critique"].grounded
    assert provider.prompts == []


def test_citations_are_extracted_in_order_without_duplicates() -> None:
    text = "See `a/b.py:1-5` and `c.py:10-12`, then `a/b.py:1-5` again."
    assert _citations(text) == ["a/b.py:1-5", "c.py:10-12"]


def test_a_spacey_filename_citation_is_extracted_whole() -> None:
    """`PLI Website Development.md:444-449` — without backtick handling only
    the tail after the last space is captured, which then fails the fabricated
    check on a parsing artifact rather than a lie."""
    text = "See `PLI Website Development.md:444-449` and `src/app.py:1-5`."
    assert _citations(text) == ["PLI Website Development.md:444-449", "src/app.py:1-5"]


def test_a_dropped_directory_prefix_is_not_a_fabricated_citation() -> None:
    """The model cites 'x.js:4-8' for retrieved 'api/x.js:3-9'; same code."""
    from src.agent.graph import _cites_retrieved_chunk

    evidence = [_result("api/x.js", "handler", start=3, end=9)]
    assert _cites_retrieved_chunk("x.js:4-8", evidence)
    assert not _cites_retrieved_chunk("x.js:40-48", evidence), "range must overlap"
    assert not _cites_retrieved_chunk("y.js:4-8", evidence), "file must match"


def test_prose_citations_into_retrieved_code_count_as_grounded() -> None:
    """Seen live on a 7B: 'The `_verify.js` file, line 1-14' points at exactly
    the retrieved chunk, and failing it on format alone rejects an honest
    answer. Basename match + overlapping range is enough."""
    from src.agent.graph import _loose_citations, _matches_evidence

    evidence = [_result("api/_verify.js", "verify", start=1, end=14)]

    for answer in (
        "The `_verify.js` file, line 1-14, deals with verifying signatures.",
        "Verification happens in api/_verify.js, lines 2-10.",
        'See line 5-9 of `_verify.js` for the check.',
    ):
        assert _matches_evidence(_loose_citations(answer), evidence), answer

    for answer in (
        "The `_verify.js` file, line 90-99, deals with verification.",  # range outside
        "The `other.js` file, line 1-14, does it.",                     # wrong file
        "Verification uses HMAC signatures.",                           # no citation at all
    ):
        assert not _matches_evidence(_loose_citations(answer), evidence), answer


def test_naming_a_retrieved_file_without_lines_reaches_the_critic() -> None:
    """Seen live: 'The README.md provides build and deploy information' — a
    checkable claim at file granularity, retrieved chunk in hand. It must go
    to the critic, not mechanically fail on citation format."""
    provider = ScriptedProvider("GROUNDED: yes\nREASON: the README shows this\nQUERY: NONE")
    state = {
        "question": "what is this repo about?",
        "answer": "The README.md provides high-level information about the project. " * 5,
        "seen": [_result("README.md", "Widget Factory", start=1, end=20)],
        "attempts": 0,
    }
    result = reflect(state, provider)
    assert result["critique"].grounded
    assert len(provider.prompts) == 1


def test_naming_only_files_that_were_never_retrieved_still_fails() -> None:
    """The original hole must stay closed: references to nothing in evidence
    give the critic nothing to check, so the mechanical gate fires."""
    provider = ScriptedProvider()
    state = {
        "question": "what is this repo about?",
        "answer": "The setup.cfg and Makefile configure the build process nicely. " * 5,
        "seen": [_result("README.md", "Widget Factory", start=1, end=20)],
        "attempts": 0,
    }
    result = reflect(state, provider)
    assert not result["critique"].grounded
    assert result["critique"].kind == "no_citations"
    assert not provider.prompts, "no critic call for an uncheckable answer"


def test_a_long_answer_with_only_prose_citations_reaches_the_critic() -> None:
    """The mechanical no-citations check must not fire when the prose cites
    retrieved code; the critic then judges substance rather than format."""
    provider = ScriptedProvider("GROUNDED: yes\nREASON: matches the excerpt\nQUERY: NONE")
    state = {
        "question": "what verifies signatures?",
        "answer": "The `_verify.js` file, line 1-14, verifies signatures. " * 6,
        "seen": [_result("api/_verify.js", "verify", start=1, end=14)],
        "attempts": 0,
    }
    result = reflect(state, provider)
    assert result["critique"].grounded
    assert len(provider.prompts) == 1, "the critic should have been consulted"


# --------------------------------------------------------------------------
# The graph end to end
# --------------------------------------------------------------------------


def test_a_grounded_first_answer_runs_exactly_one_attempt(indexed: dict) -> None:
    provider = ScriptedProvider(
        "Interest compounds annually. `billing.py:4-6`",
        "GROUNDED: yes\nREASON: supported by the excerpt\nQUERY: NONE",
    )
    result = run_agent(question="how is interest compounded?", k=4, provider=provider, **indexed)

    assert result.attempts == 1
    assert result.grounded
    assert len(result.queries) == 1
    assert result.results


def test_an_ungrounded_answer_triggers_a_second_retrieval(indexed: dict) -> None:
    """The retry must search something *different*, or it learns nothing."""
    provider = ScriptedProvider(
        "It uses `billing.py:99-120`.",                       # fabricated citation
        "Interest compounds annually. `billing.py:4-6`",       # second answer
        "GROUNDED: yes\nREASON: now supported\nQUERY: NONE",
    )
    result = run_agent(question="how is interest compounded?", k=4, provider=provider, **indexed)

    assert result.attempts == 2
    assert result.grounded
    assert len(result.queries) == 2
    assert result.queries[1] != result.queries[0], "a retry must not repeat the query"


def test_the_loop_stops_at_the_cap_and_says_so(indexed: dict) -> None:
    """A critic that never accepts must still terminate, flagged ungrounded."""
    provider = ScriptedProvider(
        *["Answer. `billing.py:4-6`", "GROUNDED: no\nREASON: nope\nQUERY: retry backoff"]
        * MAX_ATTEMPTS
    )
    result = run_agent(question="how is interest compounded?", k=4, provider=provider, **indexed)

    assert result.attempts == MAX_ATTEMPTS
    assert not result.grounded, "must not claim a checked answer it never got"
    assert len(result.critiques) == MAX_ATTEMPTS


def test_the_agent_accumulates_evidence_across_attempts(indexed: dict) -> None:
    provider = ScriptedProvider(
        "It uses `billing.py:99-120`.",
        "Retry answer. `billing.py:4-6`",
        "GROUNDED: yes\nREASON: fine\nQUERY: NONE",
    )
    result = run_agent(question="how does retrying work?", k=3, provider=provider, **indexed)
    assert len(result.results) >= 3, "evidence should grow, not be replaced"


def test_the_graph_compiles_with_the_expected_nodes() -> None:
    nodes = set(build_graph(EchoProvider()).get_graph().nodes)
    assert {"retrieve", "generate", "reflect"} <= nodes


# An answer long enough to need citations (>= MIN_ANSWER_CHARS_NEEDING_CITATION)
# but containing none, so reflect rejects it mechanically without a critic call.
UNCITED = (
    "The interest calculation multiplies the principal by one plus the rate "
    "raised to the number of years, which is the standard compound interest "
    "formula, and the retrying client wraps fetch with exponential backoff "
    "so transient failures are retried rather than surfaced."
)


def test_a_citation_failure_rewrites_without_a_second_retrieval(indexed: dict) -> None:
    """The evidence was fine; only the answer's format failed. The retry must
    go straight back to generate — with the citation note, not the generic
    drop-your-claims note that hedges correct answers into worse ones."""
    provider = ScriptedProvider(
        UNCITED,                                               # attempt 1: no citations
        "Interest compounds annually. `billing.py:4-6`",       # attempt 2: cited
        "GROUNDED: yes\nREASON: supported\nQUERY: NONE",
    )
    result = run_agent(question="how is interest compounded?", k=4, provider=provider, **indexed)

    assert result.attempts == 2
    assert result.grounded
    assert len(result.queries) == 1, "a rewrite must not re-run retrieval"

    rewrite_prompt = provider.prompts[1][1]
    assert "rejected for one reason only" in rewrite_prompt
    assert "Do not shorten the answer" in rewrite_prompt


def test_a_loop_that_never_grounds_returns_its_least_bad_answer(indexed: dict) -> None:
    """Observed live: attempt 1 was correct and each retry hedged it further.
    When every attempt fails, the answer returned should be the one that failed
    most softly (critic doubt) rather than whatever came last (uncited)."""
    cited = "Interest compounds annually. `billing.py:4-6`"
    provider = ScriptedProvider(
        cited,                                           # attempt 1: cited, but...
        "GROUNDED: no\nREASON: not convinced\nQUERY: NONE",  # ...critic rejects
        UNCITED,                                         # attempt 2: uncited
        UNCITED,                                         # attempt 3: uncited
    )
    result = run_agent(question="how is interest compounded?", k=4, provider=provider, **indexed)

    assert result.attempts == MAX_ATTEMPTS
    assert not result.grounded
    assert result.answer == cited, "keep the critic-doubted answer over the uncheckable one"
    assert any("kept attempt 1" in line for line in result.trace)


def test_a_second_identical_citation_failure_ends_the_loop() -> None:
    """One rewrite is a fair chance; a second uncited draft proves the model is
    ignoring the instruction, and each extra lap costs a full generation."""
    uncited = Critique(grounded=False, kind="no_citations")
    state = {
        "critique": uncited,
        "critiques": [uncited, uncited],
        "attempts": 2,
        "seen": [_result("a.py", "f")],
    }
    assert should_retry(state) == "end"

    first_failure = {**state, "critiques": [uncited], "attempts": 1}
    assert should_retry(first_failure) == "rewrite", "the first failure still gets its retry"


def test_best_attempt_prefers_soft_failures_and_later_ties() -> None:
    from src.agent.graph import _best_attempt

    crit = Critique(grounded=False, kind="critic")
    uncited = Critique(grounded=False, kind="no_citations")
    fabricated = Critique(grounded=False, kind="fabricated")

    assert _best_attempt(["a", "b", "c"], [crit, uncited, fabricated]) == 0
    assert _best_attempt(["a", "b"], [uncited, crit]) == 1
    assert _best_attempt(["a", "b"], [crit, crit]) == 1, "later attempt saw more evidence"
    assert _best_attempt(["a long draft", "thin"], [uncited, uncited]) == 0, (
        "equally unverifiable drafts: the fuller one tells the user more"
    )


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------


def test_search_code_returns_ranked_chunks(indexed: dict) -> None:
    results = search_code(indexed["repo_path"], "compound interest", 3,
                          indexed["persist_dir"], indexed["bm25_dir"])
    assert results and [r.rank for r in results] == list(range(1, len(results) + 1))


def test_read_file_returns_a_1_based_inclusive_span(repo: Path) -> None:
    sliced = read_file(repo, "billing.py", 4, 6)
    assert sliced.start_line == 4 and sliced.end_line == 6
    assert sliced.text.splitlines()[0].startswith("def compound_interest")
    assert len(sliced.text.splitlines()) == 3
    assert sliced.location == "billing.py:4-6"


def test_read_file_clamps_past_the_end_of_the_file(repo: Path) -> None:
    sliced = read_file(repo, "billing.py", 1, 9999)
    assert sliced.end_line == len(SAMPLE.split("\n"))


def test_read_file_refuses_to_escape_the_repository(repo: Path) -> None:
    """The path comes from a model, so it is checked rather than trusted."""
    with pytest.raises(ValueError, match="outside the repository"):
        read_file(repo, "../../../etc/passwd")


def test_read_file_reports_a_missing_file(repo: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_file(repo, "nope.py")


def test_read_file_truncates_a_huge_span(tmp_path: Path) -> None:
    big = tmp_path / "big.py"
    big.write_text("\n".join(f"x = {n}" for n in range(2000)), encoding="utf-8")
    sliced = read_file(tmp_path, "big.py", 1, 2000)
    assert sliced.truncated
    assert len(sliced.text.splitlines()) == 400


def test_detect_language_picks_the_dominant_one(tmp_path: Path) -> None:
    """A Python project with one vendored .js file should still run pytest."""
    for n in range(3):
        (tmp_path / f"mod{n}.py").write_text("x = 1", encoding="utf-8")
    (tmp_path / "vendor.js").write_text("var x = 1;", encoding="utf-8")

    spec = detect_language(tmp_path)
    assert spec is not None and spec.name == "Python"
    assert spec.test_command == ("python", "-m", "pytest", "-q")


def test_detect_language_returns_none_for_a_repo_with_no_source(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# hi", encoding="utf-8")
    assert detect_language(tmp_path) is None


def test_run_tests_uses_the_command_from_the_registry(tmp_path: Path) -> None:
    """The point of the registry: adding a language brings its runner with it."""
    (tmp_path / "mod.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (tmp_path / "test_mod.py").write_text(
        "from mod import f\n\n\ndef test_f():\n    assert f() == 1\n", encoding="utf-8"
    )

    run = run_tests(tmp_path, timeout=120)
    assert run.command == ("python", "-m", "pytest", "-q")
    assert run.language == "Python"
    assert "$ python -m pytest -q" in run.summary()


def test_run_tests_reports_a_failing_suite_rather_than_raising(tmp_path: Path) -> None:
    (tmp_path / "test_bad.py").write_text("def test_x():\n    assert False\n", encoding="utf-8")
    run = run_tests(tmp_path, timeout=120)
    assert not run.passed
    assert run.exit_code != 0


def test_run_tests_refuses_a_repo_with_no_known_language(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# hi", encoding="utf-8")
    with pytest.raises(ValueError, match="no supported source files"):
        run_tests(tmp_path)


def test_run_tests_survives_a_missing_runner(tmp_path: Path, monkeypatch) -> None:
    """A Go repo on a machine with no Go is a normal situation, not a crash."""
    (tmp_path / "main.go").write_text("package main\n\nfunc main() {}\n", encoding="utf-8")
    run = run_tests(tmp_path, timeout=30)
    assert run.language == "Go"
    assert run.command == ("go", "test", "./...")
    if run.exit_code == 127:
        assert "could not run" in run.stderr


def test_an_answer_with_no_citations_is_not_grounded() -> None:
    """Zero citations passes the fabricated-citation check vacuously.

    Seen on a real run: the model named symbols in prose but never a file and
    line, so there was nothing to verify and the critic accepted it.
    """
    provider = ScriptedProvider("GROUNDED: yes\nREASON: looks right\nQUERY: NONE")
    state = {
        "question": "q",
        "answer": "It uses the should_strip_auth method, which compares hostnames. " * 6,
        "seen": [_result("src/real.py", "real", 1, 5)],
        "attempts": 0,
    }

    critique = reflect(state, provider)["critique"]
    assert not critique.grounded
    assert "cites no file and line" in critique.reason
    assert provider.prompts == [], "no need to pay a critic for an uncheckable answer"


def test_a_short_disclaimer_does_not_need_a_citation() -> None:
    """"The excerpts do not show that" is a good answer, not an ungrounded one."""
    provider = ScriptedProvider("GROUNDED: yes\nREASON: honest about the gap\nQUERY: NONE")
    state = {
        "question": "q",
        "answer": "The excerpts do not show where that is handled.",
        "seen": [_result("src/real.py", "real", 1, 5)],
        "attempts": 0,
    }
    assert reflect(state, provider)["critique"].grounded
