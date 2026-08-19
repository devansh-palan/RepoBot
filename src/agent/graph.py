"""The LangGraph state machine: retrieve -> generate -> reflect -> (retry | end).

    START ──> retrieve ──> generate ──> reflect ──┬── grounded ──────> END
                 ^              ^                 │
                 │              └──── rewrite ────┤
                 └──────────────────── search ────┴── out of attempts ─> END

Three nodes and one decision. The decision lives in `should_retry`, a pure
function of state, so the control flow can be tested without a model.

The retry edges are the whole point of the graph. Plain RAG
(`qa.answer_question`) retrieves once and answers once; if retrieval missed,
the answer is ungrounded and nothing notices. Here the critic gets a veto, and
its verdict picks the cure: `search` feeds a *new* query into retrieval when
evidence is missing, `rewrite` goes straight back to generate when the evidence
was fine and only the answer failed — re-searching the same question would
retrieve the same chunks and pay for a model call to learn nothing.
"""

from __future__ import annotations

import re
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from src import config
from src.index import SearchResult, chunk_id

from .llm import Provider, get_provider
from .prompts import (
    CITE_RETRY_NOTE,
    NO_CONTEXT_ANSWER,
    REFLECTION_PROMPT,
    RETRY_NOTE,
    SYSTEM_PROMPT,
    build_reflection_prompt,
    build_user_prompt,
)
from .state import AgentResult, AgentState, Critique
from .tools import format_results, search_code

# One attempt is plain RAG. The cap makes the loop terminate no matter what the
# critic says — a critic that never accepts anything would otherwise spin
# forever, and on a local model that is a real possibility, not a hypothetical.
MAX_ATTEMPTS = 3

# An answer shorter than this is something like "the excerpts do not show that",
# which is a legitimate response and needs no citation. Anything longer is
# making claims, and a claim with no citation cannot be checked.
MIN_ANSWER_CHARS_NEEDING_CITATION = 200


def _merge(seen: list[SearchResult], fresh: list[SearchResult]) -> list[SearchResult]:
    """Union of chunks across attempts, first occurrence winning.

    Accumulating rather than replacing is what stops the loop making things
    worse: a retry that retrieves badly still leaves the first attempt's good
    chunks in front of the model.
    """
    known = {chunk_id(r.chunk) for r in seen}
    return list(seen) + [r for r in fresh if chunk_id(r.chunk) not in known]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def retrieve(state: AgentState) -> AgentState:
    """Search the repo and add what it finds to the accumulated evidence.

    On the first pass the query is the question. On a retry it is whatever the
    critic asked for, because re-running the original query returns the original
    chunks and the loop would not progress.
    """
    critique = state.get("critique")
    query = (critique.followup_query if critique and critique.followup_query else "") or state[
        "question"
    ]

    fresh = search_code(
        state["repo_path"],
        query,
        state.get("k", config.FINAL_TOP_K),
        state.get("persist_dir"),
        state.get("bm25_dir"),
    )
    seen = _merge(state.get("seen", []), fresh)

    return {
        "queries": [*state.get("queries", []), query],
        "results": fresh,
        "seen": seen,
        "trace": [
            *state.get("trace", []),
            f"retrieve[{len(state.get('queries', [])) + 1}] {query!r} -> "
            f"{len(fresh)} chunks ({len(seen)} known)",
        ],
    }


def generate(state: AgentState, provider: Provider) -> AgentState:
    """Answer from the accumulated evidence.

    Uses `seen`, not `results`: on a retry the model should see everything found
    so far, not only the newest search.
    """
    evidence = state.get("seen", [])
    if not evidence:
        return {
            "answer": NO_CONTEXT_ANSWER,
            "trace": [*state.get("trace", []), "generate: nothing retrieved, skipping the model"],
        }

    prompt = build_user_prompt(state["question"], evidence)
    critique = state.get("critique")
    if critique and not critique.grounded:
        # Tell the model what was wrong last time, so the retry is a correction
        # rather than a coin flip. The note matches the failure: a citation-only
        # failure asks for the same answer with citations attached, because the
        # generic "drop unsupported claims" note makes the model hedge a correct
        # answer into a thinner one.
        if critique.kind == "no_citations":
            prompt += CITE_RETRY_NOTE
        else:
            prompt += RETRY_NOTE.format(reason=critique.reason or "claims were unsupported")

    response = provider.complete(system=SYSTEM_PROMPT, user=prompt)

    return {
        "answer": response.text,
        "model": response.model,
        "input_tokens": state.get("input_tokens", 0) + response.input_tokens,
        "output_tokens": state.get("output_tokens", 0) + response.output_tokens,
        "trace": [
            *state.get("trace", []),
            f"generate: {len(response.text)} chars from {len(evidence)} chunks",
        ],
    }


def reflect(state: AgentState, provider: Provider) -> AgentState:
    """Decide whether every claim is backed by retrieved code.

    Two checks, cheapest first:

    1. Mechanical — does the answer cite a location that was never retrieved?
       That is a fabricated citation, and it is decided by set membership, so
       the model cannot argue its way out of it and it costs nothing.
    2. The critic model — are the claims actually supported by the excerpts?
       This catches the harder case: a real citation attached to a claim the
       cited code does not make.
    """
    answer = state.get("answer", "")
    evidence = state.get("seen", [])
    attempts = state.get("attempts", 0) + 1

    if not evidence:
        # Nothing was retrieved, so there is nothing to be grounded in and no
        # point paying for a critique.
        critique = Critique(
            grounded=False,
            reason="retrieval found no code for this question",
            kind="no_evidence",
        )
        return _record(state, critique, attempts)

    retrieved = {r.chunk.location for r in evidence}
    cited = _citations(answer)

    # An answer with no citations at all passes the fabricated-citation check
    # vacuously, which is how an unverifiable answer sneaks through. Observed on
    # a real run: the model named symbols but never a file:line, so there was
    # nothing to check and the critic waved it through.
    #
    # Prose citations count too. A small model writes "the `_verify.js` file,
    # line 1-14" as often as "`api/_verify.js:1-14`"; both point at retrieved
    # code, and failing the first on format alone rejects honest answers. So
    # does naming a retrieved file with no line range at all — "the README.md
    # explains how to deploy" is checkable against the retrieved README chunk,
    # just at file granularity. All of these send the answer to the critic to
    # judge substance; only an answer referencing *nothing retrieved* — bare
    # symbol names, pure prose — fails here, because then there is nothing to
    # check it against.
    checkable = (
        _matches_evidence(_loose_citations(answer), evidence)
        or _mentions_retrieved_file(answer, evidence)
    )
    if not cited and not checkable and len(answer) >= MIN_ANSWER_CHARS_NEEDING_CITATION:
        critique = Critique(
            grounded=False,
            reason="makes claims but cites no file and line, so nothing can be checked",
            followup_query="",  # retrieval was fine; the answer needs rewriting
            kind="no_citations",
        )
        return _record(state, critique, attempts)

    unsupported = tuple(
        c for c in cited if c not in retrieved and not _cites_retrieved_chunk(c, evidence)
    )

    if unsupported:
        critique = Critique(
            grounded=False,
            reason=f"cites code that was never retrieved: {', '.join(unsupported[:3])}",
            followup_query=_query_from_citations(unsupported),
            unsupported=unsupported,
            kind="fabricated",
        )
        return _record(state, critique, attempts)

    verdict = _ask_critic(provider, state["question"], answer, evidence)
    return _record(state, verdict, attempts)


def _record(state: AgentState, critique: Critique, attempts: int) -> AgentState:
    return {
        "attempts": attempts,
        "critique": critique,
        "critiques": [*state.get("critiques", []), critique],
        "answers": [*state.get("answers", []), state.get("answer", "")],
        "trace": [*state.get("trace", []), f"reflect[{attempts}] {critique}"],
    }


def _ask_critic(
    provider: Provider,
    question: str,
    answer: str,
    evidence: list[SearchResult],
) -> Critique:
    """Run the critic model and parse its three-line verdict.

    Parsing is deliberately forgiving. A small local model will not always
    produce the exact format, and a malformed reply should not be read as "not
    grounded" — that would send the graph into a retry loop over a formatting
    slip rather than a real problem.
    """
    # No model override: the critic runs on whatever the provider is configured
    # for. Splitting answerer and critic onto different models is a knob the
    # ablation may want later, but guessing at it now would hard-code a hosted
    # model name into a path that usually runs locally.
    response = provider.complete(
        system=REFLECTION_PROMPT,
        user=build_reflection_prompt(question, answer, format_results(evidence)),
    )

    grounded, reason, query = True, "", ""
    for line in response.text.splitlines():
        head, _, rest = line.partition(":")
        key, value = head.strip().upper(), rest.strip()
        if key == "GROUNDED":
            grounded = not value.lower().startswith("no")
        elif key == "REASON":
            reason = value
        elif key == "QUERY" and value.upper() != "NONE":
            query = value

    return Critique(
        grounded=grounded,
        reason=reason,
        followup_query="" if grounded else query,
        kind="" if grounded else "critic",
    )


# A backticked citation whose path contains spaces: `PLI Website Dev.md:4-9`.
# The bare regex below cannot allow spaces (it would swallow prose), but inside
# backticks the boundary is explicit. Without this, citing a spacey filename
# extracts only the tail after the last space — which then fails the
# fabricated-citation check on a parsing artifact, not a lie.
_TICKED_CITATION = re.compile(r"`([^`:\n]+\.\w{1,6}:\d+-\d+)`")


def _citations(answer: str) -> list[str]:
    """Every `path:start-end` the answer claims, deduplicated, in order."""
    from .qa import _CITATION

    seen: dict[str, None] = {}
    for match in _TICKED_CITATION.findall(answer):
        seen.setdefault(match.strip(), None)
    for match in _CITATION.findall(answer):
        # Skip tails of an already-captured spacey citation.
        if not any(full.endswith(match) for full in seen):
            seen.setdefault(match, None)
    return list(seen)


def _cites_retrieved_chunk(citation: str, evidence: list[SearchResult]) -> bool:
    """Whether a citation points into retrieved code, allowing sloppy paths.

    Exact location match is handled by the caller; this is the fallback for a
    dropped directory prefix or a sub-span — "src/api/x.js:3-9" cited as
    "x.js:4-8". Basename plus overlapping lines is the intended reference.
    """
    path, _, span = citation.rpartition(":")
    start_text, _, end_text = span.partition("-")
    try:
        start, end = int(start_text), int(end_text)
    except ValueError:
        return False
    return _matches_evidence([(path, start, end)], evidence)


# "the `_verify.js` file, line 1-14" / "README.md, lines 68-80" — a filename
# followed shortly by a line range...
_LOOSE_FILE_FIRST = re.compile(
    r"[`\"']?(?P<file>[\w ./\-]+\.\w{1,6})[`\"']?(?:\s+file)?[^.\n]{0,40}?"
    r"\blines?\s+(?P<a>\d+)\s*[-–]\s*(?P<b>\d+)"
)
# ...and "line 444-449 of `PLI Website Development.md`" — the reverse order.
_LOOSE_RANGE_FIRST = re.compile(
    r"\blines?\s+(?P<a>\d+)\s*[-–]\s*(?P<b>\d+)[^.\n]{0,40}?"
    r"[`\"'](?P<file>[\w ./\-]+\.\w{1,6})[`\"']"
)


def _loose_citations(answer: str) -> list[tuple[str, int, int]]:
    """(filename, start, end) for every prose-style citation in the answer."""
    found = []
    for pattern in (_LOOSE_FILE_FIRST, _LOOSE_RANGE_FIRST):
        for match in pattern.finditer(answer):
            found.append(
                (match.group("file").strip(), int(match.group("a")), int(match.group("b")))
            )
    return found


def _mentions_retrieved_file(answer: str, evidence: list[SearchResult]) -> bool:
    """Whether the answer names any file that was actually retrieved.

    Matched on the full relative path or the basename. Only retrieved files
    count — an answer that names files the search never returned is exactly
    the unverifiable kind this check exists to catch.
    """
    lowered = answer.lower()
    for result in evidence:
        path = result.chunk.file_path.lower()
        if path in lowered or path.rsplit("/", 1)[-1] in lowered:
            return True
    return False


def _matches_evidence(
    loose: list[tuple[str, int, int]], evidence: list[SearchResult]
) -> bool:
    """Whether any prose citation points into a retrieved chunk.

    Matched on basename plus overlapping line range: the model often drops the
    directory ("_verify.js" for "api/_verify.js") and quotes a sub-span of the
    chunk, neither of which makes the reference wrong.
    """
    for name, start, end in loose:
        base = name.rsplit("/", 1)[-1].lower()
        for result in evidence:
            chunk = result.chunk
            if chunk.file_path.rsplit("/", 1)[-1].lower() != base:
                continue
            if start <= chunk.end_line and end >= chunk.start_line:
                return True
    return False


def _query_from_citations(unsupported: tuple[str, ...]) -> str:
    """Turn a fabricated citation into something worth searching for.

    The model invented `src/foo/bar.py:10-20`, which suggests it expected code
    in `bar.py`. Searching that name is more useful than repeating the question.
    """
    names = [c.rsplit(":", 1)[0].rsplit("/", 1)[-1].removesuffix(".py") for c in unsupported]
    return " ".join(dict.fromkeys(names))[:120]


# ---------------------------------------------------------------------------
# The conditional edge
# ---------------------------------------------------------------------------


def should_retry(state: AgentState) -> str:
    """The one decision in the graph: stop, search for more code, or rewrite.

    A pure function of state, so the control flow is testable without a model
    and without a network.

    `rewrite` exists because two failures need no new evidence: an answer with
    no citations, and a critic rejection with no better query to offer. Routing
    those through retrieve would re-run the original question, retrieve the
    exact chunks already in `seen`, and add ~half the retry's wall clock for
    nothing.
    """
    critique = state.get("critique")
    if critique is not None and critique.grounded:
        return "end"
    if state.get("attempts", 0) >= MAX_ATTEMPTS:
        return "end"  # answer anyway, flagged ungrounded — see AgentResult.grounded
    if not state.get("seen") and not state.get("queries"):
        return "end"
    # The same mechanical failure twice means the model is not going to comply
    # with the correction — seen live: three uncited drafts in a row, each retry
    # paying full generation cost to fail the identical check. One retry gets
    # the benefit of the doubt; a second identical failure ends the loop.
    kinds = [c.kind for c in state.get("critiques", [])]
    if critique is not None and critique.kind == "no_citations" and kinds.count("no_citations") >= 2:
        return "end"
    if critique is not None and not critique.followup_query and state.get("seen"):
        return "rewrite"
    return "search"


# How bad each failure is, best to worst. A critic rejection may be the 7B
# critic being wrong; an uncited answer is unverifiable but may be right; a
# fabricated citation is actively false; no evidence answered nothing. Used to
# pick which attempt to return when the loop runs out — observed live: attempt
# 1 was correct, and each retry hedged it further, so returning the *last*
# answer returned the worst one.
_SEVERITY = {"": 0, "critic": 1, "no_citations": 2, "fabricated": 3, "no_evidence": 4}


def _best_attempt(answers: list[str], critiques: list[Critique]) -> int:
    """Index of the least-bad attempt.

    Ties on severity go to the fullest answer, then the latest: three uncited
    drafts are equally unverifiable, but the 1700-char one tells the user more
    than the 500-char one — observed live when retries shrank the draft.
    """
    return min(
        range(len(answers)),
        key=lambda i: (_SEVERITY.get(critiques[i].kind, 1), -len(answers[i]), -i),
    )


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


def build_graph(provider: Provider | None = None):
    """Wire the nodes together and compile.

    The provider is bound here rather than carried in state: it holds a live
    client, and state should stay plain data that can be logged or serialised.
    """
    provider = provider or get_provider()

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", lambda s: generate(s, provider))
    graph.add_node("reflect", lambda s: reflect(s, provider))

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")   # never answer without evidence
    graph.add_edge("generate", "reflect")    # never return an unchecked answer
    graph.add_conditional_edges(
        "reflect",
        should_retry,
        {"search": "retrieve", "rewrite": "generate", "end": END},
    )
    return graph.compile()


def run_agent(
    repo_path: str | Path,
    question: str,
    k: int = config.FINAL_TOP_K,
    provider: Provider | None = None,
    persist_dir: str | None = None,
    bm25_dir: str | Path | None = None,
) -> AgentResult:
    """Answer a question with retrieval, generation, and a reflection loop."""
    app = build_graph(provider)
    final = app.invoke(
        {
            "question": question,
            "repo_path": str(repo_path),
            "k": k,
            "persist_dir": persist_dir,
            "bm25_dir": str(bm25_dir) if bm25_dir else None,
            "attempts": 0,
            "queries": [],
            "seen": [],
            "critiques": [],
            "answers": [],
            "trace": [],
        },
        # One extra step per node per attempt, plus slack for the entry edge.
        {"recursion_limit": MAX_ATTEMPTS * 3 + 5},
    )

    answer = final.get("answer", "")
    answers, critiques = final.get("answers", []), final.get("critiques", [])
    trace = final.get("trace", [])
    if answers and critiques and not critiques[-1].grounded:
        best = _best_attempt(answers, critiques)
        if best != len(answers) - 1:
            answer = answers[best]
            trace = [*trace, f"kept attempt {best + 1}'s answer ({critiques[best].kind or 'least bad'}) over attempt {len(answers)}'s ({critiques[-1].kind})"]

    return AgentResult(
        question=question,
        answer=answer,
        results=final.get("seen", []),
        critiques=critiques,
        queries=final.get("queries", []),
        trace=trace,
        attempts=final.get("attempts", 0),
        model=final.get("model", ""),
        input_tokens=final.get("input_tokens", 0),
        output_tokens=final.get("output_tokens", 0),
    )
