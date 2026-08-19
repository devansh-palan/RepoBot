"""The ablation: four configurations, one gold set, one table.

The four form a ladder, each adding one thing to the one before:

    1. vector    retrieval only, dense
    2. hybrid    retrieval only, dense + BM25 fused with RRF
    3. rag       hybrid retrieval + one generation, no checking
    4. agent     the same, plus a reflection loop that can retry retrieval

So each row isolates one decision. 1 vs 2 measures fusion. 2 vs 3 measures
nothing about retrieval — it adds the answer. 3 vs 4 measures reflection, which
is the comparison CLAUDE.md's definition of done asks for.

Answer accuracy is judged two ways on purpose. A factual question has one right
place in the code, so "did the answer cite it" is objective, free, and cannot be
talked into a wrong verdict. An explanatory question has no such test, so those
go to an LLM judge — which is the weaker instrument and is labelled as such
wherever it is reported.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from src import config
from src.agent import answer_question, get_provider, run_agent
from src.agent.llm import Provider
from src.index import SearchResult
from src.retrieve import RETRIEVERS

from .gold import GoldQuestion, GoldSet
from .metrics import mean, recall_at_k, reciprocal_rank

RECALL_K = 5
RETRIEVAL_DEPTH = 20

#: name -> (what it retrieves with, whether it answers, whether it reflects)
CONFIGS: dict[str, tuple[str, bool, bool]] = {
    "vector": ("vector", False, False),
    "hybrid": ("hybrid", False, False),
    "rag": ("hybrid", True, False),
    "agent": ("hybrid", True, True),
}

CONFIG_LABELS = {
    "vector": "vector-only",
    "hybrid": "hybrid (RRF)",
    "rag": "agent, no reflection",
    "agent": "agent + reflection",
}

# correct / partially correct / wrong. Partial credit exists because an
# explanatory answer that gets two of three mechanisms right is genuinely
# better than one that gets none, and collapsing that to 0 would hide it.
VERDICT_SCORE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}

JUDGE_PROMPT = """\
You are grading an answer about a codebase against the code that actually \
answers the question.

Reply with exactly two lines and nothing else:

VERDICT: correct|partial|incorrect
REASON: <one sentence>

correct   - the answer describes what the reference code does, with no claim \
that contradicts it.
partial   - some of it is right and some is missing or wrong.
incorrect - it describes the wrong mechanism, or contradicts the reference code.

Judge only against the reference code below. Do not reward fluency, length, or \
plausible-sounding detail that the reference code does not show. An answer that \
admits it could not find something is `partial`, not `incorrect`.
"""


@dataclass
class QuestionOutcome:
    """What one config produced for one question."""

    question_id: str
    kind: str
    style: str
    repo: str
    recall: float = 0.0
    reciprocal_rank: float = 0.0
    accuracy: float | None = None   # None when the config produces no answer
    verdict: str = ""
    judged_by: str = ""             # "citation" or "llm"
    grounded: bool | None = None
    attempts: int = 0
    seconds: float = 0.0
    answer: str = ""


@dataclass
class ConfigResult:
    name: str
    outcomes: list[QuestionOutcome] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def answers(self) -> bool:
        return any(o.accuracy is not None for o in self.outcomes)

    def recall(self, **where: str) -> float | None:
        """None when the slice is empty — a score of 0.0 would be a lie.

        Learnt the hard way: a run that sampled no lexical questions reported
        `lexical R@5 = 0.000` across every config, which reads as catastrophic
        failure rather than "not measured".
        """
        rows = self._slice(**where)
        return mean(o.recall for o in rows) if rows else None

    def mrr(self, **where: str) -> float | None:
        rows = self._slice(**where)
        return mean(o.reciprocal_rank for o in rows) if rows else None

    def accuracy(self, **where: str) -> float | None:
        scores = [o.accuracy for o in self._slice(**where) if o.accuracy is not None]
        return mean(scores) if scores else None

    def seconds(self) -> float:
        return sum(o.seconds for o in self.outcomes)

    def _slice(self, **where: str) -> list[QuestionOutcome]:
        return [
            o for o in self.outcomes
            if all(getattr(o, key) == value for key, value in where.items())
        ]


# ---------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------


def gold_code(question: GoldQuestion, gold: GoldSet) -> str:
    """The reference code a judge grades against, read from the repo."""
    from src.ingest import ingest_repo

    wanted = {(a.file, a.symbol) for a in question.answers}
    chunks = [
        c for c in _chunks_for(gold, question.repo)
        if (c.file_path, c.symbol) in wanted
    ]
    return "\n\n".join(f"{c.location}  {c.symbol}\n```\n{c.code}\n```" for c in chunks)


_CHUNK_CACHE: dict[str, list] = {}


def _chunks_for(gold: GoldSet, repo: str) -> list:
    """Chunks for a repo, parsed once per process rather than once per question."""
    if repo not in _CHUNK_CACHE:
        from src.ingest import ingest_repo

        _CHUNK_CACHE[repo] = ingest_repo(config.PROJECT_ROOT / gold.repos[repo].path)
    return _CHUNK_CACHE[repo]


def judge_by_citation(answer: str, question: GoldQuestion) -> tuple[float, str]:
    """Objective grading for a factual question: did it cite the right place?

    No model involved, so it cannot be flattered by a fluent wrong answer, and
    it costs nothing. Only works because a factual question has exactly one home
    in the code.
    """
    from src.agent.graph import _citations

    cited = set(_citations(answer))
    wanted = {f"{a.file}:" for a in question.answers}

    # Compare on file and symbol rather than exact line range: the answer may
    # cite a different window of the same function, which is still correct.
    hit = any(
        any(c.startswith(prefix) for prefix in wanted) for c in cited
    )
    if hit:
        return 1.0, "correct"
    # The symbol named in prose, without a citation, is partial credit: right
    # answer, unverifiable form.
    named = any(a.symbol.split(".")[-1] in answer for a in question.answers)
    return (0.5, "partial") if named else (0.0, "incorrect")


def judge_by_llm(
    answer: str,
    question: GoldQuestion,
    gold: GoldSet,
    provider: Provider,
) -> tuple[float, str]:
    """LLM-as-judge for an explanatory question, graded against the gold code."""
    reference = gold_code(question, gold)
    if not reference:
        return 0.0, "incorrect"

    response = provider.complete(
        system=JUDGE_PROMPT,
        user=(
            f"Question: {question.question}\n\n"
            f"Reference code:\n\n{reference}\n\n"
            f"---\n\nAnswer to grade:\n\n{answer}"
        ),
    )

    verdict = "partial"
    for line in response.text.splitlines():
        head, _, rest = line.partition(":")
        if head.strip().upper() == "VERDICT":
            candidate = rest.strip().lower().split()[0] if rest.strip() else ""
            if candidate in VERDICT_SCORE:
                verdict = candidate
            break
    return VERDICT_SCORE[verdict], verdict


# ---------------------------------------------------------------------------
# Running one config
# ---------------------------------------------------------------------------


def _score_retrieval(results: list[SearchResult], question: GoldQuestion) -> tuple[float, float]:
    return (
        recall_at_k(results, question.answers, RECALL_K),
        reciprocal_rank(results, question.answers),
    )


def run_config(
    name: str,
    gold: GoldSet,
    questions: list[GoldQuestion],
    provider: Provider | None = None,
    cache: Path | None = None,
) -> ConfigResult:
    """Run one configuration over the questions given."""
    retriever, answers, reflects = CONFIGS[name]
    provider = provider or get_provider()
    result = ConfigResult(name=name)

    for question in questions:
        cached = _load(cache, name, question, provider)
        if cached is not None:
            result.outcomes.append(cached)
            continue

        started = time.perf_counter()
        repo = gold.repo_path(question)

        if not answers:
            found = RETRIEVERS[retriever](repo, question.question, RETRIEVAL_DEPTH)
            recall, rr = _score_retrieval(found, question)
            outcome = QuestionOutcome(
                question_id=question.id, kind=question.kind, style=question.style,
                repo=question.repo, recall=recall, reciprocal_rank=rr,
                seconds=time.perf_counter() - started,
            )
        else:
            if reflects:
                run = run_agent(repo, question.question, k=config.FINAL_TOP_K, provider=provider)
                text, found = run.answer, run.results
                grounded, attempts = run.grounded, run.attempts
                result.input_tokens += run.input_tokens
                result.output_tokens += run.output_tokens
            else:
                run = answer_question(
                    repo, question.question, k=config.FINAL_TOP_K, provider=provider
                )
                text, found = run.text, run.results
                grounded, attempts = None, 1
                result.input_tokens += run.input_tokens
                result.output_tokens += run.output_tokens

            recall, rr = _score_retrieval(found, question)
            if question.kind == "factual":
                score, verdict = judge_by_citation(text, question)
                judged_by = "citation"
            else:
                score, verdict = judge_by_llm(text, question, gold, provider)
                judged_by = "llm"

            outcome = QuestionOutcome(
                question_id=question.id, kind=question.kind, style=question.style,
                repo=question.repo, recall=recall, reciprocal_rank=rr,
                accuracy=score, verdict=verdict, judged_by=judged_by,
                grounded=grounded, attempts=attempts,
                seconds=time.perf_counter() - started, answer=text,
            )

        result.outcomes.append(outcome)
        _save(cache, name, question, provider, outcome)

    return result


def run_ablation(
    gold: GoldSet,
    questions: list[GoldQuestion],
    names: list[str] | None = None,
    provider: Provider | None = None,
    cache: Path | None = None,
    on_progress=None,
) -> dict[str, ConfigResult]:
    results: dict[str, ConfigResult] = {}
    for name in names or list(CONFIGS):
        if on_progress:
            on_progress(name)
        results[name] = run_config(name, gold, questions, provider, cache)
    return results


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def _settings_fingerprint() -> str:
    """Hash of everything outside the model that shapes an outcome.

    A cached score is only reusable while retrieval tuning and prompts are what
    they were when it was computed. Without this in the key, editing config or
    a prompt and re-running the ablation silently reports the old system's
    numbers under the new system's name.
    """
    import hashlib

    from src.agent import prompts
    from src.retrieve.retrievers import _IDENTIFIER

    payload = repr((
        _IDENTIFIER.pattern,
        config.VECTOR_TOP_K, config.BM25_TOP_K, config.FINAL_TOP_K,
        config.RRF_K, config.BM25_K1, config.BM25_B,
        config.RRF_BM25_WEIGHT_LEXICAL, config.RRF_BM25_WEIGHT_SEMANTIC,
        config.EMBEDDING_MODEL,
        sorted(config.DOC_EXTENSIONS), config.DOC_STEM_PATTERN,
        prompts.SYSTEM_PROMPT, prompts.REFLECTION_PROMPT,
        prompts.RETRY_NOTE, prompts.CITE_RETRY_NOTE,
    ))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]


def _cache_file(cache: Path, name: str, question: GoldQuestion, provider: Provider) -> Path:
    model = getattr(provider, "model", provider.name).replace(":", "-").replace("/", "-")
    return cache / model / _settings_fingerprint() / name / f"{question.id}.json"


def _load(
    cache: Path | None, name: str, question: GoldQuestion, provider: Provider
) -> QuestionOutcome | None:
    """Reuse a previous run of the same question under the same config.

    A full ablation is hours of local generation; without this, one crash or one
    added config means starting over.
    """
    if cache is None:
        return None
    path = _cache_file(cache, name, question, provider)
    if not path.is_file():
        return None
    try:
        return QuestionOutcome(**json.loads(path.read_text(encoding="utf-8")))
    except (OSError, TypeError, ValueError):
        return None


def _save(
    cache: Path | None,
    name: str,
    question: GoldQuestion,
    provider: Provider,
    outcome: QuestionOutcome,
) -> None:
    if cache is None:
        return
    path = _cache_file(cache, name, question, provider)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(outcome.__dict__, indent=1), encoding="utf-8")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _cell(value: float | None, width: int = 5) -> str:
    # "n/a" rather than an em dash: the report is printed to a Windows console
    # as often as it is read as markdown, and cp1252 cannot encode the dash.
    return "n/a" if value is None else f"{value:.{width - 2}f}"


def markdown_table(results: dict[str, ConfigResult], questions: list[GoldQuestion]) -> str:
    """The headline table: one row per config."""
    n = len(questions)
    lines = [
        f"| config | R@{RECALL_K} | MRR | answer accuracy | wall clock |",
        "|---|---|---|---|---|",
    ]
    for name, result in results.items():
        lines.append(
            f"| {CONFIG_LABELS[name]} | {_cell(result.recall())} | {_cell(result.mrr())} "
            f"| {_cell(result.accuracy())} | {result.seconds():.0f}s |"
        )
    lines.append("")
    lines.append(
        f"_{n} questions. Retrieval metrics over the top {RETRIEVAL_DEPTH} retrieved; "
        f"answer accuracy is citation-checked for factual questions and "
        f"LLM-judged for explanatory ones._"
    )
    return "\n".join(lines)


def breakdown_table(results: dict[str, ConfigResult], gold: GoldSet) -> str:
    """Where each config wins and loses, sliced by the axes the gold set carries."""
    slices = [
        ("factual", {"kind": "factual"}),
        ("explanatory", {"kind": "explanatory"}),
        ("semantic", {"style": "semantic"}),
        ("lexical", {"style": "lexical"}),
    ]
    header = "| config | " + " | ".join(f"{label} R@{RECALL_K}" for label, _ in slices) + " |"
    lines = [header, "|---" * (len(slices) + 1) + "|"]
    for name, result in results.items():
        cells = " | ".join(_cell(result.recall(**where)) for _, where in slices)
        lines.append(f"| {CONFIG_LABELS[name]} | {cells} |")

    lines += ["", "| config | " + " | ".join(f"{label} acc" for label, _ in slices) + " |",
              "|---" * (len(slices) + 1) + "|"]
    for name, result in results.items():
        cells = " | ".join(_cell(result.accuracy(**where)) for _, where in slices)
        lines.append(f"| {CONFIG_LABELS[name]} | {cells} |")
    return "\n".join(lines)


def failure_report(results: dict[str, ConfigResult], limit: int = 12) -> str:
    """Questions the best answering config still got wrong, worst first."""
    answering = [r for r in results.values() if r.answers]
    if not answering:
        return "_no answering configs were run_"

    best = max(answering, key=lambda r: r.accuracy() or 0.0)
    failures = sorted(
        (o for o in best.outcomes if (o.accuracy or 0.0) < 1.0),
        key=lambda o: (o.accuracy or 0.0, -o.recall),
    )[:limit]

    lines = [
        f"Failures under **{CONFIG_LABELS[best.name]}** "
        f"({len(failures)} of {len(best.outcomes)} below full marks):",
        "",
        f"| question | kind | style | R@{RECALL_K} | verdict | grounded |",
        "|---|---|---|---|---|---|",
    ]
    for o in failures:
        grounded = "n/a" if o.grounded is None else ("yes" if o.grounded else "no")
        lines.append(
            f"| {o.question_id} | {o.kind} | {o.style} | {o.recall:.2f} "
            f"| {o.verdict} | {grounded} |"
        )
    return "\n".join(lines)


def report(
    results: dict[str, ConfigResult],
    gold: GoldSet,
    questions: list[GoldQuestion],
) -> str:
    return "\n\n".join([
        "## Ablation",
        markdown_table(results, questions),
        "### By question kind and style",
        breakdown_table(results, gold),
        "### Where the failures cluster",
        failure_report(results),
    ])
