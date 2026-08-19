"""Loading and validating the gold question set.

Three axes exist so a retrieval result can be trusted rather than merely
reported:

* **split** — `dev` is what tuning is allowed to see; `test` is held out. A
  config change that helps dev and not test is overfitting, and the runner is
  what makes that visible instead of flattering.
* **style** — `semantic` questions use plain English ("how does X work"),
  `lexical` ones name identifiers ("what does `should_strip_auth` return").
  Embeddings win the first, BM25 the second, so a set of only one kind measures
  the question writer's habits rather than the retriever.
* **repo** — a fix that only helps one codebase is a fit to that codebase.

The most valuable thing here is `validate`: a label naming a symbol that no
longer exists is invisible at runtime — it silently scores 0 and makes a
retriever look worse than it is.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from src import config

GOLD_PATH: Path = Path(__file__).parent / "gold.json"

# Factual questions have one home in the code; explanatory ones span several
# chunks and are where hybrid retrieval and reflection should show their value.
KINDS = frozenset({"factual", "explanatory"})
STYLES = frozenset({"semantic", "lexical"})
SPLITS = frozenset({"dev", "test"})


@dataclass(frozen=True)
class RepoRef:
    """A checkout the questions are written against."""

    name: str
    path: str    # relative to the project root
    commit: str  # so a label failure can be traced to an upstream change


@dataclass(frozen=True)
class GoldAnswer:
    """One chunk that should be retrieved, identified the way chunks are named."""

    file: str    # repo-relative, POSIX separators, as Chunk.file_path
    symbol: str  # qualified, as Chunk.symbol — e.g. "Session.resolve_redirects"


@dataclass(frozen=True)
class GoldQuestion:
    id: str
    repo: str
    split: str
    kind: str
    style: str
    question: str
    answers: tuple[GoldAnswer, ...]
    note: str = ""


@dataclass(frozen=True)
class GoldSet:
    repos: dict[str, RepoRef]
    questions: tuple[GoldQuestion, ...]

    def __len__(self) -> int:
        return len(self.questions)

    def where(
        self,
        split: str | None = None,
        kind: str | None = None,
        style: str | None = None,
        repo: str | None = None,
    ) -> list[GoldQuestion]:
        """Questions matching every filter given, so the report can slice freely."""
        return [
            q
            for q in self.questions
            if (split is None or q.split == split)
            and (kind is None or q.kind == kind)
            and (style is None or q.style == style)
            and (repo is None or q.repo == repo)
        ]

    def repo_path(self, question: GoldQuestion) -> Path:
        return config.PROJECT_ROOT / self.repos[question.repo].path


def load_gold(path: str | Path = GOLD_PATH) -> GoldSet:
    """Read and structurally validate the gold set."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    repos = {name: RepoRef(name, r["path"], r["commit"]) for name, r in data["repos"].items()}

    questions = []
    seen: set[str] = set()
    for raw in data["questions"]:
        qid = raw["id"]
        if qid in seen:
            raise ValueError(f"duplicate question id {qid!r}")
        seen.add(qid)

        for field, allowed in (("kind", KINDS), ("style", STYLES), ("split", SPLITS)):
            if raw[field] not in allowed:
                raise ValueError(f"{qid}: {field} {raw[field]!r} not in {sorted(allowed)}")
        if raw["repo"] not in repos:
            raise ValueError(f"{qid}: unknown repo {raw['repo']!r}")
        if not raw["answers"]:
            raise ValueError(f"{qid}: every question needs at least one answer")

        questions.append(
            GoldQuestion(
                id=qid,
                repo=raw["repo"],
                split=raw["split"],
                kind=raw["kind"],
                style=raw["style"],
                question=raw["question"],
                answers=tuple(GoldAnswer(a["file"], a["symbol"]) for a in raw["answers"]),
                note=raw.get("note", ""),
            )
        )

    return GoldSet(repos=repos, questions=tuple(questions))


def validate(gold: GoldSet) -> list[str]:
    """Check every label against the chunks the ingester actually produces.

    Returns a list of problems, empty when the set is clean. Returning rather
    than raising so one run reports every bad label at once — fixing them one
    traceback at a time would be miserable.
    """
    from src.ingest import ingest_repo

    problems: list[str] = []
    for name, ref in gold.repos.items():
        root = config.PROJECT_ROOT / ref.path
        if not root.exists():
            problems.append(f"{name}: no checkout at {ref.path}")
            continue

        real = {(c.file_path, c.symbol) for c in ingest_repo(root)}
        for question in gold.where(repo=name):
            for answer in question.answers:
                if (answer.file, answer.symbol) not in real:
                    near = sorted(s for f, s in real if f == answer.file)
                    hint = (
                        f" (symbols in that file: {near[:6]}...)" if near else " (file not indexed)"
                    )
                    problems.append(f"{question.id}: no chunk {answer.file}::{answer.symbol}{hint}")
    return problems


def summary(gold: GoldSet) -> str:
    """A few lines describing the set, for the CLI and the README."""
    lines = [f"{len(gold)} questions across {len(gold.repos)} repos"]
    for split in sorted(SPLITS):
        rows = gold.where(split=split)
        by_style = ", ".join(
            f"{len([q for q in rows if q.style == s])} {s}" for s in sorted(STYLES)
        )
        by_kind = ", ".join(f"{len([q for q in rows if q.kind == k])} {k}" for k in sorted(KINDS))
        lines.append(f"  {split:<5} {len(rows):>3}  ({by_kind}; {by_style})")
    for name, ref in sorted(gold.repos.items()):
        lines.append(f"  {name:<10} {len(gold.where(repo=name)):>3} questions @ {ref.commit}")
    return "\n".join(lines)


def questions_for(gold: GoldSet, ids: Sequence[str] | None = None) -> list[GoldQuestion]:
    """All questions, or just the named ones — useful when debugging one failure."""
    if not ids:
        return list(gold.questions)
    wanted = set(ids)
    return [q for q in gold.questions if q.id in wanted]
