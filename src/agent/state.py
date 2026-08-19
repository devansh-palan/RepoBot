"""The state that flows through the graph.

One dict, passed node to node, each node returning only the keys it changed.
Everything the reflection loop needs to decide is in here, which is what makes
the conditional edge a pure function of state rather than of hidden context.

Two fields are worth calling out:

* `seen` accumulates chunks across *all* retrieval attempts, while `results`
  holds only the latest. Without `seen`, a second attempt that retrieves worse
  chunks would throw away good ones from the first, and the loop could make the
  answer worse rather than better.
* `queries` records what has already been searched, so the retry can be told to
  ask something different. Retrying the same query returns the same chunks and
  the loop spins without progress.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

from src.index import SearchResult


@dataclass(frozen=True)
class Critique:
    """The reflect node's verdict on one answer."""

    grounded: bool
    reason: str = ""
    # What to search for next. Empty when grounded, or when the critic had no
    # better idea — the graph falls back to the original question in that case.
    followup_query: str = ""
    # Citations the answer made that were never retrieved. Found mechanically,
    # not by the model, so it cannot be talked out of them.
    unsupported: tuple[str, ...] = ()
    # Which check rejected the answer: "" (grounded), "no_evidence",
    # "no_citations", "fabricated", or "critic". The retry uses this to decide
    # whether the *answer* needs rewriting or the *evidence* needs improving —
    # a citation-format failure re-searched with the same query would retrieve
    # the same chunks and burn a model call for nothing.
    kind: str = ""

    def __str__(self) -> str:
        verdict = "grounded" if self.grounded else "NOT grounded"
        return f"{verdict}: {self.reason}" if self.reason else verdict


class AgentState(TypedDict, total=False):
    """Everything the graph carries between nodes."""

    # -- inputs, set once --
    question: str
    repo_path: str
    k: int
    # Where the indexes live. None means the configured defaults; the eval sets
    # these to score against a separately built index.
    persist_dir: str | None
    bm25_dir: str | None

    # -- retrieval --
    queries: list[str]          # every query searched, in order
    results: list[SearchResult]  # the most recent retrieval
    seen: list[SearchResult]     # union across attempts, deduplicated

    # -- generation --
    answer: str
    model: str
    input_tokens: int
    output_tokens: int

    # -- reflection --
    attempts: int               # completed generate+reflect cycles
    critique: Critique | None
    critiques: list[Critique]   # one per attempt, for the trace
    answers: list[str]          # the answer each attempt produced, same order.
    # Kept so that a loop that runs out of attempts can return its *least bad*
    # answer instead of its last one — observed live: a retry told to drop
    # unsupported claims hedged a correct answer into a worse one.

    # -- bookkeeping --
    trace: list[str]            # human-readable log of what the graph did


@dataclass
class AgentResult:
    """What the caller gets back: the answer plus the evidence and the history."""

    question: str
    answer: str
    results: list[SearchResult] = field(default_factory=list)
    critiques: list[Critique] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)
    attempts: int = 0
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def grounded(self) -> bool:
        """Whether the final answer passed reflection.

        False when the loop ran out of attempts, which the CLI surfaces rather
        than hides: an ungrounded answer the user is warned about is far safer
        than one presented as if it were checked.
        """
        return bool(self.critiques) and self.critiques[-1].grounded

    @property
    def unsupported_citations(self) -> list[str]:
        return list(self.critiques[-1].unsupported) if self.critiques else []
