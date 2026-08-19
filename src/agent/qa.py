"""Plain RAG: retrieve, prompt, answer.

Retrieve-then-generate with no loop and no tools. This is the baseline the
reflection agent has to beat, so it stays deliberately simple and separately
callable — the ablation needs to run it on its own.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from src import config
from src.index import SearchResult
from src.retrieve import hybrid_search

from .llm import LLMResponse, Provider, get_provider
from .prompts import NO_CONTEXT_ANSWER, SYSTEM_PROMPT, build_user_prompt

# `src/ingest/walker.py:13-38` — the citation format the system prompt asks for.
_CITATION = re.compile(r"[\w./\-]+\.\w+:\d+-\d+")


@dataclass(frozen=True)
class Answer:
    """An answer plus everything needed to check it."""

    question: str
    text: str
    results: list[SearchResult] = field(default_factory=list)
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def cited(self) -> list[str]:
        """Every `path:start-end` the answer claims, in order of first mention."""
        seen: dict[str, None] = {}
        for match in _CITATION.findall(self.text):
            seen.setdefault(match, None)
        return list(seen)

    @property
    def retrieved(self) -> list[str]:
        """The locations that were actually put in front of the model."""
        return [r.chunk.location for r in self.results]

    @property
    def uncited_sources(self) -> list[str]:
        """Retrieved chunks the answer never used — retrieval noise, roughly."""
        return [loc for loc in self.retrieved if loc not in set(self.cited)]

    @property
    def unsupported_citations(self) -> list[str]:
        """Citations pointing at code the model was never shown.

        Non-empty means the model invented a location, which is the failure
        mode grounded Q&A exists to prevent. The eval scores this directly.
        """
        return [loc for loc in self.cited if loc not in set(self.retrieved)]


def answer_question(
    repo_path: str | Path,
    question: str,
    k: int = config.FINAL_TOP_K,
    provider: Provider | None = None,
    on_text: Callable[[str], None] | None = None,
    persist_dir: str | None = None,
    bm25_dir: str | Path | None = None,
) -> Answer:
    """Answer a question about an indexed repository from retrieved code alone.

    `persist_dir` / `bm25_dir` override where the indexes are read from, the
    same way index_repo overrides where they are written. The ablation needs it
    to point at separately built indexes without disturbing the real ones.

    Returns early without calling the LLM when retrieval finds nothing: there is
    no grounded answer to give, and asking anyway invites the model to fall back
    on general knowledge, which is the one thing the system prompt forbids.
    """
    results = hybrid_search(repo_path, question, k, persist_dir, bm25_dir)
    if not results:
        return Answer(question=question, text=NO_CONTEXT_ANSWER)

    provider = provider or get_provider()
    # No model argument: each provider owns its own default, so switching to a
    # local model is a provider swap rather than an edit here.
    response: LLMResponse = provider.complete(
        system=SYSTEM_PROMPT,
        user=build_user_prompt(question, results),
        on_text=on_text,
    )

    return Answer(
        question=question,
        text=response.text,
        results=results,
        model=response.model,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
    )
