"""Ask questions against an indexed repo: `python -m src.agent <repo> -q "..."`.

Runs the reflection agent by default. `--plain` runs single-shot RAG instead,
which is the baseline the ablation compares against.

Index first with `python -m src.index <repo>`.
"""

from __future__ import annotations

import argparse
import sys
import textwrap

from src import config

from .graph import MAX_ATTEMPTS, run_agent
from .llm import PROVIDERS, LLMError, get_provider, has_credentials
from .qa import answer_question


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m src.agent", description=__doc__)
    parser.add_argument("repo", help="path to an already-indexed repository")
    parser.add_argument("--query", "-q", required=True, help="the question")
    parser.add_argument("-k", type=int, default=config.FINAL_TOP_K, help="chunks per retrieval")
    parser.add_argument(
        "--provider",
        choices=sorted(PROVIDERS),
        default=config.LLM_PROVIDER,
        help="'echo' prints the prompt instead of calling a model",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help=f"single-shot RAG, no reflection loop (the agent retries up to {MAX_ATTEMPTS}x)",
    )
    parser.add_argument("--sources", action="store_true", help="list the retrieved chunks")
    parser.add_argument("--trace", action="store_true", help="show what the graph did")
    args = parser.parse_args()

    if args.provider == "anthropic" and not has_credentials():
        print("note: ANTHROPIC_API_KEY is not set.\n", file=sys.stderr)

    provider = get_provider(args.provider)

    try:
        if args.plain:
            _plain(args, provider)
        else:
            _agent(args, provider)
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def _plain(args, provider) -> None:
    answer = answer_question(
        args.repo,
        args.query,
        k=args.k,
        provider=provider,
        on_text=lambda text: print(text, end="", flush=True),
    )
    print()
    if args.sources:
        _sources(answer.results, set(answer.cited))
    print(f"\n[plain RAG] {answer.model}  {len(answer.results)} chunks")
    if answer.unsupported_citations:
        print(f"WARNING ungrounded citations: {answer.unsupported_citations}")


def _agent(args, provider) -> None:
    # Not streamed: the graph may discard an answer and generate again, and
    # streaming a draft that is about to be thrown away would be worse than a
    # short wait.
    result = run_agent(args.repo, args.query, k=args.k, provider=provider)

    print(result.answer)

    if args.trace:
        print("\ntrace:")
        for line in result.trace:
            print(f"  {textwrap.shorten(line, 110)}")

    if args.sources:
        cited = {c for c in result.critiques[-1].unsupported} if result.critiques else set()
        _sources(result.results, {r.chunk.location for r in result.results} - cited)

    verdict = "grounded" if result.grounded else "NOT grounded"
    print(
        f"\n[agent] {result.model}  {result.attempts}/{MAX_ATTEMPTS} attempts  "
        f"{len(result.results)} chunks  {result.input_tokens} in / {result.output_tokens} out"
    )
    print(f"        reflection: {verdict}")
    if not result.grounded and result.critiques:
        print(f"        {result.critiques[-1].reason}")
    if len(result.queries) > 1:
        print(f"        queries: {result.queries}")


def _sources(results, highlighted: set[str]) -> None:
    print("\nretrieved:")
    for result in results:
        mark = "*" if result.chunk.location in highlighted else " "
        print(f" {mark} {result.rank:>2}. {result.chunk.header()}")


if __name__ == "__main__":
    main()
