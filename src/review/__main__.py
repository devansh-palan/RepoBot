"""Review a diff: `python -m src.review <repo> --diff change.patch`.

Reads the diff from a file, or from stdin with `--diff -`, so it composes with
git directly:

    git diff main... | python -m src.review . --diff -

Index the repo first with `python -m src.index <repo>`; the reviewer needs it to
find related code elsewhere in the tree.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.agent import LLMError, get_provider
from src.agent.llm import PROVIDERS

from .models import SEVERITIES
from .reviewer import review_diff


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m src.review", description=__doc__)
    parser.add_argument("repo", help="path to the indexed repository the diff applies to")
    parser.add_argument("--diff", "-d", required=True, help="path to a diff file, or - for stdin")
    parser.add_argument("--provider", choices=sorted(PROVIDERS), default=None)
    parser.add_argument("--max-hunks", type=int, default=None, help="stop after N hunks")
    parser.add_argument(
        "--min-severity",
        choices=SEVERITIES,
        default="note",
        help="hide anything less severe than this",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument(
        "--fail-on-blocker",
        action="store_true",
        help="exit 1 if any blocker is found, for use in CI",
    )
    args = parser.parse_args()

    text = sys.stdin.read() if args.diff == "-" else Path(args.diff).read_text(
        encoding="utf-8", errors="replace"
    )
    if not text.strip():
        print("error: empty diff", file=sys.stderr)
        raise SystemExit(2)

    try:
        result = review_diff(
            text, args.repo, provider=get_provider(args.provider), max_hunks=args.max_hunks
        )
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    keep = SEVERITIES[: SEVERITIES.index(args.min_severity) + 1]
    shown = [c for c in result.sorted() if c.severity in keep]

    if args.json:
        print(result.to_json())
    else:
        print(result.summary())
        if result.model:
            print(f"{result.model}  {result.input_tokens} in / {result.output_tokens} out")
        print()
        for comment in shown:
            print(comment)
        if not shown:
            print("no comments at or above severity "
                  f"{args.min_severity!r} — the diff looks clean")
        if result.skipped:
            print(f"\nskipped ({len(result.skipped)}):")
            for note in result.skipped[:10]:
                print(f"  {note}")

    if args.fail_on_blocker and result.blocking:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
