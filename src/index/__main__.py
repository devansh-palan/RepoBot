"""CLI for the index stage: `python -m src.index <repo> --query "..."`.

The interface to this project until the FastAPI service exists.
"""

from __future__ import annotations

import argparse
import textwrap

from src import config

from .pipeline import index_repo
from .store import get_collection, search


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m src.index", description=__doc__)
    parser.add_argument("repo", help="path to the repository to index")
    parser.add_argument("--query", "-q", help="question to run against the index")
    parser.add_argument("-k", type=int, default=config.FINAL_TOP_K, help="results to show")
    parser.add_argument("--show-code", action="store_true", help="print the retrieved code")
    parser.add_argument(
        "--query-only",
        action="store_true",
        help="skip indexing and query the existing collection",
    )
    args = parser.parse_args()

    if not args.query_only:
        print(index_repo(args.repo))

    if not args.query:
        return

    print(f"\nquery: {args.query}\n")
    for result in search(get_collection(args.repo), args.query, args.k):
        print(result)
        if args.show_code:
            print(textwrap.indent(result.chunk.code, "    "), end="\n\n")


if __name__ == "__main__":
    main()
