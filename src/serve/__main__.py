"""Run the service: `python -m src.serve`.

Opens http://127.0.0.1:8000 — the UI is at / and the OpenAPI docs at /docs.
"""

from __future__ import annotations

import argparse

import uvicorn

from src import config


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m src.serve", description=__doc__)
    parser.add_argument("--host", default=config.API_HOST)
    parser.add_argument("--port", type=int, default=config.API_PORT)
    parser.add_argument("--reload", action="store_true", help="restart on code changes")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="worker processes; each loads its own embedding model, so raise with care",
    )
    args = parser.parse_args()

    print(f"UI      http://{args.host}:{args.port}/")
    print(f"docs    http://{args.host}:{args.port}/docs")
    uvicorn.run(
        "src.serve.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else None,
    )


if __name__ == "__main__":
    main()
