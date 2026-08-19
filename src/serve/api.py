"""FastAPI service: /ask (streaming) and /review.

Everything underneath — retrieval, the model call, the graph — is synchronous
and CPU- or IO-blocking. Calling it directly from an async endpoint would block
the event loop and serialise every concurrent request behind one model call, so
all of it runs in a worker thread via `asyncio.to_thread`.

Streaming needs a second piece. The provider hands tokens to a *synchronous*
callback from inside that worker thread, so tokens are pushed onto an
`asyncio.Queue` with `call_soon_threadsafe` and the endpoint drains the queue.
That is the whole bridge between the blocking world and the event loop.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from src import config
from src.agent import LLMError, answer_question, get_provider, run_agent
from src.index import SearchResult, bm25_path_for, collection_name, index_repo
from src.ingest import CloneError, clone_github_repo, parse_github_ref
from src.retrieve import MissingIndexError
from src.review import review_diff

from .metrics import RequestTimer
from .schemas import (
    AskRequest,
    AskResult,
    CommentOut,
    HealthResponse,
    IndexRequest,
    MetricsOut,
    RepoOut,
    ReposResponse,
    ReviewRequest,
    ReviewResponse,
    SourceOut,
)

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="Codebase Q&A + PR Review Agent",
    version="0.1.0",
    description="Grounded answers and diff review over an indexed repository.",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sse(event: str, data: Any) -> str:
    """One Server-Sent Event.

    The blank line terminates the event and is not optional — without it the
    client buffers forever waiting for the end of the frame.
    """
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _sources(results: list[SearchResult]) -> list[SourceOut]:
    return [
        SourceOut(
            location=r.chunk.location,
            file=r.chunk.file_path,
            symbol=r.chunk.symbol,
            kind=r.chunk.kind,
            language=r.chunk.language,
            start_line=r.chunk.start_line,
            end_line=r.chunk.end_line,
            score=round(r.score, 4),
            rank=r.rank,
        )
        for r in results
    ]


def _metrics(timer: RequestTimer) -> MetricsOut:
    return MetricsOut(**timer.as_dict())


def _translate(exc: Exception) -> HTTPException:
    """Turn a domain error into a status code a client can act on."""
    if isinstance(exc, MissingIndexError):
        return HTTPException(404, f"repository is not indexed: {exc}")
    if isinstance(exc, LLMError):
        # 503: the model is unreachable or unconfigured. The request was fine.
        return HTTPException(503, str(exc))
    if isinstance(exc, (FileNotFoundError, NotADirectoryError)):
        return HTTPException(404, str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(400, str(exc))
    return HTTPException(500, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness plus which model is actually configured.

    Reports the configured provider without calling it: a health check that
    depends on a downstream model reports the model's health, not ours.
    """
    provider = get_provider()
    return HealthResponse(
        status="ok",
        provider=provider.name,
        model=getattr(provider, "model", ""),
        embedding_model=config.EMBEDDING_MODEL,
    )


@app.post("/ask")
async def ask(request: AskRequest) -> StreamingResponse:
    """Answer a question, streaming tokens over SSE.

    Events, in order:
      `meta`    — echoes the request; lets a client render a header immediately
      `token`   — a fragment of the answer (plain mode: as generated)
      `stage`   — agent mode only: retrieve / generate / reflect progress
      `sources` — the retrieved chunks, once known
      `done`    — the full AskResult, including metrics
      `error`   — something failed; the stream ends after this
    """
    return StreamingResponse(
        _ask_events(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Without this, nginx buffers the whole response and streaming
            # silently becomes a single delayed blob.
            "X-Accel-Buffering": "no",
        },
    )


async def _ask_events(request: AskRequest) -> AsyncIterator[str]:
    timer = RequestTimer()
    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def emit(kind: str, payload: Any) -> None:
        """Called from the worker thread; hands the event to the event loop."""
        loop.call_soon_threadsafe(queue.put_nowait, (kind, payload))

    def work() -> None:
        try:
            provider = get_provider(request.provider)
            if request.mode == "agent":
                result = _run_agent_blocking(request, provider, emit)
            else:
                result = _run_plain_blocking(request, provider, emit)
            emit("result", result)
        except Exception as exc:  # noqa: BLE001 - surfaced to the client as an event
            emit("failed", exc)

    yield _sse("meta", {"question": request.question, "mode": request.mode, "k": request.k})

    task = asyncio.create_task(asyncio.to_thread(work))
    try:
        while True:
            kind, payload = await queue.get()

            if kind == "token":
                timer.mark_first_token()
                yield _sse("token", {"text": payload})
            elif kind == "stage":
                # A stage event is content too: it is the first thing the user
                # sees in agent mode, so it starts the TTFT clock.
                timer.mark_first_token()
                yield _sse("stage", payload)
            elif kind == "sources":
                yield _sse("sources", [s.model_dump() for s in _sources(payload)])
            elif kind == "failed":
                timer.finish()
                error = _translate(payload)
                yield _sse("error", {"status": error.status_code, "detail": error.detail})
                return
            elif kind == "result":
                timer.input_tokens = payload.metrics.input_tokens
                timer.output_tokens = payload.metrics.output_tokens
                timer.finish()
                payload.metrics = _metrics(timer)
                yield _sse("done", payload.model_dump())
                return
    finally:
        await task


def _run_plain_blocking(request: AskRequest, provider, emit) -> AskResult:
    """Single-shot RAG. Tokens stream as the model produces them."""
    answer = answer_question(
        request.repo,
        request.question,
        k=request.k,
        provider=provider,
        on_text=lambda text: emit("token", text),
    )
    emit("sources", answer.results)
    return AskResult(
        question=answer.question,
        answer=answer.text,
        mode="plain",
        model=answer.model,
        grounded=None,  # plain RAG does not check; null is honest, False is not
        # Repeated in `done` as well as the `sources` event, so a client that
        # only reads the final payload still gets them.
        sources=_sources(answer.results),
        unsupported_citations=answer.unsupported_citations,
        metrics=MetricsOut(
            input_tokens=answer.input_tokens, output_tokens=answer.output_tokens
        ),
    )


def _run_agent_blocking(request: AskRequest, provider, emit) -> AskResult:
    """The reflection loop.

    Streams stage events rather than tokens: the graph may reject a draft and
    answer again, and streaming text that is about to be discarded would show
    the user a wrong answer and then silently replace it.
    """
    emit("stage", {"stage": "retrieve", "detail": "searching the repository"})

    result = run_agent(request.repo, request.question, k=request.k, provider=provider)

    for line in result.trace:
        emit("stage", {"stage": "trace", "detail": line})
    emit("sources", result.results)
    emit("token", result.answer)

    return AskResult(
        question=result.question,
        answer=result.answer,
        mode="agent",
        model=result.model,
        grounded=result.grounded,
        attempts=result.attempts,
        queries=result.queries,
        sources=_sources(result.results),
        unsupported_citations=result.unsupported_citations,
        metrics=MetricsOut(
            input_tokens=result.input_tokens, output_tokens=result.output_tokens
        ),
    )


@app.post("/index")
async def index_endpoint(request: IndexRequest) -> StreamingResponse:
    """Clone (if needed) and index a GitHub repository, streaming progress over SSE.

    `repo` is a GitHub reference — `psf/requests` or a full https URL. Nothing
    else: cloning is restricted to GitHub over https and the ref is validated
    before it reaches git, because the input comes from a web form and must
    never be able to smuggle a git option, another protocol, or a local path.

    Events: `stage` lines as work progresses, then `done` with the path to use
    as AskRequest.repo, or `error`.
    """
    return StreamingResponse(
        _index_events(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _index_events(request: IndexRequest) -> AsyncIterator[str]:
    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def emit(kind: str, payload: Any) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, (kind, payload))

    def work() -> None:
        try:
            owner, repo = parse_github_ref(request.repo)
            repo_path = clone_github_repo(
                request.repo, on_progress=lambda line: emit("stage", line)
            )
            name = f"{owner}/{repo}"

            stats = index_repo(repo_path, on_progress=lambda line: emit("stage", line))
            emit(
                "done",
                {
                    "name": name,
                    "path": str(repo_path),
                    "chunks": stats.chunks,
                    "embedded": stats.embedded,
                    "cached": stats.cached,
                },
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the client as an event
            emit("failed", exc)

    task = asyncio.create_task(asyncio.to_thread(work))
    try:
        while True:
            kind, payload = await queue.get()
            if kind == "stage":
                yield _sse("stage", {"detail": payload})
            elif kind == "failed":
                error = _translate_index(payload)
                yield _sse("error", {"status": error.status_code, "detail": error.detail})
                return
            elif kind == "done":
                yield _sse("done", payload)
                return
    finally:
        await task


def _translate_index(exc: Exception) -> HTTPException:
    if isinstance(exc, CloneError):
        return HTTPException(400, str(exc))
    return _translate(exc)


@app.get("/repos", response_model=ReposResponse)
async def repos() -> ReposResponse:
    """Every repository in the store, and whether it is ready to ask about.

    Lets the UI offer already-indexed repos without the user retyping paths,
    and lets it show "cloned but not indexed" honestly rather than failing the
    first question with a 404.
    """
    found: list[RepoOut] = []
    if config.REPOS_DIR.is_dir():
        for path in sorted(config.REPOS_DIR.iterdir()):
            if not path.is_dir():
                continue
            found.append(
                RepoOut(
                    name=path.name.replace("__", "/", 1),
                    path=str(path),
                    indexed=bm25_path_for(collection_name(path.resolve())).is_file(),
                )
            )
    return ReposResponse(repos=found)


@app.post("/review", response_model=ReviewResponse)
async def review(request: ReviewRequest) -> ReviewResponse:
    """Review a unified diff and return structured comments.

    Not streamed: a review is a set of records rather than prose, and half a
    JSON array is not useful to anyone.
    """
    timer = RequestTimer()
    try:
        result = await asyncio.to_thread(
            review_diff,
            request.diff,
            request.repo,
            get_provider(request.provider),
            request.max_hunks,
        )
    except Exception as exc:  # noqa: BLE001
        raise _translate(exc) from exc

    timer.input_tokens = result.input_tokens
    timer.output_tokens = result.output_tokens
    timer.mark_first_token()
    timer.finish()

    return ReviewResponse(
        comments=[CommentOut(**c.to_dict()) for c in result.sorted()],
        files_reviewed=result.files_reviewed,
        hunks_reviewed=result.hunks_reviewed,
        by_severity={k: v for k, v in result.counts().items() if v},
        skipped=result.skipped,
        model=result.model,
        metrics=_metrics(timer),
    )


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    """The single-page UI. Served from here so there is nothing else to run."""
    page = STATIC_DIR / "index.html"
    if not page.is_file():
        raise HTTPException(404, "frontend not installed")
    return FileResponse(page)
