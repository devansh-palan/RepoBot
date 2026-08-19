"""Request and response schemas.

Named `schemas` rather than `models` so it is never confused with the dataclass
models in src/index and src/ingest — these are the wire format, not the domain.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src import config


class AskRequest(BaseModel):
    repo: str = Field(description="path to an indexed repository")
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=config.FINAL_TOP_K, ge=1, le=50)
    provider: str | None = Field(default=None, description="local | anthropic | echo")
    mode: Literal["plain", "agent"] = Field(
        default="plain",
        description=(
            "plain streams the model's tokens as they arrive. agent runs the "
            "reflection loop, which may discard a draft and answer again, so it "
            "streams stage events and then the final answer."
        ),
    )


class SourceOut(BaseModel):
    location: str
    file: str
    symbol: str
    kind: str
    language: str
    start_line: int
    end_line: int
    score: float
    rank: int


class MetricsOut(BaseModel):
    ttft_ms: float | None = None
    total_ms: float = 0.0
    generation_ms: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    tokens_per_second: float | None = None


class AskResult(BaseModel):
    """The `done` event payload, and the body of the non-streaming variant."""

    question: str
    answer: str
    mode: str
    model: str = ""
    grounded: bool | None = Field(
        default=None,
        description="agent mode only; null in plain mode, which does not check",
    )
    attempts: int = 0
    queries: list[str] = Field(default_factory=list)
    sources: list[SourceOut] = Field(default_factory=list)
    unsupported_citations: list[str] = Field(default_factory=list)
    metrics: MetricsOut = Field(default_factory=MetricsOut)


class ReviewRequest(BaseModel):
    repo: str
    diff: str = Field(min_length=1, description="a unified diff")
    max_hunks: int | None = Field(default=None, ge=1, le=200)
    provider: str | None = None


class CommentOut(BaseModel):
    file: str
    line: int
    severity: str
    message: str


class ReviewResponse(BaseModel):
    comments: list[CommentOut] = Field(default_factory=list)
    files_reviewed: int = 0
    hunks_reviewed: int = 0
    by_severity: dict[str, int] = Field(default_factory=dict)
    skipped: list[str] = Field(default_factory=list)
    model: str = ""
    metrics: MetricsOut = Field(default_factory=MetricsOut)


class HealthResponse(BaseModel):
    status: str
    provider: str
    model: str
    embedding_model: str


class IndexRequest(BaseModel):
    """What to index: a GitHub URL or owner/repo shorthand. Nothing else."""

    repo: str = Field(min_length=1, max_length=500)


class RepoOut(BaseModel):
    """One repository the service can answer questions about."""

    name: str          # display name, e.g. "psf/requests"
    path: str          # what to send as AskRequest.repo
    indexed: bool      # False when cloned but not yet (fully) indexed


class ReposResponse(BaseModel):
    repos: list[RepoOut]
