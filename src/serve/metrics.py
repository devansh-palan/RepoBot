"""Per-request timing.

Time to first token is the number that matters for a streaming endpoint. Total
latency hides the thing users actually feel: whether the page sat blank for
eight seconds before anything appeared. Retrieval, prompt assembly, and model
prefill all land before the first token, so TTFT is also the fastest way to see
*which* stage got slower.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class RequestTimer:
    """Wall-clock timings for one request, in milliseconds.

    Uses `perf_counter`, not `time()`: it is monotonic, so a clock adjustment
    mid-request cannot produce a negative latency.
    """

    started: float = field(default_factory=time.perf_counter)
    first_token_at: float | None = None
    finished_at: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0

    def mark_first_token(self) -> None:
        """Called on the first byte of content sent to the client.

        Idempotent, because the caller usually invokes it from inside a
        per-token callback and should not have to track whether it is the first.
        """
        if self.first_token_at is None:
            self.first_token_at = time.perf_counter()

    def finish(self) -> None:
        if self.finished_at is None:
            self.finished_at = time.perf_counter()

    @property
    def ttft_ms(self) -> float | None:
        """None when nothing was ever streamed — an error, or an empty answer."""
        if self.first_token_at is None:
            return None
        return round((self.first_token_at - self.started) * 1000, 1)

    @property
    def total_ms(self) -> float:
        end = self.finished_at if self.finished_at is not None else time.perf_counter()
        return round((end - self.started) * 1000, 1)

    @property
    def generation_ms(self) -> float | None:
        """Time spent streaming, i.e. total minus everything before the first token.

        Separating this from TTFT is what distinguishes "retrieval is slow" from
        "the model is slow", which are fixed in completely different places.
        """
        if self.first_token_at is None or self.finished_at is None:
            return None
        return round((self.finished_at - self.first_token_at) * 1000, 1)

    @property
    def tokens_per_second(self) -> float | None:
        """Output tokens per second of generation, once streaming started."""
        elapsed = self.generation_ms
        if not elapsed or not self.output_tokens:
            return None
        return round(self.output_tokens / (elapsed / 1000), 1)

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "ttft_ms": self.ttft_ms,
            "total_ms": self.total_ms,
            "generation_ms": self.generation_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "tokens_per_second": self.tokens_per_second,
        }
