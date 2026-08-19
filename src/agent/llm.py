"""A thin swappable wrapper around the LLM.

Everything above this module talks to `Provider.complete()` and never imports a
vendor SDK directly. That buys three things: the eval can swap a local model for
a hosted one without touching the agent, tests run offline, and adding a vendor
later means adding a class here rather than editing call sites.

Each provider owns its own model name. `complete()` takes an optional `model`
override for the cases that genuinely need one — the reflection critic runs on a
cheaper model than the answerer — but callers should normally leave it alone.

Deliberately not a framework. One method, one response object, no chat history;
the reflection loop manages its own state.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol

from src import config


class LLMError(RuntimeError):
    """A provider could not produce an answer."""


_NO_CREDENTIALS = (
    "no Anthropic credentials found. Set ANTHROPIC_API_KEY, or run `ant auth login`. "
    "To run against a local model instead, use --provider local; to inspect the "
    "prompt without any model, use --provider echo."
)


@dataclass(frozen=True)
class LLMResponse:
    """One completion, plus what it cost."""

    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0

    def cost_line(self) -> str:
        return f"{self.model}  {self.input_tokens} in / {self.output_tokens} out"


class Provider(Protocol):
    """What the rest of the codebase is allowed to assume about an LLM."""

    name: str
    model: str

    def complete(
        self,
        system: str,
        user: str,
        model: str | None = None,
        max_tokens: int = config.LLM_MAX_TOKENS,
        on_text: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        """Answer one prompt. `on_text` receives text as it streams, if given."""
        ...


# --------------------------------------------------------------------------
# Local model
# --------------------------------------------------------------------------


class LocalProvider:
    """Any OpenAI-compatible server running on this machine.

    Ollama, LM Studio, llama.cpp's server, and vLLM all expose the same
    `/v1/chat/completions` shape, so one class covers whichever the user
    installs — only `LOCAL_BASE_URL` changes.

    Note this is *not* a shim for calling Claude: AnthropicProvider below is the
    Claude path and uses the official SDK. This talks to a local Qwen/Llama, and
    the OpenAI wire format is simply what every local runtime happens to speak.
    """

    name = "local"

    def __init__(
        self,
        model: str = config.LOCAL_MODEL,
        base_url: str = config.LOCAL_BASE_URL,
        timeout: float = config.LOCAL_TIMEOUT_SECONDS,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # A local server needs no auth, but the same wire format is what hosted
        # providers of open models speak — so one key turns this into a cheap
        # hosted path without a second provider class.
        self.api_key = api_key or os.environ.get(config.LOCAL_API_KEY_ENV, "")

    def complete(
        self,
        system: str,
        user: str,
        model: str | None = None,
        max_tokens: int = config.LLM_MAX_TOKENS,
        on_text: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        """Stream a completion from the local server.

        Streams even when no callback is given: a local model can take minutes
        on a long answer, and a streaming request keeps the connection alive
        rather than sitting on a single silent POST.
        """
        import httpx

        payload = {
            "model": model or self.model,
            "max_tokens": max_tokens,
            "stream": True,
            # Local runtimes vary in whether they report usage on streamed
            # responses; asking for it costs nothing where unsupported.
            "stream_options": {"include_usage": True},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

        chunks: list[str] = []
        usage: dict[str, Any] = {}
        served_by = model or self.model

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

        try:
            with httpx.Client(timeout=self.timeout, headers=headers) as client:
                with client.stream(
                    "POST", f"{self.base_url}/chat/completions", json=payload
                ) as response:
                    if response.status_code != 200:
                        response.read()
                        raise LLMError(
                            f"local server at {self.base_url} returned "
                            f"{response.status_code}: {response.text[:300]}"
                        )
                    for text, chunk_usage, chunk_model in _iter_sse(response.iter_lines()):
                        if text:
                            chunks.append(text)
                            if on_text is not None:
                                on_text(text)
                        if chunk_usage:
                            usage = chunk_usage
                        if chunk_model:
                            served_by = chunk_model
        except httpx.ConnectError as exc:
            raise LLMError(
                f"no local model server at {self.base_url}. Start one — e.g. "
                f"`ollama serve` after `ollama pull {self.model}` — or point "
                f"LOCAL_BASE_URL at the one you use."
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMError(
                f"local model timed out after {self.timeout}s. A smaller model or "
                f"a lower LLM_MAX_TOKENS would help."
            ) from exc

        return LLMResponse(
            text="".join(chunks).strip(),
            model=served_by,
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
        )


def _iter_sse(lines: Iterator[str]) -> Iterator[tuple[str, dict[str, Any], str]]:
    """Decode an OpenAI-style SSE stream into (text, usage, model) triples.

    Tolerant on purpose: local runtimes differ in whether they send a final
    usage-only chunk, keep-alive comments, or a trailing `[DONE]`. A malformed
    line is skipped rather than aborting a half-finished answer.
    """
    for line in lines:
        line = line.strip()
        if not line or not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if data == "[DONE]":
            return
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue

        choices = event.get("choices") or [{}]
        text = (choices[0].get("delta") or {}).get("content") or ""
        yield text, event.get("usage") or {}, event.get("model") or ""


# --------------------------------------------------------------------------
# Anthropic
# --------------------------------------------------------------------------


class AnthropicProvider:
    """The hosted path: one Messages API call per question."""

    name = "anthropic"

    def __init__(
        self,
        model: str = config.ANSWER_MODEL,
        effort: str = config.LLM_EFFORT,
    ) -> None:
        self.model = model
        self.effort = effort

    @staticmethod
    @lru_cache(maxsize=1)
    def _client():
        """Built once per process. Imported lazily so `import src.agent` is cheap."""
        import anthropic

        # The SDK resolves credentials itself: ANTHROPIC_API_KEY, then
        # ANTHROPIC_AUTH_TOKEN, then an `ant auth login` profile on disk. Passing
        # a key explicitly would break the profile path for no benefit.
        return anthropic.Anthropic()

    def complete(
        self,
        system: str,
        user: str,
        model: str | None = None,
        max_tokens: int = config.LLM_MAX_TOKENS,
        on_text: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        """Call Claude and return the answer text.

        Streams rather than blocking: max_tokens is 16k and a code answer can
        run long, which is exactly the case where a non-streaming request risks
        an HTTP timeout. `get_final_message` reassembles the whole response, so
        callers that do not care about tokens-as-they-arrive see no difference.
        """
        import anthropic

        try:
            with self._client().messages.stream(
                model=model or self.model,
                max_tokens=max_tokens,
                system=system,
                # Adaptive thinking: the model decides its own depth. On Opus 5
                # this is the default, stated explicitly so a model swap does
                # not silently change behaviour. `budget_tokens` is rejected.
                thinking={"type": "adaptive"},
                output_config={"effort": self.effort},
                messages=[{"role": "user", "content": user}],
            ) as stream:
                if on_text is not None:
                    for delta in stream.text_stream:
                        on_text(delta)
                message = stream.get_final_message()
        except anthropic.AuthenticationError as exc:
            raise LLMError(_NO_CREDENTIALS) from exc
        except TypeError as exc:
            # With no resolvable credentials at all the SDK raises TypeError
            # while building headers, before any AuthenticationError can exist.
            # Matched on the message so a genuine TypeError still propagates.
            if "authentication" not in str(exc).lower():
                raise
            raise LLMError(_NO_CREDENTIALS) from exc
        except anthropic.RateLimitError as exc:
            raise LLMError(f"rate limited by the Anthropic API: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"could not reach the Anthropic API: {exc}") from exc

        if message.stop_reason == "refusal":
            detail = getattr(message.stop_details, "explanation", None) or "no explanation given"
            raise LLMError(f"the model declined to answer: {detail}")

        # content is a list of blocks; thinking blocks are in there too.
        text = "".join(block.text for block in message.content if block.type == "text")
        return LLMResponse(
            text=text.strip(),
            model=message.model,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
        )


# --------------------------------------------------------------------------
# Echo
# --------------------------------------------------------------------------


class EchoProvider:
    """An offline stand-in that returns the prompt instead of an answer.

    Not a mock in the test-double sense — it is a real provider that makes the
    whole pipeline runnable and inspectable with no model at all. `--provider
    echo` prints the exact prompt the model would have seen, which is the
    fastest way to debug prompt or retrieval problems.
    """

    name = "echo"
    model = "echo"

    def complete(
        self,
        system: str,
        user: str,
        model: str | None = None,
        max_tokens: int = config.LLM_MAX_TOKENS,
        on_text: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        text = (
            f"[echo provider — no model was called]\n\n"
            f"--- system ---\n{system}\n\n--- user ---\n{user}"
        )
        if on_text is not None:
            on_text(text)
        return LLMResponse(text=text, model=self.model)


PROVIDERS: dict[str, Callable[[], Provider]] = {
    "local": LocalProvider,
    "anthropic": AnthropicProvider,
    "echo": EchoProvider,
}


def get_provider(name: str | None = None) -> Provider:
    """Build a provider by name, defaulting to config.LLM_PROVIDER."""
    name = name or config.LLM_PROVIDER
    if name not in PROVIDERS:
        raise LLMError(f"unknown provider {name!r}; known: {sorted(PROVIDERS)}")
    return PROVIDERS[name]()


def has_credentials() -> bool:
    """Whether an Anthropic call could plausibly succeed.

    Only checks the environment — an `ant auth login` profile on disk would not
    show up here, so this is used to soften an error message, never to block a
    call the SDK might well have been able to make.
    """
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
