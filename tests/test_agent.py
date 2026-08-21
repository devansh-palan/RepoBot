"""Prompt construction, the provider wrapper, and plain RAG.

Every test here runs offline. The Anthropic provider is exercised only through
its contract and its construction; a live call belongs in the eval, not in a
unit test that should pass on a laptop with no key.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from src import config
from src.agent import (
    PROVIDERS,
    SYSTEM_PROMPT,
    Answer,
    EchoProvider,
    LLMError,
    LLMResponse,
    answer_question,
    build_user_prompt,
    format_context,
    get_provider,
)
from src.index import SearchResult, index_repo
from src.ingest import Chunk

SAMPLE = '''"""Billing helpers."""


def compound_interest(principal, rate, years):
    """Grow a principal at a fixed annual rate."""
    return principal * (1 + rate) ** years


class RetryingHttpClient:
    """Issues requests and retries with exponential backoff on failure."""

    def fetch(self, url):
        return url
'''


@pytest.fixture(scope="module")
def repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("agent_repo")
    (root / "billing.py").write_text(SAMPLE, encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def indexed(repo: Path, tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Index into throwaway dirs; yields the kwargs that point answer_question there."""
    base = tmp_path_factory.mktemp("agent_indexes")
    dirs = {"persist_dir": str(base / "chroma"), "bm25_dir": base / "bm25"}
    index_repo(repo, cache_dir=base / "cache", **dirs)
    return {"repo_path": repo, **dirs}


class RecordingProvider:
    """Captures the prompt it was handed and returns a scripted answer."""

    name = "recording"
    model = "recording-1"

    def __init__(self, reply: str = "It compounds annually. `billing.py:4-6`") -> None:
        self.reply = reply
        self.system: str | None = None
        self.user: str | None = None
        self.calls = 0

    def complete(
        self,
        system: str,
        user: str,
        model: str | None = None,
        max_tokens: int = config.LLM_MAX_TOKENS,
        on_text: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        self.system, self.user, self.calls = system, user, self.calls + 1
        if on_text is not None:
            on_text(self.reply)
        return LLMResponse(
            text=self.reply, model=model or self.model, input_tokens=11, output_tokens=7
        )


def _result(path: str, start: int, end: int, symbol: str = "f", rank: int = 1) -> SearchResult:
    chunk = Chunk(
        code=f"def {symbol}(): pass",
        language="Python",
        file_path=path,
        symbol=symbol,
        kind="function",
        start_line=start,
        end_line=end,
    )
    return SearchResult(chunk=chunk, score=1.0, rank=rank, contributions={"vector": rank})


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------


def test_system_prompt_states_the_rules_that_matter() -> None:
    """Grounding and citation are the whole point; a reworded prompt must keep them."""
    lowered = SYSTEM_PROMPT.lower()
    assert "only from the provided excerpts" in lowered
    assert "line range" in lowered
    assert "do not contain the answer" in lowered


def test_format_context_labels_each_excerpt_with_its_citation() -> None:
    """The model should be able to copy a citation, not construct one."""
    rendered = format_context([_result("src/a.py", 10, 42, "parse")])
    assert "src/a.py:10-42" in rendered
    assert "function parse" in rendered
    assert "Python" in rendered
    assert "def parse(): pass" in rendered


def test_format_context_caps_an_oversized_chunk() -> None:
    """Prefill is over half of time-to-first-token on a CPU-bound model; one
    8000-char module chunk must not double the prompt. The head survives —
    signature and docstring are what grounding needs — and the cut is marked
    so the model does not treat the truncation point as the end of the code."""
    big = _result("big.py", 1, 200)
    big.chunk.__dict__["code"] = "x = 1\n" * 2_000  # ~12000 chars, frozen dataclass

    rendered = format_context([big])
    assert len(rendered) < config.PROMPT_CHUNK_CHAR_CAP + 500
    assert "[truncated for length]" in rendered

    small = format_context([_result("small.py", 1, 2)])
    assert "[truncated for length]" not in small


def test_format_context_numbers_excerpts_in_rank_order() -> None:
    rendered = format_context(
        [_result("a.py", 1, 2, "a", rank=1), _result("b.py", 3, 4, "b", rank=2)]
    )
    assert rendered.index("[1]") < rendered.index("[2]")
    assert rendered.index("a.py:1-2") < rendered.index("b.py:3-4")


def test_user_prompt_puts_the_question_after_the_excerpts() -> None:
    """Volatile content last, so a stable prefix stays cacheable later."""
    prompt = build_user_prompt("how does it work?", [_result("a.py", 1, 2)])
    assert prompt.index("a.py:1-2") < prompt.index("how does it work?")
    assert prompt.rstrip().endswith("how does it work?")


# --------------------------------------------------------------------------
# Provider wrapper
# --------------------------------------------------------------------------


def test_known_providers_are_registered() -> None:
    assert set(PROVIDERS) == {"anthropic", "echo", "local"}


def test_get_provider_defaults_to_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "LLM_PROVIDER", "echo")
    assert get_provider().name == "echo"


def test_unknown_provider_names_the_known_ones() -> None:
    with pytest.raises(LLMError, match="anthropic"):
        get_provider("gpt")


def test_echo_provider_returns_the_prompt_and_streams_it() -> None:
    streamed: list[str] = []
    response = EchoProvider().complete("SYS", "USER", on_text=streamed.append)
    assert "SYS" in response.text and "USER" in response.text
    assert "".join(streamed) == response.text


def test_anthropic_provider_is_constructible_without_credentials() -> None:
    """Constructing must not touch the network; only .complete() may."""
    from src.agent.llm import AnthropicProvider

    assert AnthropicProvider().name == "anthropic"


# --------------------------------------------------------------------------
# Answer bookkeeping
# --------------------------------------------------------------------------


def test_answer_extracts_citations_in_order_without_duplicates() -> None:
    answer = Answer(
        question="q",
        text="See `src/a.py:1-5` and `src/b.py:10-12`, then `src/a.py:1-5` again.",
        results=[],
    )
    assert answer.cited == ["src/a.py:1-5", "src/b.py:10-12"]


def test_answer_flags_citations_the_model_was_never_shown() -> None:
    """The hallucination this whole design exists to catch."""
    answer = Answer(
        question="q",
        text="Handled in `src/real.py:1-5` and `src/invented.py:99-120`.",
        results=[_result("src/real.py", 1, 5)],
    )
    assert answer.unsupported_citations == ["src/invented.py:99-120"]


def test_answer_reports_retrieved_but_unused_chunks() -> None:
    answer = Answer(
        question="q",
        text="Only `src/a.py:1-5` matters.",
        results=[_result("src/a.py", 1, 5), _result("src/b.py", 7, 9)],
    )
    assert answer.uncited_sources == ["src/b.py:7-9"]
    assert answer.unsupported_citations == []


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------


def test_answer_question_grounds_the_prompt_in_retrieved_code(indexed: dict) -> None:
    provider = RecordingProvider()
    answer = answer_question(question="how is interest compounded?", k=4,
                             provider=provider, **indexed)

    assert provider.calls == 1
    assert provider.system == SYSTEM_PROMPT
    assert answer.results, "retrieval should find the sample code"
    # Every retrieved chunk must actually reach the model.
    for result in answer.results:
        assert result.chunk.location in provider.user
    assert "how is interest compounded?" in provider.user
    assert answer.text == provider.reply
    # qa must not force a model on the provider — each provider owns its own.
    assert answer.model == provider.model


def test_answer_question_streams_to_the_callback(indexed: dict) -> None:
    streamed: list[str] = []
    answer_question(
        question="retry logic", k=2, provider=RecordingProvider(),
        on_text=streamed.append, **indexed
    )
    assert streamed, "on_text should receive the answer"


def test_answer_question_respects_k(indexed: dict) -> None:
    answer = answer_question(question="http retry", k=2,
                             provider=RecordingProvider(), **indexed)
    assert len(answer.results) == 2


def test_no_retrieval_means_no_llm_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With no evidence there is no grounded answer, and asking invites a guess."""
    monkeypatch.setattr("src.agent.qa.hybrid_search", lambda *a, **kw: [])
    provider = RecordingProvider()

    answer = answer_question(tmp_path, "anything", provider=provider)
    assert provider.calls == 0
    assert answer.results == []
    assert "no code" in answer.text.lower()


def test_echo_provider_runs_the_whole_pipeline(indexed: dict) -> None:
    """The offline path a user gets with --provider echo."""
    answer = answer_question(question="how is interest compounded?", k=3,
                             provider=EchoProvider(), **indexed)
    assert answer.results
    assert "Question: how is interest compounded?" in answer.text
    assert answer.results[0].chunk.location in answer.text


def test_missing_credentials_become_an_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """The SDK raises TypeError while building headers, not AuthenticationError."""
    from src.agent.llm import AnthropicProvider

    class NoAuthClient:
        class messages:  # noqa: N801 - mirrors the SDK's attribute layout
            @staticmethod
            def stream(**kwargs):
                raise TypeError("Could not resolve authentication method. Expected one of ...")

    monkeypatch.setattr(AnthropicProvider, "_client", staticmethod(lambda: NoAuthClient()))

    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        AnthropicProvider().complete("sys", "user")


def test_unrelated_type_errors_still_propagate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Matching on the message must not turn every TypeError into a credentials hint."""
    from src.agent.llm import AnthropicProvider

    class BrokenClient:
        class messages:  # noqa: N801
            @staticmethod
            def stream(**kwargs):
                raise TypeError("stream() got an unexpected keyword argument 'nonsense'")

    monkeypatch.setattr(AnthropicProvider, "_client", staticmethod(lambda: BrokenClient()))

    with pytest.raises(TypeError, match="nonsense"):
        AnthropicProvider().complete("sys", "user")


# --------------------------------------------------------------------------
# Local provider
# --------------------------------------------------------------------------


def _sse(*events: str) -> list[str]:
    """Render OpenAI-style SSE lines, blank separators and all."""
    return [line for e in events for line in (f"data: {e}", "")]


class _FakeStream:
    """Stands in for httpx.Client, recording the request and replaying a stream."""

    def __init__(self, response, sent: dict) -> None:
        self._response = response
        self._sent = sent

    def __call__(self, **kwargs):  # httpx.Client(timeout=...)
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def stream(self, method, url, json):
        self._sent.update(method=method, url=url, payload=json)
        return self  # doubles as the response context manager

    # -- response side --
    @property
    def status_code(self):
        return self._response["status_code"]

    @property
    def text(self):
        return self._response.get("text", "")

    def read(self):
        return None

    def iter_lines(self):
        return iter(self._response.get("lines", ()))


def test_sse_decoder_reassembles_a_streamed_answer() -> None:
    from src.agent.llm import _iter_sse

    lines = _sse(
        '{"model":"qwen2.5-coder:3b","choices":[{"delta":{"content":"Hello "}}]}',
        '{"model":"qwen2.5-coder:3b","choices":[{"delta":{"content":"world"}}]}',
        '{"choices":[{"delta":{}}],"usage":{"prompt_tokens":12,"completion_tokens":3}}',
        "[DONE]",
    )
    decoded = list(_iter_sse(iter(lines)))

    assert "".join(text for text, _, _ in decoded) == "Hello world"
    assert [u for _, u, _ in decoded if u][-1]["prompt_tokens"] == 12


def test_sse_decoder_tolerates_runtime_quirks() -> None:
    """Keep-alives, comments, and malformed lines differ between local runtimes."""
    from src.agent.llm import _iter_sse

    lines = [
        "",
        ": keep-alive",
        "data: {not json}",
        'data: {"choices":[{"delta":{"content":"ok"}}]}',
        "data: [DONE]",
        'data: {"choices":[{"delta":{"content":"after done"}}]}',
    ]
    assert "".join(t for t, _, _ in _iter_sse(iter(lines))) == "ok"


def test_local_provider_posts_a_chat_completion_and_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    from src.agent.llm import LocalProvider

    sent: dict = {}
    fake = _FakeStream(
        {
            "status_code": 200,
            "lines": _sse(
                '{"model":"qwen2.5-coder:3b","choices":[{"delta":{"content":"See "}}]}',
                '{"choices":[{"delta":{"content":"`a.py:1-2`"}}],'
                '"usage":{"prompt_tokens":40,"completion_tokens":5}}',
                "[DONE]",
            ),
        },
        sent,
    )
    monkeypatch.setattr(httpx, "Client", fake)

    streamed: list[str] = []
    response = LocalProvider().complete("SYS", "USER", on_text=streamed.append)

    assert sent["method"] == "POST"
    assert sent["url"].endswith("/chat/completions")
    assert sent["payload"]["stream"] is True
    assert sent["payload"]["model"] == config.LOCAL_MODEL
    assert sent["payload"]["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USER"},
    ]
    assert response.text == "See `a.py:1-2`"
    assert response.model == "qwen2.5-coder:3b"
    assert (response.input_tokens, response.output_tokens) == (40, 5)
    assert "".join(streamed) == "See `a.py:1-2`"


def test_local_provider_explains_a_dead_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """The most common failure by far: the user has not started the server."""
    import httpx

    from src.agent.llm import LocalProvider

    class Refusing(_FakeStream):
        def stream(self, *a, **kw):
            raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "Client", Refusing({"status_code": 200}, {}))

    with pytest.raises(LLMError, match="ollama pull"):
        LocalProvider().complete("SYS", "USER")


def test_local_provider_surfaces_a_server_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pulled-but-missing model is a 404, not a connection failure."""
    import httpx

    from src.agent.llm import LocalProvider

    monkeypatch.setattr(
        httpx,
        "Client",
        _FakeStream({"status_code": 404, "text": 'model "x" not found, try pulling it'}, {}),
    )

    with pytest.raises(LLMError, match="404"):
        LocalProvider().complete("SYS", "USER")


def test_local_provider_reports_a_timeout_as_a_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    from src.agent.llm import LocalProvider

    class Slow(_FakeStream):
        def stream(self, *a, **kw):
            raise httpx.ReadTimeout("too slow")

    monkeypatch.setattr(httpx, "Client", Slow({"status_code": 200}, {}))

    with pytest.raises(LLMError, match="timed out"):
        LocalProvider().complete("SYS", "USER")


def test_local_provider_uses_the_configured_model_by_default() -> None:
    from src.agent.llm import LocalProvider

    assert LocalProvider().model == config.LOCAL_MODEL
    assert LocalProvider(model="llama3.2:3b").model == "llama3.2:3b"
    assert LocalProvider(base_url="http://host:1234/v1/").base_url == "http://host:1234/v1"
