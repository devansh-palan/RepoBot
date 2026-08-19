"""The FastAPI service: SSE framing, metrics, and error translation.

The provider is scripted, so streaming behaviour is pinned exactly rather than
depending on a model being up. What is actually under test is the bridge between
the blocking pipeline and the event loop — the part that would silently
serialise every request or deadlock if it were wrong.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.agent import LLMError, LLMResponse
from src.index import index_repo
from src.serve import api, app
from src.serve.metrics import RequestTimer

SAMPLE = '''"""Billing."""


def compound_interest(principal, rate, years):
    """Grow a principal at a fixed annual rate."""
    return principal * (1 + rate) ** years
'''

DIFF = """\
diff --git a/billing.py b/billing.py
--- a/billing.py
+++ b/billing.py
@@ -5,2 +5,2 @@ def compound_interest(principal, rate, years):
     \"\"\"Grow a principal at a fixed annual rate.\"\"\"
-    return principal * (1 + rate) ** years
+    return principal * (1 + rate) * years
"""


@pytest.fixture(scope="module")
def repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("serve_repo")
    (root / "billing.py").write_text(SAMPLE, encoding="utf-8")
    base = tmp_path_factory.mktemp("serve_idx")
    # The service reads the default index locations, so this repo is indexed
    # there. Isolated by its path hash, so it cannot collide with a real one.
    index_repo(root, cache_dir=base / "cache")
    return root


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


class ScriptedProvider:
    """Streams fixed chunks so token framing and TTFT can be asserted."""

    name = model = "scripted"

    def __init__(self, *chunks: str, delay: float = 0.0, fail: Exception | None = None) -> None:
        self.chunks = list(chunks) or ["Interest compounds. `billing.py:4-6`"]
        self.delay = delay
        self.fail = fail
        self.calls = 0

    def complete(self, system, user, model=None, max_tokens=0, on_text=None) -> LLMResponse:
        self.calls += 1
        if self.fail is not None:
            raise self.fail
        for chunk in self.chunks:
            if self.delay:
                time.sleep(self.delay)
            if on_text is not None:
                on_text(chunk)
        return LLMResponse(
            text="".join(self.chunks), model=self.model, input_tokens=42, output_tokens=9
        )


def use(monkeypatch: pytest.MonkeyPatch, provider) -> None:
    """Point every get_provider() call in the service at a scripted provider."""
    monkeypatch.setattr(api, "get_provider", lambda name=None: provider)


def sse_events(text: str) -> list[tuple[str, dict]]:
    """Parse an SSE body into (event, data) pairs."""
    events = []
    for frame in text.split("\n\n"):
        if not frame.strip():
            continue
        name = next((l[7:] for l in frame.splitlines() if l.startswith("event: ")), None)
        raw = next((l[6:] for l in frame.splitlines() if l.startswith("data: ")), None)
        if name and raw is not None:
            events.append((name, json.loads(raw)))
    return events


# --------------------------------------------------------------------------
# Timer
# --------------------------------------------------------------------------


def test_ttft_is_none_until_something_is_streamed() -> None:
    timer = RequestTimer()
    assert timer.ttft_ms is None
    timer.mark_first_token()
    assert timer.ttft_ms is not None and timer.ttft_ms >= 0


def test_marking_the_first_token_twice_keeps_the_first_time() -> None:
    """The caller marks from inside a per-token callback and should not care."""
    timer = RequestTimer()
    timer.mark_first_token()
    first = timer.first_token_at
    time.sleep(0.01)
    timer.mark_first_token()
    assert timer.first_token_at == first


def test_generation_time_separates_model_speed_from_retrieval_speed() -> None:
    timer = RequestTimer()
    time.sleep(0.02)
    timer.mark_first_token()
    time.sleep(0.02)
    timer.output_tokens = 10
    timer.finish()

    assert timer.ttft_ms >= 15
    assert timer.generation_ms >= 15
    assert timer.total_ms >= timer.ttft_ms
    assert timer.tokens_per_second is not None


def test_tokens_per_second_is_none_without_output() -> None:
    timer = RequestTimer()
    timer.mark_first_token()
    timer.finish()
    assert timer.tokens_per_second is None


# --------------------------------------------------------------------------
# /health
# --------------------------------------------------------------------------


def test_health_reports_the_configured_models(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["provider"] in {"local", "anthropic", "echo"}
    assert body["embedding_model"]


# --------------------------------------------------------------------------
# /ask streaming
# --------------------------------------------------------------------------


def test_ask_streams_tokens_then_a_done_event(
    client: TestClient, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use(monkeypatch, ScriptedProvider("Interest ", "compounds. ", "`billing.py:4-6`"))

    response = client.post(
        "/ask", json={"repo": str(repo), "question": "how does interest work?", "k": 3}
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = sse_events(response.text)
    names = [name for name, _ in events]

    assert names[0] == "meta"
    assert names[-1] == "done"
    assert names.count("token") == 3, "one event per streamed chunk, not one blob"

    tokens = "".join(d["text"] for n, d in events if n == "token")
    assert tokens == "Interest compounds. `billing.py:4-6`"


def test_done_carries_metrics_including_ttft(
    client: TestClient, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use(monkeypatch, ScriptedProvider("a", "b", delay=0.02))

    events = sse_events(
        client.post("/ask", json={"repo": str(repo), "question": "q", "k": 2}).text
    )
    done = next(d for n, d in events if n == "done")
    metrics = done["metrics"]

    assert metrics["ttft_ms"] is not None and metrics["ttft_ms"] > 0
    assert metrics["total_ms"] >= metrics["ttft_ms"], "ttft cannot exceed total"
    assert metrics["input_tokens"] == 42 and metrics["output_tokens"] == 9
    assert metrics["generation_ms"] is not None


def test_ask_reports_sources_and_repeats_them_in_done(
    client: TestClient, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use(monkeypatch, ScriptedProvider())
    events = sse_events(
        client.post("/ask", json={"repo": str(repo), "question": "interest", "k": 3}).text
    )

    sources = next(d for n, d in events if n == "sources")
    done = next(d for n, d in events if n == "done")
    assert sources, "retrieval should find the sample code"
    assert [s["location"] for s in done["sources"]] == [s["location"] for s in sources]
    assert "start_line" in sources[0] and "symbol" in sources[0]


def test_plain_mode_reports_grounded_as_null_not_false(
    client: TestClient, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plain RAG never checks, so claiming "not grounded" would be a lie."""
    use(monkeypatch, ScriptedProvider())
    events = sse_events(
        client.post("/ask", json={"repo": str(repo), "question": "q", "mode": "plain"}).text
    )
    assert next(d for n, d in events if n == "done")["grounded"] is None


def test_agent_mode_streams_stages_and_a_grounded_verdict(
    client: TestClient, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use(monkeypatch, ScriptedProvider(
        "Interest compounds. `billing.py:4-6`",
    ))
    # The scripted provider answers, then its default reply is used as the
    # critique, which parses as grounded.
    events = sse_events(
        client.post("/ask", json={"repo": str(repo), "question": "q", "mode": "agent"}).text
    )
    names = [n for n, _ in events]
    done = next(d for n, d in events if n == "done")

    assert "stage" in names, "agent mode should report progress"
    assert done["mode"] == "agent"
    assert done["grounded"] in (True, False), "agent mode always has a verdict"
    assert done["attempts"] >= 1


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


def test_an_unindexed_repo_is_a_404_error_event(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stream has already started, so the failure arrives as an event."""
    use(monkeypatch, ScriptedProvider())
    (tmp_path / "x.py").write_text("x = 1", encoding="utf-8")

    events = sse_events(
        client.post("/ask", json={"repo": str(tmp_path), "question": "q"}).text
    )
    name, data = events[-1]
    assert name == "error"
    assert data["status"] == 404
    assert "not indexed" in data["detail"]


def test_an_unreachable_model_is_a_503(
    client: TestClient, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The request was fine; the dependency was not."""
    use(monkeypatch, ScriptedProvider(fail=LLMError("no local model server")))

    events = sse_events(client.post("/ask", json={"repo": str(repo), "question": "q"}).text)
    name, data = events[-1]
    assert name == "error" and data["status"] == 503


def test_a_blank_question_is_rejected_before_any_work(client: TestClient) -> None:
    assert client.post("/ask", json={"repo": ".", "question": ""}).status_code == 422


def test_k_is_bounded(client: TestClient) -> None:
    assert client.post("/ask", json={"repo": ".", "question": "q", "k": 0}).status_code == 422
    assert client.post("/ask", json={"repo": ".", "question": "q", "k": 999}).status_code == 422


# --------------------------------------------------------------------------
# /review
# --------------------------------------------------------------------------


def test_review_returns_structured_comments(
    client: TestClient, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use(monkeypatch, ScriptedProvider(
        '[{"line": 6, "severity": "blocker", "message": "Multiplication is not exponentiation."}]'
    ))

    response = client.post("/review", json={"repo": str(repo), "diff": DIFF})
    assert response.status_code == 200

    body = response.json()
    assert body["comments"] == [{
        "file": "billing.py", "line": 6, "severity": "blocker",
        "message": "Multiplication is not exponentiation.",
    }]
    assert body["by_severity"] == {"blocker": 1}
    assert body["hunks_reviewed"] == 1
    assert body["metrics"]["input_tokens"] == 42


def test_review_of_a_clean_diff_returns_no_comments(
    client: TestClient, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use(monkeypatch, ScriptedProvider("[]"))
    body = client.post("/review", json={"repo": str(repo), "diff": DIFF}).json()
    assert body["comments"] == []
    assert body["by_severity"] == {}


def test_review_rejects_an_empty_diff(client: TestClient) -> None:
    assert client.post("/review", json={"repo": ".", "diff": ""}).status_code == 422


def test_review_max_hunks_is_bounded(client: TestClient) -> None:
    body = {"repo": ".", "diff": DIFF, "max_hunks": 0}
    assert client.post("/review", json=body).status_code == 422


# --------------------------------------------------------------------------
# /index and /repos — load a repository through the API
# --------------------------------------------------------------------------


def test_indexing_streams_progress_then_done(
    client: TestClient, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api, "clone_github_repo", lambda ref, on_progress=None: repo)
    events = sse_events(client.post("/index", json={"repo": "psf/requests"}).text)
    names = [n for n, _ in events]

    assert "stage" in names, "the user is waiting; indexing must narrate"
    assert names[-1] == "done"

    done = next(d for n, d in events if n == "done")
    assert done["path"] == str(repo)
    assert done["chunks"] > 0


def test_a_local_path_is_not_accepted_by_index(client: TestClient, repo: Path) -> None:
    """The product takes GitHub repositories only. A path that exists on the
    server must not be indexable through the web — that would let any visitor
    read arbitrary local directories through the Q&A."""
    events = sse_events(client.post("/index", json={"repo": str(repo)}).text)
    name, data = events[-1]
    assert name == "error"
    assert data["status"] == 400


def test_indexing_a_github_ref_clones_then_indexes(
    client: TestClient, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The clone itself is scripted: this pins the endpoint's wiring — ref in,
    clone called, resulting path indexed and reported."""
    def fake_clone(ref, on_progress=None):
        if on_progress:
            on_progress("cloning (scripted)")
        return repo

    monkeypatch.setattr(api, "clone_github_repo", fake_clone)
    events = sse_events(client.post("/index", json={"repo": "psf/requests"}).text)

    done = next(d for n, d in events if n == "done")
    assert done["name"] == "psf/requests"
    assert done["path"] == str(repo)
    assert any("scripted" in d.get("detail", "") for n, d in events if n == "stage")


def test_an_invalid_ref_is_a_400_error_event(client: TestClient) -> None:
    events = sse_events(client.post("/index", json={"repo": "not a repo ref !!"}).text)
    name, data = events[-1]
    assert name == "error"
    assert data["status"] == 400
    assert "GitHub" in data["detail"]


def test_repos_lists_the_store_with_indexed_flags(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src import config as cfg

    (tmp_path / "psf__requests").mkdir()
    (tmp_path / "local-checkout").mkdir()
    (tmp_path / "stray-file.txt").write_text("x", encoding="utf-8")
    monkeypatch.setattr(cfg, "REPOS_DIR", tmp_path)

    body = client.get("/repos").json()
    names = {r["name"] for r in body["repos"]}
    assert names == {"psf/requests", "local-checkout"}, "files are not repos"
    assert all(r["indexed"] is False for r in body["repos"]), "nothing indexed yet"


def test_repos_survives_a_missing_store(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src import config as cfg

    monkeypatch.setattr(cfg, "REPOS_DIR", tmp_path / "never-created")
    assert client.get("/repos").json() == {"repos": []}


# --------------------------------------------------------------------------
# Frontend
# --------------------------------------------------------------------------


def test_the_ui_is_served_from_the_root(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Codebase Q&amp;A" in response.text


def test_the_ui_has_no_external_dependencies() -> None:
    """It must work offline: this whole project runs against a local model."""
    html = (Path("src/serve/static/index.html")).read_text(encoding="utf-8")
    assert "http://" not in html.replace("http://127.0.0.1", "").replace("http://localhost", "")
    assert "https://" not in html
    assert "<script src=" not in html and "<link rel=\"stylesheet\"" not in html


def test_openapi_documents_both_endpoints(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert "/ask" in paths and "/review" in paths and "/health" in paths
