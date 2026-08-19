"""The three tools the agent can use.

Each is a plain function returning a plain string or dataclass — no framework
binding — so they stay callable from tests, from the graph, and later from a
tool-calling loop without being rewritten.

`run_tests` is the only one that touches the outside world, and it reads its
command from LANGUAGE_REGISTRY rather than assuming pytest, which is what keeps
the "add a language = add a registry entry" promise true for this tool too.
"""

from __future__ import annotations

import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from src import config
from src.index import SearchResult
from src.retrieve import hybrid_search

# ---------------------------------------------------------------------------
# search_code
# ---------------------------------------------------------------------------


def search_code(
    repo_path: str | Path,
    query: str,
    k: int = config.FINAL_TOP_K,
    persist_dir: str | None = None,
    bm25_dir: str | Path | None = None,
) -> list[SearchResult]:
    """Retrieve the k most relevant chunks for a query.

    Hybrid by default because that is what the agent should use; the eval calls
    the vector-only and bm25-only retrievers directly when it needs to compare.

    The index-location overrides mirror index_repo's, so the eval can point the
    agent at a separately built index without disturbing the real one.
    """
    return hybrid_search(repo_path, query, k, persist_dir, Path(bm25_dir) if bm25_dir else None)


def format_results(results: list[SearchResult]) -> str:
    """Render chunks as numbered, citable excerpts for a prompt."""
    return "\n\n".join(
        f"[{n}] {r.chunk.location}  {r.chunk.kind} {r.chunk.symbol}  ({r.chunk.language})\n"
        f"```\n{r.chunk.code}\n```"
        for n, r in enumerate(results, start=1)
    )


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------

MAX_READ_LINES = 400


@dataclass(frozen=True)
class FileSlice:
    """A span of a file, with the line numbers needed to cite it."""

    file_path: str
    start_line: int
    end_line: int
    text: str
    truncated: bool = False

    @property
    def location(self) -> str:
        return f"{self.file_path}:{self.start_line}-{self.end_line}"


def read_file(
    repo_path: str | Path,
    file_path: str,
    start_line: int = 1,
    end_line: int | None = None,
) -> FileSlice:
    """Read a span of a file, 1-based and inclusive like every citation.

    Retrieval returns chunks, and a chunk is sometimes the wrong window — a
    caller may need the imports above a function, or the branch just past the
    end of a split chunk. This is the escape hatch for that.

    Refuses to escape the repository: `file_path` comes from a model, so it is
    resolved and checked against the root rather than trusted.
    """
    root = Path(repo_path).resolve()
    target = (root / file_path).resolve()

    # A model-supplied path could contain ../.. and walk out of the repo.
    if not target.is_relative_to(root):
        raise ValueError(f"{file_path!r} resolves outside the repository")
    if not target.is_file():
        raise FileNotFoundError(f"no file at {file_path}")

    lines = target.read_text(encoding="utf-8", errors="replace").split("\n")
    start = max(1, start_line)
    end = len(lines) if end_line is None else min(end_line, len(lines))

    truncated = end - start + 1 > MAX_READ_LINES
    if truncated:
        end = start + MAX_READ_LINES - 1

    return FileSlice(
        file_path=target.relative_to(root).as_posix(),
        start_line=start,
        end_line=end,
        text="\n".join(lines[start - 1 : end]),
        truncated=truncated,
    )


# ---------------------------------------------------------------------------
# run_tests
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TestRun:
    """The outcome of running a repository's test suite."""

    command: tuple[str, ...]
    language: str
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def passed(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def summary(self, limit: int = 2000) -> str:
        """Tail of the output — failures are at the end of every runner's log."""
        verdict = "timed out" if self.timed_out else ("passed" if self.passed else "failed")
        tail = (self.stdout + self.stderr)[-limit:]
        return f"$ {' '.join(self.command)}\n[{self.language}] {verdict}\n{tail}"


def detect_language(repo_path: str | Path) -> config.LanguageSpec | None:
    """The repository's dominant language, by source file count.

    Counting files rather than reading a manifest keeps this working on any
    repo, and dominance is the right question: a Python project with one
    vendored .js file should still run pytest.
    """
    from src.ingest import collect_source_files

    counts: Counter[str] = Counter()
    specs: dict[str, config.LanguageSpec] = {}
    for path in collect_source_files(repo_path):
        spec = config.spec_for_path(path)
        if spec is not None:
            counts[spec.name] += 1
            specs[spec.name] = spec

    if not counts:
        return None
    return specs[counts.most_common(1)[0][0]]


def run_tests(
    repo_path: str | Path,
    timeout: int = config.TEST_TIMEOUT_SECONDS,
) -> TestRun:
    """Run the repository's test suite, choosing the command by language.

    The command comes from LANGUAGE_REGISTRY, so adding a language brings its
    test runner with it and this function never grows a branch.

    NOTE: this executes code from the repository under review. A plain
    subprocess is acceptable for this project, as agreed in CLAUDE.md. In
    production it must run sandboxed: a container with no network, a read-only
    mount of everything outside the workspace, a memory and CPU cap, and a
    non-root user. A test suite can do anything a shell can.
    """
    root = Path(repo_path).resolve()
    spec = detect_language(root)
    if spec is None:
        raise ValueError(f"no supported source files under {root}, so no test command to pick")

    try:
        completed = subprocess.run(
            spec.test_command,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,  # argv form; never let repo content reach a shell
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return TestRun(
            command=spec.test_command,
            language=spec.name,
            exit_code=-1,
            stdout=exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
            stderr="",
            timed_out=True,
        )
    except (OSError, FileNotFoundError) as exc:
        # The runner is not installed — a normal situation, not a crash.
        return TestRun(
            command=spec.test_command,
            language=spec.name,
            exit_code=127,
            stdout="",
            stderr=f"could not run {spec.test_command[0]!r}: {exc}",
        )

    return TestRun(
        command=spec.test_command,
        language=spec.name,
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


TOOLS = {
    "search_code": search_code,
    "read_file": read_file,
    "run_tests": run_tests,
}
