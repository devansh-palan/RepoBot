"""Cloning: ref validation and the subprocess boundary.

No network anywhere. The parse tests pin what the untrusted web input may look
like; the subprocess tests script `subprocess.run` and assert on the argv,
because the security property is *what reaches git*, not what git does.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src import config
from src.ingest import CloneError, clone_github_repo, clone_path_for, parse_github_ref


# --------------------------------------------------------------------------
# Ref parsing
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "psf/requests",
        "https://github.com/psf/requests",
        "https://github.com/psf/requests.git",
        "https://github.com/psf/requests/",
        "http://github.com/psf/requests",
        "https://www.github.com/psf/requests",
        "github.com/psf/requests",
        "  psf/requests  ",
    ],
)
def test_every_reasonable_spelling_parses_to_the_same_ref(text: str) -> None:
    assert parse_github_ref(text) == ("psf", "requests")


def test_dots_and_underscores_are_legal_in_repo_names() -> None:
    assert parse_github_ref("pallets/flask.wiki_v2")[1] == "flask.wiki_v2"


@pytest.mark.parametrize(
    "text",
    [
        "",                                   # nothing
        "requests",                           # no owner
        "a/b/c",                              # extra path segment
        "gitlab.com/psf/requests",            # not github
        "https://evil.example/psf/requests",  # not github
        "ssh://git@github.com/psf/requests",  # wrong protocol
        "file:///etc/passwd",                 # wrong protocol
        "-flag/repo",                         # owner shaped like a git option
        "psf/-flag",                          # repo shaped like a git option
        "https://github.com/psf/requests?x=1",  # query string
    ],
)
def test_anything_else_is_rejected(text: str) -> None:
    with pytest.raises(CloneError):
        parse_github_ref(text)


# --------------------------------------------------------------------------
# Cloning
# --------------------------------------------------------------------------


@pytest.fixture
def repo_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(config, "REPOS_DIR", tmp_path)
    return tmp_path


def test_the_clone_command_is_shallow_and_option_free(
    repo_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The security property: exactly this argv, built only from validated parts."""
    seen: dict = {}

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        seen["shell"] = kwargs.get("shell", False)
        Path(argv[-1], ".git").mkdir(parents=True)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    dest = clone_github_repo("psf/requests")

    assert seen["argv"][:5] == ["git", "clone", "--depth", "1", "--quiet"]
    assert seen["argv"][5] == "https://github.com/psf/requests.git"
    assert seen["argv"][6] == str(dest)
    assert seen["shell"] is False
    assert dest == clone_path_for("psf", "requests")


def test_an_existing_clone_is_reused_without_touching_git(
    repo_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-indexing an already-loaded repo must not hit the network, and must
    not silently advance the checkout old citations point into."""
    dest = clone_path_for("psf", "requests")
    (dest / ".git").mkdir(parents=True)

    def explode(*args, **kwargs):
        raise AssertionError("git must not be invoked for an existing clone")

    monkeypatch.setattr(subprocess, "run", explode)
    assert clone_github_repo("psf/requests") == dest


def test_a_failed_clone_surfaces_gits_reason(
    repo_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(
            argv, 128, "", "fatal: repository 'x' not found\n"
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(CloneError, match="not found"):
        clone_github_repo("psf/definitely-not-a-repo")


def test_progress_is_reported_at_each_step(
    repo_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(argv, **kwargs):
        Path(argv[-1], ".git").mkdir(parents=True)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    lines: list[str] = []
    clone_github_repo("psf/requests", on_progress=lines.append)
    assert any("cloning" in line for line in lines)
    assert any("cloned into" in line for line in lines)
