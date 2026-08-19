"""Gitignore filtering, delegated to git itself.

Reimplementing gitignore semantics (negations, precedence, nested .gitignore
files, core.excludesFile) is a lot of subtle work to get wrong. Since the repos
we index are git clones, `git check-ignore` is both correct and free.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path

_GIT_TIMEOUT_SECONDS = 60


def is_git_repo(repo_path: Path) -> bool:
    """True if `repo_path` is inside a git working tree."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and result.stdout.strip() == "true"


def filter_ignored(repo_path: Path, files: Sequence[Path]) -> list[Path]:
    """Drop the paths git would ignore, preserving input order.

    Falls back to returning `files` unchanged when the path is not a git repo
    or git is unavailable — indexing a plain directory should still work.
    """
    if not files or not is_git_repo(repo_path):
        return list(files)

    # -z on both sides so paths containing spaces or newlines survive intact.
    stdin = "\0".join(str(f) for f in files)
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "check-ignore", "-z", "--stdin"],
            input=stdin,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return list(files)

    # Exit 0 = some paths ignored, 1 = none ignored. Anything else is a real
    # failure, and silently indexing everything beats crashing the ingest.
    if result.returncode not in (0, 1):
        return list(files)

    ignored = {Path(p) for p in result.stdout.split("\0") if p}
    return [f for f in files if f not in ignored]
