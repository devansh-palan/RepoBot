"""Cloning a GitHub repository into the local repo store.

The serve layer lets a user paste a repo they want to ask about, so the input
is untrusted. Two rules keep this safe: only GitHub over https (no arbitrary
URLs, no ssh, no local file:// tricks), and the owner/name are validated
against GitHub's own naming rules before they go anywhere near a command line,
so nothing that parses as a git option can be smuggled in.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable
from pathlib import Path

from src import config

# GitHub's actual rules: owners are alphanumeric plus hyphens, repo names also
# allow ., _ . Neither may start with "-", which is also what keeps a name from
# ever being read as a flag by git.
_OWNER = r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?"
_REPO = r"[A-Za-z0-9_.][A-Za-z0-9_.-]*"

_GITHUB_REF = re.compile(
    rf"^(?:(?:https?://)?(?:www\.)?github\.com/)?({_OWNER})/({_REPO})$"
)


class CloneError(RuntimeError):
    """The ref was invalid or git could not fetch it."""


def parse_github_ref(text: str) -> tuple[str, str]:
    """Extract (owner, repo) from a GitHub URL or an owner/repo shorthand.

    Accepts `https://github.com/psf/requests`, `github.com/psf/requests`,
    `psf/requests`, with or without a trailing `.git` or `/`. Anything else —
    other hosts, extra path segments, option-shaped names — is rejected.
    """
    cleaned = text.strip().removesuffix("/").removesuffix(".git")
    match = _GITHUB_REF.match(cleaned)
    if not match:
        raise CloneError(
            f"not a GitHub repository reference: {text!r} "
            "(expected owner/repo or https://github.com/owner/repo)"
        )
    return match.group(1), match.group(2)


def clone_path_for(owner: str, repo: str) -> Path:
    """Where a clone lives. Owner is part of the name so forks cannot collide."""
    return config.REPOS_DIR / f"{owner}__{repo}"


def clone_github_repo(
    ref: str,
    on_progress: Callable[[str], None] | None = None,
) -> Path:
    """Shallow-clone a GitHub repo into the repo store, reusing an existing clone.

    Reuse rather than pull: indexing is keyed to what is on disk, and silently
    advancing a repo the user has already asked questions about would make old
    citations point at moved lines. Delete the directory to refresh.
    """
    owner, repo = parse_github_ref(ref)
    dest = clone_path_for(owner, repo)

    if (dest / ".git").exists():
        if on_progress:
            on_progress(f"already cloned at {dest}, reusing")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/{owner}/{repo}.git"
    if on_progress:
        on_progress(f"cloning {url} (shallow)")

    # shell=False and a validated URL: nothing user-controlled is parsed by a
    # shell, and nothing option-shaped survives parse_github_ref.
    result = subprocess.run(
        ["git", "clone", "--depth", "1", "--quiet", url, str(dest)],
        capture_output=True,
        text=True,
        timeout=config.CLONE_TIMEOUT_SECONDS,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        raise CloneError(f"git clone failed for {url}: {detail[-1] if detail else 'unknown error'}")

    if on_progress:
        on_progress(f"cloned into {dest}")
    return dest
