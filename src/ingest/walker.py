"""Finding the source files in a repository that are worth indexing."""

from __future__ import annotations

import os
from pathlib import Path

from src import config

from .docs import qualifies_as_doc
from .gitignore import filter_ignored


def collect_source_files(repo_path: str | Path) -> list[Path]:
    """Return every indexable file under `repo_path`, sorted.

    A file is indexable when its extension is in LANGUAGE_REGISTRY, or it is a
    qualifying doc file (README-style names and repo-root docs — they answer
    the overview questions code cannot; see docs.qualifies_as_doc). It must
    also not be inside an ignored directory, not be larger than
    MAX_FILE_BYTES, and not be git-ignored. Paths are absolute so callers can
    read them directly.
    """
    root = Path(repo_path).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"not a directory: {root}")

    candidates: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Mutating dirnames in place prunes the walk — os.walk will not descend
        # into node_modules/.git/etc. at all, rather than filtering afterwards.
        dirnames[:] = sorted(d for d in dirnames if d not in config.IGNORE_DIRS)

        for filename in filenames:
            path = Path(dirpath) / filename
            suffix = path.suffix.lower()
            if suffix not in config.SUPPORTED_EXTENSIONS:
                if suffix not in config.DOC_EXTENSIONS or not qualifies_as_doc(path, root):
                    continue
            if _is_too_large(path):
                continue
            candidates.append(path)

    return sorted(filter_ignored(root, candidates))


def _is_too_large(path: Path) -> bool:
    """True for files past MAX_FILE_BYTES, or that we cannot stat at all.

    Anything that big is almost always generated or vendored; skipping beats
    spending the embedding budget on it.
    """
    try:
        return path.stat().st_size > config.MAX_FILE_BYTES
    except OSError:
        return True
