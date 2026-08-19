"""Repo path in, chunks out — the whole ingest stage in one call."""

from __future__ import annotations

import warnings
from pathlib import Path

from .chunker import MissingGrammarError, chunk_file
from .docs import chunk_doc, is_doc_file
from .models import Chunk
from .walker import collect_source_files


def ingest_repo(repo_path: str | Path) -> list[Chunk]:
    """Chunk every indexable file in a repository.

    Files that fail to read or parse are skipped rather than aborting the run:
    one unreadable file in a large repo should not cost the whole index. A
    language whose grammar is missing is warned about once, not once per file,
    so a misconfigured registry entry is visible without drowning the output.
    """
    root = Path(repo_path).resolve()
    chunks: list[Chunk] = []
    reported: set[str] = set()

    for path in collect_source_files(root):
        try:
            chunks.extend(chunk_doc(path, root) if is_doc_file(path) else chunk_file(path, root))
        except MissingGrammarError as exc:
            if str(exc) not in reported:
                reported.add(str(exc))
                warnings.warn(f"skipping {path.suffix} files: {exc}", stacklevel=2)
        except (OSError, UnicodeError):
            continue

    return chunks
