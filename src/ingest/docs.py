"""Chunking documentation files: README, docs/, changelogs.

Prose answers the questions code cannot — "what is this repo about", "how do I
install it" — so docs are indexed alongside code. Markdown splits on headings,
which gives each chunk a citable symbol (the heading) the way code chunks get a
function name; everything else falls back to fixed-line windows.
"""

from __future__ import annotations

import re
from pathlib import Path

from src import config

from .models import Chunk

# A markdown ATX heading. Setext headings (underlined with === / ---) are rare
# in modern READMEs and are simply absorbed into the preceding section.
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")

_LANGUAGE_BY_SUFFIX = {
    ".md": "Markdown",
    ".markdown": "Markdown",
    ".rst": "reStructuredText",
    ".txt": "Text",
}


_DOC_STEM = re.compile(config.DOC_STEM_PATTERN, re.IGNORECASE)


def is_doc_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in config.DOC_EXTENSIONS


def qualifies_as_doc(path: str | Path, repo_root: str | Path) -> bool:
    """Whether a doc file earns a place in the index.

    Root-level docs and README-style names anywhere qualify; whole docs/ trees
    do not (see the DOC_STEM_PATTERN note in config for the measurement).
    """
    path = Path(path)
    if not is_doc_file(path):
        return False
    return path.parent == Path(repo_root) or bool(_DOC_STEM.match(path.stem))


def chunk_doc(path: str | Path, repo_root: str | Path) -> list[Chunk]:
    """Split one documentation file into citable chunks.

    Line numbers are 1-based and inclusive, exactly like code chunks, so a
    `README.md:1-18` citation survives the same reflection check.
    """
    path, root = Path(path), Path(repo_root)
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    if not any(line.strip() for line in lines):
        return []

    rel = path.relative_to(root).as_posix()
    language = _LANGUAGE_BY_SUFFIX.get(path.suffix.lower(), "Text")

    if language == "Markdown":
        sections = _split_on_headings(lines, fallback_symbol=path.stem)
    else:
        sections = [(path.stem, 1, len(lines))]

    chunks: list[Chunk] = []
    for symbol, start, end in sections:
        chunks.extend(_windows(lines, rel, language, symbol, start, end))
    return chunks


def _split_on_headings(
    lines: list[str], fallback_symbol: str
) -> list[tuple[str, int, int]]:
    """(symbol, start_line, end_line) per section, 1-based inclusive.

    A section runs from its heading to the line before the next heading of any
    level. Splitting on *every* level rather than rebuilding the outline keeps
    sections small and the code simple; the heading text is the symbol, so
    retrieval can match it and citations read naturally.
    """
    headings = [
        (number, match.group(2).strip())
        for number, line in enumerate(lines, start=1)
        if (match := _HEADING.match(line))
    ]
    if not headings:
        return [(fallback_symbol, 1, len(lines))]

    sections: list[tuple[str, int, int]] = []
    first_heading_line = headings[0][0]
    if first_heading_line > 1 and any(line.strip() for line in lines[: first_heading_line - 1]):
        sections.append((fallback_symbol, 1, first_heading_line - 1))

    for i, (start, title) in enumerate(headings):
        end = (headings[i + 1][0] - 1) if i + 1 < len(headings) else len(lines)
        sections.append((title, start, end))
    return sections


def _windows(
    lines: list[str], rel: str, language: str, symbol: str, start: int, end: int
) -> list[Chunk]:
    """Bound a section the same way oversized code symbols are bounded."""
    spans: list[tuple[int, int]] = []
    cursor = start
    while cursor <= end:
        spans.append((cursor, min(cursor + config.MAX_CHUNK_LINES - 1, end)))
        cursor += config.MAX_CHUNK_LINES

    total = len(spans)
    chunks = []
    for part, (lo, hi) in enumerate(spans, start=1):
        body = "\n".join(lines[lo - 1 : hi])
        if not body.strip():
            continue
        chunks.append(
            Chunk(
                code=body[: config.MAX_CHUNK_CHARS],
                language=language,
                file_path=rel,
                symbol=symbol,
                kind="doc",
                start_line=lo,
                end_line=hi,
                part=part,
                part_count=total,
            )
        )
    return chunks
