"""Documentation chunking: headings, windows, and pipeline integration."""

from __future__ import annotations

from pathlib import Path

import pytest

from src import config
from src.ingest import chunk_doc, ingest_repo, is_doc_file

README = """\
# Widget Factory

Turns raw widgets into shipped widgets. Fast.

## Installation

pip install widget-factory

## How it works

Widgets flow through a pipeline of stages.
Each stage is a plain function.
"""


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "README.md").write_text(README, encoding="utf-8")
    return tmp_path


def test_doc_files_are_recognised_by_extension() -> None:
    assert is_doc_file("README.md")
    assert is_doc_file("docs/guide.rst")
    assert is_doc_file("NOTES.TXT")
    assert not is_doc_file("main.py")
    assert not is_doc_file("Makefile")


def test_readme_style_names_qualify_anywhere_but_doc_trees_do_not(tmp_path: Path) -> None:
    """Measured: indexing a whole docs/ tree floods retrieval with prose that
    outranks the code it describes (held-out semantic MRR 0.649 -> 0.564)."""
    from src.ingest import qualifies_as_doc

    assert qualifies_as_doc(tmp_path / "README.md", tmp_path)
    assert qualifies_as_doc(tmp_path / "notes.txt", tmp_path), "root-level docs qualify"
    assert qualifies_as_doc(tmp_path / "pkg" / "CHANGELOG.rst", tmp_path)
    assert qualifies_as_doc(tmp_path / "sub" / "readme.md", tmp_path)
    assert not qualifies_as_doc(tmp_path / "docs" / "advanced.rst", tmp_path)
    assert not qualifies_as_doc(tmp_path / "docs" / "guide" / "intro.md", tmp_path)
    assert not qualifies_as_doc(tmp_path / "main.py", tmp_path)


def test_markdown_splits_on_headings_with_the_heading_as_symbol(repo: Path) -> None:
    chunks = chunk_doc(repo / "README.md", repo)
    assert [c.symbol for c in chunks] == ["Widget Factory", "Installation", "How it works"]
    assert all(c.kind == "doc" for c in chunks)
    assert all(c.language == "Markdown" for c in chunks)


def test_doc_line_numbers_are_citable(repo: Path) -> None:
    """`README.md:1-4` must contain exactly what the chunk says it does."""
    lines = README.splitlines()
    for chunk in chunk_doc(repo / "README.md", repo):
        assert chunk.code == "\n".join(lines[chunk.start_line - 1 : chunk.end_line])
        assert chunk.file_path == "README.md"


def test_text_before_the_first_heading_is_kept(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("An intro line.\n\n# Title\n\nBody.\n", encoding="utf-8")
    chunks = chunk_doc(tmp_path / "README.md", tmp_path)
    assert chunks[0].symbol == "README"
    assert "An intro line." in chunks[0].code


def test_a_headingless_file_is_windowed_whole(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("line one\nline two\n", encoding="utf-8")
    chunks = chunk_doc(tmp_path / "notes.txt", tmp_path)
    assert len(chunks) == 1
    assert chunks[0].symbol == "notes"
    assert chunks[0].language == "Text"


def test_a_huge_section_is_split_into_parts(tmp_path: Path) -> None:
    body = "# One Big Section\n" + "\n".join(f"line {n}" for n in range(500))
    (tmp_path / "BIG.md").write_text(body, encoding="utf-8")
    chunks = chunk_doc(tmp_path / "BIG.md", tmp_path)
    assert len(chunks) > 1
    assert all(c.line_count <= config.MAX_CHUNK_LINES for c in chunks)
    assert {c.part for c in chunks} == set(range(1, len(chunks) + 1))


def test_an_empty_doc_produces_nothing(tmp_path: Path) -> None:
    (tmp_path / "EMPTY.md").write_text("\n\n\n", encoding="utf-8")
    assert chunk_doc(tmp_path / "EMPTY.md", tmp_path) == []


def test_ingest_indexes_readme_alongside_code(repo: Path) -> None:
    """The reason docs exist in the index: 'what is this repo about' must have
    a chunk that answers it."""
    (repo / "main.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    chunks = ingest_repo(repo)

    kinds = {c.kind for c in chunks}
    assert "doc" in kinds and "function" in kinds
    readme = [c for c in chunks if c.file_path == "README.md"]
    assert any("shipped widgets" in c.code for c in readme)
