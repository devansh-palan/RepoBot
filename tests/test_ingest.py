"""Tests for the ingest stage: walking, gitignore filtering, and chunking."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src import config
from src.ingest import MODULE_SYMBOL, Chunk, chunk_file, chunk_source, collect_source_files

FIXTURE = Path(__file__).parent / "fixtures" / "sample_module.py"
PY = config.LANGUAGE_REGISTRY[".py"]


@pytest.fixture(scope="module")
def chunks() -> list[Chunk]:
    return chunk_file(FIXTURE, FIXTURE.parent)


def _by_symbol(chunks: list[Chunk], symbol: str) -> Chunk:
    """The first chunk for a symbol, in source order.

    A class can own more than one chunk — the header, plus any gap-filled
    members declared after its last method — so this returns the earliest.
    """
    matches = [c for c in chunks if c.symbol == symbol]
    assert matches, f"no chunk for {symbol}; got {sorted({c.symbol for c in chunks})}"
    return matches[0]


# --------------------------------------------------------------------------
# Symbol extraction
# --------------------------------------------------------------------------


def test_finds_top_level_function(chunks: list[Chunk]) -> None:
    chunk = _by_symbol(chunks, "simple")
    assert chunk.kind == "function"
    assert chunk.code.startswith("def simple(value: int) -> int:")
    assert "return value * 2" in chunk.code


def test_methods_are_qualified_by_class(chunks: list[Chunk]) -> None:
    interest = _by_symbol(chunks, "Account.interest")
    assert interest.kind == "function"
    assert "def interest" in interest.code
    assert "def with_deposit" not in interest.code  # one method per chunk


def test_decorator_stays_with_its_symbol(chunks: list[Chunk]) -> None:
    account = _by_symbol(chunks, "Account")
    assert account.kind == "class"
    assert account.code.startswith("@dataclass(frozen=True)")
    # The class chunk is the header, not the whole body.
    assert "def interest" not in account.code


def test_class_chunk_keeps_docstring_and_fields(chunks: list[Chunk]) -> None:
    account = _by_symbol(chunks, "Account")
    assert "A bank account" in account.code
    assert "balance: float" in account.code


def test_closure_stays_inside_its_parent(chunks: list[Chunk]) -> None:
    parent = _by_symbol(chunks, "make_counter")
    assert "def increment" in parent.code
    assert not any(c.symbol == "increment" for c in chunks)


def test_each_function_produces_exactly_one_chunk(chunks: list[Chunk]) -> None:
    names = [c.symbol for c in chunks if c.kind == "function"]
    assert len(names) == len(set(names)), f"duplicate function chunks: {names}"


def test_async_function_is_found(chunks: list[Chunk]) -> None:
    chunk = _by_symbol(chunks, "fetch_rate")
    assert chunk.kind == "function"
    assert chunk.code.startswith("async def fetch_rate")


def test_trailing_class_attribute_is_attributed_to_the_class(
    chunks: list[Chunk],
) -> None:
    """CURRENCY sits after the last method, so it falls to gap filling."""
    holder = [c for c in chunks if "CURRENCY" in c.code]
    assert holder, "CURRENCY was dropped"
    assert holder[0].symbol == "Account"


# --------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------


def test_every_chunk_carries_full_metadata(chunks: list[Chunk]) -> None:
    for chunk in chunks:
        assert chunk.code
        assert chunk.language == "Python"
        assert chunk.file_path == "sample_module.py"
        assert chunk.symbol
        assert chunk.kind in {"function", "class", "type", "module"}
        assert 1 <= chunk.start_line <= chunk.end_line


def test_line_numbers_point_at_the_real_source(chunks: list[Chunk]) -> None:
    source_lines = FIXTURE.read_text(encoding="utf-8").split("\n")
    for chunk in chunks:
        expected = "\n".join(source_lines[chunk.start_line - 1 : chunk.end_line])
        assert chunk.code == expected, f"{chunk.location} does not match the file"


def test_module_level_code_is_kept(chunks: list[Chunk]) -> None:
    module_code = "\n".join(c.code for c in chunks if c.symbol == MODULE_SYMBOL)
    assert "DEFAULT_RATE = 0.05" in module_code
    assert "import math" in module_code


# --------------------------------------------------------------------------
# The core guarantee: nothing is dropped
# --------------------------------------------------------------------------


def test_no_content_line_is_dropped(chunks: list[Chunk]) -> None:
    source_lines = FIXTURE.read_text(encoding="utf-8").split("\n")
    covered = {
        line
        for chunk in chunks
        for line in range(chunk.start_line, chunk.end_line + 1)
    }
    missing = [
        number
        for number, text in enumerate(source_lines, start=1)
        if text.strip() and number not in covered
    ]
    assert not missing, f"dropped lines: {missing}"


def test_oversized_function_is_split_with_overlap() -> None:
    body = "\n".join(f"    x = {i}" for i in range(config.MAX_CHUNK_LINES * 2))
    source = f"def huge():\n{body}\n".encode()

    parts = chunk_source(source, "huge.py", PY)

    assert len(parts) > 1
    assert {p.part for p in parts} == set(range(1, len(parts) + 1))
    assert all(p.part_count == len(parts) for p in parts)
    assert all(p.line_count <= config.MAX_CHUNK_LINES for p in parts)
    # Consecutive windows must overlap, so a construct on the seam survives whole.
    for earlier, later in zip(parts, parts[1:]):
        assert later.start_line <= earlier.end_line


def test_split_parts_still_cover_every_line() -> None:
    body = "\n".join(f"    x = {i}" for i in range(config.MAX_CHUNK_LINES * 2))
    source = f"def huge():\n{body}\n".encode()

    parts = chunk_source(source, "huge.py", PY)

    covered = {n for p in parts for n in range(p.start_line, p.end_line + 1)}
    assert covered == set(range(1, config.MAX_CHUNK_LINES * 2 + 2))


def test_long_single_line_still_produces_a_chunk() -> None:
    """A line longer than MAX_CHUNK_CHARS must not stall the window loop."""
    source = f"X = '{'a' * (config.MAX_CHUNK_CHARS * 2)}'\n".encode()
    result = chunk_source(source, "long.py", PY)
    assert len(result) == 1


def test_empty_file_produces_no_chunks() -> None:
    assert chunk_source(b"", "empty.py", PY) == []


def test_syntax_errors_do_not_lose_code() -> None:
    """tree-sitter recovers from bad syntax; gap filling catches the rest."""
    source = b"def ok():\n    pass\n\ndef broken(:\n    still_here = 1\n"
    result = chunk_source(source, "broken.py", PY)
    assert any("still_here" in c.code for c in result)


# --------------------------------------------------------------------------
# Walking and filtering
# --------------------------------------------------------------------------


def test_unsupported_extension_is_skipped(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("# not code", encoding="utf-8")
    assert chunk_file(tmp_path / "notes.md", tmp_path) == []


def test_walker_finds_supported_files_and_prunes_ignored_dirs(tmp_path: Path) -> None:
    (tmp_path / "keep.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("docs\n", encoding="utf-8")  # docs are indexed too
    (tmp_path / "logo.png").write_bytes(b"\x89PNG")
    vendored = tmp_path / "node_modules" / "pkg"
    vendored.mkdir(parents=True)
    (vendored / "bundled.py").write_text("y = 2\n", encoding="utf-8")

    found = {p.name for p in collect_source_files(tmp_path)}
    assert found == {"keep.py", "README.md"}


def test_walker_respects_gitignore(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / ".gitignore").write_text("generated/\n", encoding="utf-8")
    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")
    generated = tmp_path / "generated"
    generated.mkdir()
    (generated / "pb2.py").write_text("y = 2\n", encoding="utf-8")

    found = {p.name for p in collect_source_files(tmp_path)}
    assert found == {"app.py"}


def test_walker_skips_oversized_files(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "big.py").write_text("x = 1\n" * 100, encoding="utf-8")
    monkeypatch.setattr(config, "MAX_FILE_BYTES", 10)
    assert collect_source_files(tmp_path) == []
