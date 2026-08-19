"""The Chunk record — the unit that flows through the whole pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """One retrievable piece of code plus the metadata needed to cite it.

    Line numbers are 1-based and inclusive on both ends, matching how editors
    and diffs number lines.
    """

    code: str
    language: str      # LanguageSpec.name, e.g. "Python"
    file_path: str     # POSIX-style, relative to the repo root
    symbol: str        # qualified name, e.g. "Chunk.location"; MODULE_SYMBOL for loose code
    kind: str          # "function" | "class" | "type" | "module"
    start_line: int
    end_line: int
    part: int = 1      # 1-based index when an oversized symbol was split
    part_count: int = 1

    @property
    def line_count(self) -> int:
        return self.end_line - self.start_line + 1

    @property
    def location(self) -> str:
        """Human- and LLM-readable citation, e.g. 'src/config.py:10-42'."""
        return f"{self.file_path}:{self.start_line}-{self.end_line}"

    def header(self) -> str:
        """One-line description prepended when a chunk is shown to the model."""
        part = f" (part {self.part}/{self.part_count})" if self.part_count > 1 else ""
        return f"{self.location} {self.kind} {self.symbol}{part} [{self.language}]"


# Symbol name used for code that lives outside any function or class.
MODULE_SYMBOL = "<module>"
