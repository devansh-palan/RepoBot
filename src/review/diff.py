"""Unified diff parsing.

A review comment has to land on a line the reviewer can actually see, so the
one thing this module cannot get wrong is line numbering. Every parsed line
carries the file line number it corresponds to on each side, computed while
walking the hunk rather than reconstructed later.

Written by hand rather than pulled from a library: the format is small, and
owning it means the hunk objects carry exactly what the reviewer needs
(new-file line numbers, added identifiers) instead of a generic tree that would
need translating anyway.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# @@ -oldStart,oldCount +newStart,newCount @@ optional section heading
_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")

# Identifiers worth searching for, ignoring the noise of short names.
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")

# Language keywords and builtins only. Generic-sounding nouns like `value`,
# `data`, and `item` are deliberately NOT here: they name real things, and
# `entry.value` is exactly the identifier a change to `entry.value` should be
# searched by. BM25's IDF already discounts terms that appear everywhere, which
# is a better instrument than a hand-written blocklist.
_STOP_IDENTIFIERS = frozenset({
    "and", "assert", "async", "await", "break", "class", "continue", "def",
    "del", "elif", "else", "except", "false", "finally", "for", "from",
    "global", "import", "lambda", "none", "nonlocal", "not", "pass", "raise",
    "return", "self", "true", "try", "while", "with", "yield",
    "int", "str", "bool", "float", "list", "dict", "set", "tuple", "bytes",
})


@dataclass(frozen=True)
class DiffLine:
    """One line of a hunk, with its line number on whichever sides it exists."""

    kind: str            # "add" | "remove" | "context"
    text: str
    old_line: int | None  # None for added lines
    new_line: int | None  # None for removed lines

    @property
    def marker(self) -> str:
        return {"add": "+", "remove": "-", "context": " "}[self.kind]


@dataclass(frozen=True)
class Hunk:
    """One @@ block: a contiguous change with its surrounding context lines."""

    file_path: str        # the new path, which is what comments anchor to
    old_start: int
    new_start: int
    heading: str          # the text after @@, usually the enclosing function
    lines: tuple[DiffLine, ...] = ()

    @property
    def added(self) -> list[DiffLine]:
        return [line for line in self.lines if line.kind == "add"]

    @property
    def removed(self) -> list[DiffLine]:
        return [line for line in self.lines if line.kind == "remove"]

    @property
    def new_line_range(self) -> tuple[int, int]:
        """First and last new-file line this hunk covers, added or context.

        A comment outside this range is anchored to code the reviewer was never
        shown, so this is what validation checks against.
        """
        numbers = [line.new_line for line in self.lines if line.new_line is not None]
        return (min(numbers), max(numbers)) if numbers else (self.new_start, self.new_start)

    @property
    def anchor_line(self) -> int:
        """Where a comment goes when the model does not say, or says something silly."""
        added = self.added
        return added[0].new_line if added else self.new_line_range[0]

    def identifiers(self) -> list[str]:
        """Distinct identifiers introduced or touched, for driving retrieval.

        Added lines first: what the change *introduces* is a better search than
        what it deletes, and ordering matters because the caller truncates.
        """
        found: dict[str, None] = {}
        for group in (self.added, self.removed):
            for line in group:
                for name in _IDENTIFIER.findall(line.text):
                    if name.lower() not in _STOP_IDENTIFIERS:
                        found.setdefault(name, None)
        return list(found)

    def render(self) -> str:
        """The hunk as it appears in a diff, with new-file line numbers attached.

        Numbers are shown because the model is asked to anchor comments to
        them; leaving it to infer them from an @@ header invites arithmetic
        mistakes that land comments on the wrong line.
        """
        out = [f"@@ {self.file_path} lines {self.new_line_range[0]}-{self.new_line_range[1]} @@"]
        for line in self.lines:
            number = f"{line.new_line:>5}" if line.new_line is not None else "     "
            out.append(f"{number} {line.marker}{line.text}")
        return "\n".join(out)


@dataclass
class FileDiff:
    """Every hunk touching one file, plus how the file itself changed."""

    path: str                 # new path; for a deletion, the old one
    old_path: str | None = None
    hunks: list[Hunk] = field(default_factory=list)
    is_new: bool = False
    is_deleted: bool = False
    is_binary: bool = False

    @property
    def is_rename(self) -> bool:
        return self.old_path is not None and self.old_path != self.path


def parse_diff(text: str) -> list[FileDiff]:
    """Parse a unified diff into per-file hunks.

    Tolerant of what real diffs carry around the parts we care about: `git
    --stat` preambles, mode changes, `index` lines, binary markers, and commit
    messages above the first `diff --git`. Anything unrecognised is skipped
    rather than raising, because a diff that mostly parses is far more useful
    than an exception.
    """
    files: list[FileDiff] = []
    current: FileDiff | None = None
    hunk_lines: list[DiffLine] = []
    hunk: Hunk | None = None
    old_no = new_no = 0

    def close_hunk() -> None:
        nonlocal hunk, hunk_lines
        if hunk is not None and current is not None:
            current.hunks.append(
                Hunk(
                    file_path=hunk.file_path,
                    old_start=hunk.old_start,
                    new_start=hunk.new_start,
                    heading=hunk.heading,
                    lines=tuple(hunk_lines),
                )
            )
        hunk, hunk_lines = None, []

    for raw in text.splitlines():
        if raw.startswith("diff --git "):
            close_hunk()
            current = FileDiff(path=_path_from_git_header(raw))
            files.append(current)
            continue

        if current is None:
            # Preamble: commit message, --stat output, anything before the first
            # file header. Nothing to do until a file is announced.
            continue

        if raw.startswith("new file mode"):
            current.is_new = True
        elif raw.startswith("deleted file mode"):
            current.is_deleted = True
        elif raw.startswith("Binary files ") or raw.startswith("GIT binary patch"):
            current.is_binary = True
        elif raw.startswith("rename from "):
            current.old_path = raw[len("rename from ") :].strip()
        elif raw.startswith("rename to "):
            current.path = raw[len("rename to ") :].strip()
        elif raw.startswith("--- "):
            old = _strip_prefix(raw[4:].strip())
            current.old_path = None if old == "/dev/null" else old
        elif raw.startswith("+++ "):
            new = _strip_prefix(raw[4:].strip())
            if new != "/dev/null":
                current.path = new
        elif (match := _HUNK_HEADER.match(raw)) is not None:
            close_hunk()
            old_no, new_no = int(match.group(1)), int(match.group(3))
            hunk = Hunk(
                file_path=current.path,
                old_start=old_no,
                new_start=new_no,
                heading=match.group(5).strip(),
            )
        elif hunk is not None:
            if raw.startswith("\\"):
                continue  # "\ No newline at end of file"
            if raw.startswith("+"):
                hunk_lines.append(DiffLine("add", raw[1:], None, new_no))
                new_no += 1
            elif raw.startswith("-"):
                hunk_lines.append(DiffLine("remove", raw[1:], old_no, None))
                old_no += 1
            elif raw.startswith(" ") or raw == "":
                hunk_lines.append(DiffLine("context", raw[1:], old_no, new_no))
                old_no += 1
                new_no += 1
            else:
                # Trailing junk after the last hunk (e.g. "-- \n2.39.0").
                close_hunk()

    close_hunk()
    return files


def _path_from_git_header(line: str) -> str:
    """Pull the new path out of `diff --git a/foo b/foo`.

    Falls back to the old path when the b/ side is missing, which happens in
    hand-edited diffs.
    """
    parts = line[len("diff --git ") :].split(" b/", 1)
    if len(parts) == 2:
        return parts[1].strip()
    return _strip_prefix(parts[0].strip())


def _strip_prefix(path: str) -> str:
    """Drop git's a/ and b/ prefixes and any trailing tab-separated timestamp."""
    path = path.split("\t", 1)[0].strip()
    for prefix in ("a/", "b/"):
        if path.startswith(prefix):
            return path[len(prefix) :]
    return path


def all_hunks(files: list[FileDiff]) -> list[Hunk]:
    """Every reviewable hunk, skipping files with nothing to read.

    Binary files and deletions are dropped: there is no new code to comment on,
    and asking a model to review a deletion in isolation produces noise about
    code that is no longer there.
    """
    return [h for f in files if not f.is_binary and not f.is_deleted for h in f.hunks]
