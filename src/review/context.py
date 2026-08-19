"""Gathering the code a hunk needs to be judged against.

A diff hunk on its own is close to unreviewable. `if not x: return None` might
be a fix or a regression depending on what the caller expects, and no amount of
staring at the hunk resolves it. So each hunk is paired with two kinds of code,
both fetched with the agent's existing tools:

* **surrounding** — the rest of the file around the change, via `read_file`.
  This is what answers "what does this function actually do".
* **related** — chunks elsewhere in the repo that mention the same identifiers,
  via `search_code`. This is what answers "who calls this, and what else
  implements the same thing".

Related code is the reason a retrieval system is worth having here at all: the
caller that breaks is usually in a different file from the change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src import config
from src.agent import read_file, search_code
from src.index import SearchResult

from .diff import Hunk

# Lines of the file to show either side of the hunk. Enough to see the whole
# enclosing function in most cases without burying the diff itself.
SURROUNDING_LINES = 40

# Identifiers from the hunk used to drive the related-code search. More than a
# handful and the query stops being about the change and starts matching
# everything.
MAX_QUERY_IDENTIFIERS = 6

RELATED_CHUNKS = 4


@dataclass
class HunkContext:
    """One hunk with everything gathered to review it."""

    hunk: Hunk
    surrounding: str = ""
    related: list[SearchResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        """The whole evidence package for one hunk, as the model sees it."""
        parts = [f"### Change to {self.hunk.file_path}", "", self.hunk.render()]

        if self.surrounding:
            parts += ["", f"### {self.hunk.file_path} around the change", "```",
                      self.surrounding, "```"]

        if self.related:
            parts += ["", "### Related code elsewhere in the repository"]
            for result in self.related:
                parts += [
                    "",
                    f"{result.chunk.location}  {result.chunk.kind} {result.chunk.symbol}",
                    "```",
                    result.chunk.code,
                    "```",
                ]
        return "\n".join(parts)


def search_query(hunk: Hunk) -> str:
    """A retrieval query built from what the hunk touches.

    Identifiers rather than prose: the useful related code is whatever else
    mentions these names, and a natural-language summary of the change would
    only be a lossy paraphrase of them.
    """
    names = hunk.identifiers()[:MAX_QUERY_IDENTIFIERS]
    return " ".join(names) or hunk.heading or Path(hunk.file_path).stem


def gather(
    hunk: Hunk,
    repo_path: str | Path,
    persist_dir: str | None = None,
    bm25_dir: str | Path | None = None,
) -> HunkContext:
    """Collect surrounding and related code for one hunk.

    Both lookups fail soft. A hunk against a file that is not in the working
    tree — a diff from another branch, or a new file not yet written — should
    still be reviewed from the diff alone rather than aborting the run.
    """
    context = HunkContext(hunk=hunk)
    start, end = hunk.new_line_range

    try:
        sliced = read_file(
            repo_path,
            hunk.file_path,
            max(1, start - SURROUNDING_LINES),
            end + SURROUNDING_LINES,
        )
        context.surrounding = sliced.text
    except (FileNotFoundError, ValueError, OSError) as exc:
        context.notes.append(f"no file context: {exc}")

    try:
        query = search_query(hunk)
        found = search_code(repo_path, query, RELATED_CHUNKS, persist_dir, bm25_dir)
        # Chunks from the changed file itself are already in `surrounding`;
        # repeating them crowds out the cross-file code that is the point.
        context.related = [r for r in found if r.chunk.file_path != hunk.file_path]
    except Exception as exc:  # noqa: BLE001 - retrieval is best-effort here
        context.notes.append(f"no related code: {exc}")

    return context


def gather_all(
    hunks: list[Hunk],
    repo_path: str | Path,
    persist_dir: str | None = None,
    bm25_dir: str | Path | None = None,
) -> list[HunkContext]:
    return [gather(h, repo_path, persist_dir, bm25_dir) for h in hunks]
