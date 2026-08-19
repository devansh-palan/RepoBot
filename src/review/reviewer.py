"""Diff in, structured review comments out.

One model call per hunk rather than one for the whole diff. That costs more
calls, but it keeps each prompt small enough to fit a local model's context,
gives every hunk its own retrieval, and means one hunk's malformed reply cannot
take the rest of the review down with it.
"""

from __future__ import annotations

from pathlib import Path

from src import config
from src.agent import get_provider
from src.agent.llm import Provider

from .context import HunkContext, gather
from .diff import FileDiff, all_hunks, parse_diff
from .models import ReviewComment, ReviewResult
from .prompts import REVIEW_SYSTEM_PROMPT, build_review_prompt, parse_comments


def review_diff(
    diff_text: str,
    repo_path: str | Path,
    provider: Provider | None = None,
    max_hunks: int | None = None,
    persist_dir: str | None = None,
    bm25_dir: str | Path | None = None,
) -> ReviewResult:
    """Review a unified diff against the repository it applies to."""
    files = parse_diff(diff_text)
    return review_files(
        files, repo_path, provider, max_hunks, persist_dir, bm25_dir
    )


def review_files(
    files: list[FileDiff],
    repo_path: str | Path,
    provider: Provider | None = None,
    max_hunks: int | None = None,
    persist_dir: str | None = None,
    bm25_dir: str | Path | None = None,
) -> ReviewResult:
    """Review already-parsed files. Split out so tests can build hunks directly."""
    provider = provider or get_provider()
    result = ReviewResult()

    for file in files:
        if file.is_binary:
            result.skipped.append(f"{file.path}: binary")
        elif file.is_deleted:
            result.skipped.append(f"{file.path}: deleted")

    hunks = all_hunks(files)
    result.files_reviewed = len({h.file_path for h in hunks})

    if max_hunks is not None and len(hunks) > max_hunks:
        result.skipped.append(
            f"{len(hunks) - max_hunks} hunks beyond --max-hunks={max_hunks}"
        )
        hunks = hunks[:max_hunks]

    for hunk in hunks:
        context = gather(hunk, repo_path, persist_dir, bm25_dir)
        comments, problems = review_hunk(context, provider, result)
        result.comments.extend(comments)
        result.hunks_reviewed += 1
        result.skipped.extend(f"{hunk.file_path}:{hunk.anchor_line}: {p}" for p in problems)

    result.comments = _deduplicate(result.comments)
    return result


def review_hunk(
    context: HunkContext,
    provider: Provider,
    result: ReviewResult | None = None,
) -> tuple[list[ReviewComment], list[str]]:
    """Ask for a review of one hunk and parse the reply.

    A hunk with no added or removed lines is skipped without a model call —
    context-only hunks appear in diffs with large `-U` settings and there is
    nothing in them to review.
    """
    hunk = context.hunk
    if not hunk.added and not hunk.removed:
        return [], []

    response = provider.complete(
        system=REVIEW_SYSTEM_PROMPT,
        user=build_review_prompt(context.render()),
    )

    if result is not None:
        result.model = response.model
        result.input_tokens += response.input_tokens
        result.output_tokens += response.output_tokens

    return parse_comments(
        response.text,
        file_path=hunk.file_path,
        valid_lines=hunk.new_line_range,
        fallback_line=hunk.anchor_line,
    )


def _deduplicate(comments: list[ReviewComment]) -> list[ReviewComment]:
    """Drop repeats of the same point on the same line.

    Adjacent hunks often retrieve the same related code and draw the same
    conclusion, and a reviewer reading the same sentence twice loses trust in
    all of it. Keyed on the opening of the message so near-identical wording
    still collapses.
    """
    seen: set[tuple[str, int, str]] = set()
    unique = []
    for comment in comments:
        key = (comment.file, comment.line, comment.message[:60].lower())
        if key not in seen:
            seen.add(key)
            unique.append(comment)
    return unique


def review_diff_file(
    diff_path: str | Path,
    repo_path: str | Path,
    provider: Provider | None = None,
    max_hunks: int | None = None,
) -> ReviewResult:
    """Convenience wrapper for the CLI: read a .patch/.diff file and review it."""
    text = Path(diff_path).read_text(encoding="utf-8", errors="replace")
    return review_diff(text, repo_path, provider, max_hunks)
