"""Diff and PR review mode.

Parses a unified diff into hunks, gathers surrounding and related code for each
using the agent's tools, and emits structured (file, line, severity, message)
comments.
"""

from .context import HunkContext, gather, gather_all, search_query
from .diff import DiffLine, FileDiff, Hunk, all_hunks, parse_diff
from .models import SEVERITIES, ReviewComment, ReviewResult
from .prompts import REVIEW_SYSTEM_PROMPT, build_review_prompt, parse_comments
from .reviewer import review_diff, review_diff_file, review_files, review_hunk

__all__ = [
    "REVIEW_SYSTEM_PROMPT",
    "SEVERITIES",
    "DiffLine",
    "FileDiff",
    "Hunk",
    "HunkContext",
    "ReviewComment",
    "ReviewResult",
    "all_hunks",
    "build_review_prompt",
    "gather",
    "gather_all",
    "parse_comments",
    "parse_diff",
    "review_diff",
    "review_diff_file",
    "review_files",
    "review_hunk",
    "search_query",
]
