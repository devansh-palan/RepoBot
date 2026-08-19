"""The reviewer's prompt and the parser for what comes back.

Output is JSON because a review is a list of records, not prose. Parsing is
deliberately forgiving: a 7B local model produces *nearly* valid JSON often
enough that failing the whole hunk on one malformed entry would throw away good
comments. Each entry is validated on its own and a bad one is dropped.
"""

from __future__ import annotations

import json
import re

from .models import SEVERITY_GUIDE, ReviewComment

_SEVERITY_LINES = "\n".join(f"  {name}: {why}" for name, why in SEVERITY_GUIDE.items())

REVIEW_SYSTEM_PROMPT = f"""\
You are reviewing one hunk of a code change. You are given the diff, the code \
around it, and related code from elsewhere in the repository.

Report only problems you can point at in the code you were shown. The related \
code is there so you can check whether a change breaks a caller or contradicts \
how the same thing is done elsewhere — use it.

Severity levels:
{_SEVERITY_LINES}

Reply with a JSON array and nothing else. Each element:

  {{"line": <int>, "severity": "<level>", "message": "<one or two sentences>"}}

`line` must be a line number shown in the diff, and it must be a line the \
change added or touched — not a line from the surrounding context, and not a \
line from the related code.

Return `[]` when the hunk is fine. That is a normal outcome and a far better \
answer than inventing a nitpick. Do not comment on style, formatting, or naming \
unless it causes a real problem. Do not restate what the diff does.
"""


def build_review_prompt(rendered_context: str) -> str:
    return f"{rendered_context}\n\n---\n\nReview the change above."


def parse_comments(
    text: str,
    file_path: str,
    valid_lines: tuple[int, int],
    fallback_line: int,
) -> tuple[list[ReviewComment], list[str]]:
    """Turn a model reply into comments, dropping anything unusable.

    Returns (comments, problems). Line numbers are checked against the hunk's
    range because a model will confidently anchor a comment to a line it was
    never shown; such a comment would render against unrelated code and is worse
    than no comment at all. A line just outside the range is snapped to the
    hunk's anchor rather than dropped, since the observation is usually still
    about this change.
    """
    payload = _extract_json_array(text)
    if payload is None:
        return [], ["no JSON array in the reply"]

    low, high = valid_lines
    comments: list[ReviewComment] = []
    problems: list[str] = []

    for entry in payload:
        if not isinstance(entry, dict):
            problems.append(f"not an object: {entry!r:.60}")
            continue

        message = str(entry.get("message", "")).strip()
        severity = str(entry.get("severity", "")).strip().lower()
        if not message:
            problems.append("entry with no message")
            continue
        if severity not in SEVERITY_GUIDE:
            problems.append(f"unknown severity {severity!r}, treating as a note")
            severity = "note"

        try:
            line = int(entry.get("line", fallback_line))
        except (TypeError, ValueError):
            line = fallback_line

        if not low <= line <= high:
            problems.append(f"line {line} outside hunk {low}-{high}, anchored to {fallback_line}")
            line = fallback_line

        comments.append(
            ReviewComment(file=file_path, line=line, severity=severity, message=message)
        )

    return comments, problems


def _extract_json_array(text: str) -> list | None:
    """Find the JSON array in a reply that may be wrapped in prose or fences."""
    text = text.strip()

    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        return None

    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, list) else None
