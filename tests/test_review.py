"""Diff parsing, context gathering, and review-comment extraction.

The parser tests carry the most weight: a comment anchored to the wrong line
renders against unrelated code, which is worse than no comment at all. Model
replies are scripted so parsing is checked against real failure modes — prose
around the JSON, invented line numbers, unknown severities — rather than only
against well-formed output.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from src import config
from src.agent import LLMResponse
from src.index import index_repo
from src.review import (
    Hunk,
    ReviewComment,
    ReviewResult,
    all_hunks,
    gather,
    parse_comments,
    parse_diff,
    review_diff,
    search_query,
)
from src.review.diff import DiffLine

SIMPLE_DIFF = """\
diff --git a/src/pay/billing.py b/src/pay/billing.py
index 1234567..89abcde 100644
--- a/src/pay/billing.py
+++ b/src/pay/billing.py
@@ -10,7 +10,8 @@ class Ledger:
     def total(self):
         amount = 0
         for entry in self.entries:
-            amount += entry.value
+            if entry.value is not None:
+                amount += entry.value
         return amount

     def close(self):
"""


def test_a_simple_hunk_parses_with_correct_line_numbers() -> None:
    """The property everything else depends on."""
    files = parse_diff(SIMPLE_DIFF)
    assert len(files) == 1
    assert files[0].path == "src/pay/billing.py"

    hunk = files[0].hunks[0]
    assert hunk.new_start == 10
    assert [line.new_line for line in hunk.added] == [13, 14]
    assert [line.text for line in hunk.added] == [
        "            if entry.value is not None:",
        "                amount += entry.value",
    ]
    assert [line.old_line for line in hunk.removed] == [13]


def test_new_line_range_spans_context_as_well_as_additions() -> None:
    hunk = parse_diff(SIMPLE_DIFF)[0].hunks[0]
    low, high = hunk.new_line_range
    assert low == 10
    assert high == 17
    assert hunk.anchor_line == 13, "comments default to the first added line"


def test_the_hunk_heading_is_kept() -> None:
    """git puts the enclosing symbol there; it is free context for the prompt."""
    assert parse_diff(SIMPLE_DIFF)[0].hunks[0].heading == "class Ledger:"


def test_rendered_hunk_shows_line_numbers_and_markers() -> None:
    """The model anchors comments to these numbers, so they must be visible."""
    rendered = parse_diff(SIMPLE_DIFF)[0].hunks[0].render()
    assert "   13 +            if entry.value is not None:" in rendered
    assert "src/pay/billing.py lines 10-17" in rendered


# --------------------------------------------------------------------------
# Harder diffs
# --------------------------------------------------------------------------


def test_multiple_hunks_in_one_file_each_track_their_own_numbering() -> None:
    diff = """\
diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -1,3 +1,4 @@
 import os
+import sys

 x = 1
@@ -50,3 +51,4 @@
 def far_away():
+    pass
     return None
"""
    hunks = parse_diff(diff)[0].hunks
    assert len(hunks) == 2
    assert hunks[0].added[0].new_line == 2
    assert hunks[1].added[0].new_line == 52


def test_multiple_files_are_separated() -> None:
    diff = SIMPLE_DIFF + """\
diff --git a/src/pay/tax.py b/src/pay/tax.py
--- a/src/pay/tax.py
+++ b/src/pay/tax.py
@@ -1,2 +1,3 @@
 RATE = 0.2
+VAT = 0.05
"""
    files = parse_diff(diff)
    assert [f.path for f in files] == ["src/pay/billing.py", "src/pay/tax.py"]
    assert len(all_hunks(files)) == 2


def test_a_new_file_is_marked_and_still_reviewable() -> None:
    diff = """\
diff --git a/new.py b/new.py
new file mode 100644
index 0000000..1234567
--- /dev/null
+++ b/new.py
@@ -0,0 +1,2 @@
+def added():
+    return 1
"""
    files = parse_diff(diff)
    assert files[0].is_new and files[0].old_path is None
    assert files[0].path == "new.py"
    assert [line.new_line for line in files[0].hunks[0].added] == [1, 2]
    assert len(all_hunks(files)) == 1, "a new file is the most worth reviewing"


def test_a_deleted_file_is_not_reviewed() -> None:
    """There is no new code to comment on, and comments would point at nothing."""
    diff = """\
diff --git a/gone.py b/gone.py
deleted file mode 100644
--- a/gone.py
+++ /dev/null
@@ -1,2 +0,0 @@
-def removed():
-    return 1
"""
    files = parse_diff(diff)
    assert files[0].is_deleted
    assert all_hunks(files) == []


def test_a_binary_file_is_not_reviewed() -> None:
    diff = """\
diff --git a/logo.png b/logo.png
index 1234567..89abcde 100644
Binary files a/logo.png and b/logo.png differ
"""
    files = parse_diff(diff)
    assert files[0].is_binary
    assert all_hunks(files) == []


def test_a_rename_keeps_both_paths() -> None:
    diff = """\
diff --git a/old_name.py b/new_name.py
similarity index 90%
rename from old_name.py
rename to new_name.py
--- a/old_name.py
+++ b/new_name.py
@@ -1,2 +1,2 @@
-x = 1
+x = 2
"""
    file = parse_diff(diff)[0]
    assert file.path == "new_name.py"
    assert file.old_path == "old_name.py"
    assert file.is_rename


def test_hunk_header_without_counts_is_understood() -> None:
    """`@@ -1 +1 @@` is legal and means a count of one."""
    diff = """\
diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -1 +1 @@
-x = 1
+x = 2
"""
    hunk = parse_diff(diff)[0].hunks[0]
    assert hunk.added[0].new_line == 1


def test_no_newline_marker_is_ignored() -> None:
    diff = """\
diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -1,1 +1,1 @@
-x = 1
\\ No newline at end of file
+x = 2
"""
    hunk = parse_diff(diff)[0].hunks[0]
    assert len(hunk.added) == 1 and hunk.added[0].new_line == 1


def test_a_commit_message_preamble_is_skipped() -> None:
    """`git format-patch` output starts with prose, not a file header."""
    diff = "From abc123 Mon Sep 17\nSubject: [PATCH] fix a thing\n\n---\n a.py | 2 +-\n\n" + SIMPLE_DIFF
    files = parse_diff(diff)
    assert len(files) == 1 and files[0].hunks


def test_a_git_signature_footer_does_not_corrupt_the_last_hunk() -> None:
    files = parse_diff(SIMPLE_DIFF + "-- \n2.39.0\n")
    assert len(files[0].hunks) == 1
    assert len(files[0].hunks[0].added) == 2


def test_an_empty_diff_parses_to_nothing() -> None:
    assert parse_diff("") == []
    assert parse_diff("just some text\nnot a diff\n") == []


# --------------------------------------------------------------------------
# Retrieval query
# --------------------------------------------------------------------------


def test_identifiers_come_from_the_change_not_the_context() -> None:
    hunk = parse_diff(SIMPLE_DIFF)[0].hunks[0]
    names = hunk.identifiers()
    assert "entry" in names and "value" in names
    assert "close" not in names, "close() is context, not part of the change"


def test_keywords_and_builtins_are_not_searched_for() -> None:
    hunk = parse_diff(SIMPLE_DIFF)[0].hunks[0]
    assert "not" not in hunk.identifiers()
    assert "if" not in hunk.identifiers()


def test_search_query_falls_back_when_a_hunk_has_no_identifiers() -> None:
    bare = Hunk(
        file_path="src/pay/billing.py",
        old_start=1,
        new_start=1,
        heading="def total",
        lines=(DiffLine("add", "    +1", None, 1),),
    )
    assert search_query(bare) in {"def total", "billing"}


# --------------------------------------------------------------------------
# Parsing model replies
# --------------------------------------------------------------------------

VALID = (10, 20)


def test_a_clean_json_reply_parses() -> None:
    reply = '[{"line": 13, "severity": "warning", "message": "Silently skips None."}]'
    comments, problems = parse_comments(reply, "a.py", VALID, 13)
    assert problems == []
    assert comments == [ReviewComment("a.py", 13, "warning", "Silently skips None.")]


def test_json_wrapped_in_prose_and_fences_still_parses() -> None:
    """What a small local model actually returns."""
    reply = (
        "Sure! Here is my review:\n\n```json\n"
        '[{"line": 14, "severity": "suggestion", "message": "Consider a guard clause."}]\n'
        "```\nHope that helps."
    )
    comments, _ = parse_comments(reply, "a.py", VALID, 13)
    assert len(comments) == 1 and comments[0].line == 14


def test_an_empty_array_is_a_clean_review_not_a_failure() -> None:
    comments, problems = parse_comments("[]", "a.py", VALID, 13)
    assert comments == [] and problems == []


def test_a_line_outside_the_hunk_is_anchored_rather_than_dropped() -> None:
    """A model will confidently cite a line it was never shown."""
    reply = '[{"line": 900, "severity": "blocker", "message": "Off by one."}]'
    comments, problems = parse_comments(reply, "a.py", VALID, 13)
    assert comments[0].line == 13
    assert "outside hunk" in problems[0]


def test_an_unknown_severity_degrades_to_note() -> None:
    reply = '[{"line": 13, "severity": "nitpick", "message": "Naming."}]'
    comments, problems = parse_comments(reply, "a.py", VALID, 13)
    assert comments[0].severity == "note"
    assert "unknown severity" in problems[0]


def test_a_comment_with_no_message_is_dropped() -> None:
    reply = '[{"line": 13, "severity": "warning", "message": ""}, ' \
            '{"line": 14, "severity": "note", "message": "Real one."}]'
    comments, problems = parse_comments(reply, "a.py", VALID, 13)
    assert len(comments) == 1, "one bad entry must not lose the good one"
    assert problems


def test_a_missing_line_number_falls_back_to_the_anchor() -> None:
    reply = '[{"severity": "note", "message": "General remark."}]'
    comments, _ = parse_comments(reply, "a.py", VALID, 13)
    assert comments[0].line == 13


def test_an_unparseable_reply_yields_no_comments_and_says_so() -> None:
    comments, problems = parse_comments("I could not review this.", "a.py", VALID, 13)
    assert comments == []
    assert problems == ["no JSON array in the reply"]


def test_an_invalid_severity_on_the_dataclass_is_rejected() -> None:
    with pytest.raises(ValueError, match="severity"):
        ReviewComment("a.py", 1, "catastrophic", "boom")


# --------------------------------------------------------------------------
# Result shaping
# --------------------------------------------------------------------------


def test_comments_sort_worst_first_then_by_position() -> None:
    result = ReviewResult(comments=[
        ReviewComment("b.py", 5, "note", "n"),
        ReviewComment("a.py", 9, "blocker", "b"),
        ReviewComment("a.py", 2, "blocker", "a"),
        ReviewComment("a.py", 1, "warning", "w"),
    ])
    assert [(c.severity, c.file, c.line) for c in result.sorted()] == [
        ("blocker", "a.py", 2),
        ("blocker", "a.py", 9),
        ("warning", "a.py", 1),
        ("note", "b.py", 5),
    ]


def test_counts_and_blocking_are_reported() -> None:
    result = ReviewResult(comments=[
        ReviewComment("a.py", 1, "blocker", "x"),
        ReviewComment("a.py", 2, "note", "y"),
    ])
    assert result.counts()["blocker"] == 1
    assert len(result.blocking) == 1
    assert "1 blocker" in result.summary()


def test_result_serialises_to_json() -> None:
    import json

    result = ReviewResult(comments=[ReviewComment("a.py", 1, "warning", "x")], hunks_reviewed=1)
    payload = json.loads(result.to_json())
    assert payload["comments"][0] == {
        "file": "a.py", "line": 1, "severity": "warning", "message": "x"
    }
    assert payload["summary"]["hunks"] == 1


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------

REPO_FILE = '''"""Ledger."""


class Ledger:
    """Tracks entries and totals them."""

    def __init__(self, entries):
        self.entries = entries

    def total(self):
        amount = 0
        for entry in self.entries:
            if entry.value is not None:
                amount += entry.value
        return amount

    def close(self):
        self.entries = []
'''


@pytest.fixture(scope="module")
def repo(tmp_path_factory: pytest.TempPathFactory) -> dict:
    root = tmp_path_factory.mktemp("review_repo")
    (root / "billing.py").write_text(REPO_FILE, encoding="utf-8")
    (root / "report.py").write_text(
        "from billing import Ledger\n\n\n"
        "def render(ledger):\n"
        '    """Callers assume total() never returns None."""\n'
        "    return f'{ledger.total():.2f}'\n",
        encoding="utf-8",
    )
    base = tmp_path_factory.mktemp("review_idx")
    dirs = {"persist_dir": str(base / "chroma"), "bm25_dir": str(base / "bm25")}
    index_repo(root, cache_dir=base / "cache", **dirs)
    return {"root": root, **dirs}


class ScriptedProvider:
    name = model = "scripted"

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def complete(self, system, user, model=None, max_tokens=0, on_text=None) -> LLMResponse:
        self.prompts.append(user)
        text = self.replies.pop(0) if self.replies else "[]"
        return LLMResponse(text=text, model=self.model, input_tokens=7, output_tokens=3)


def test_gather_collects_surrounding_and_related_code(repo: dict) -> None:
    """The point of reusing the agent's tools: cross-file callers get found."""
    diff = SIMPLE_DIFF.replace("src/pay/billing.py", "billing.py")
    hunk = parse_diff(diff)[0].hunks[0]

    context = gather(hunk, repo["root"], repo["persist_dir"], repo["bm25_dir"])

    assert "def total" in context.surrounding
    assert context.related, "should find code elsewhere that mentions the change"
    assert all(r.chunk.file_path != "billing.py" for r in context.related), (
        "the changed file is already in `surrounding`; related must be elsewhere"
    )
    assert "### Related code elsewhere" in context.render()


def test_gather_survives_a_file_that_is_not_in_the_tree(repo: dict) -> None:
    """A diff from another branch should still be reviewable from the diff alone."""
    hunk = parse_diff(SIMPLE_DIFF)[0].hunks[0]  # path src/pay/billing.py does not exist
    context = gather(hunk, repo["root"], repo["persist_dir"], repo["bm25_dir"])
    assert context.surrounding == ""
    assert any("no file context" in note for note in context.notes)
    assert context.render(), "the hunk itself is still reviewable"


def test_review_diff_produces_anchored_comments(repo: dict) -> None:
    provider = ScriptedProvider(
        '[{"line": 13, "severity": "warning", '
        '"message": "Skipping None silently changes total() semantics for callers."}]'
    )
    diff = SIMPLE_DIFF.replace("src/pay/billing.py", "billing.py")

    result = review_diff(
        diff, repo["root"], provider=provider,
        persist_dir=repo["persist_dir"], bm25_dir=repo["bm25_dir"],
    )

    assert result.hunks_reviewed == 1
    assert result.files_reviewed == 1
    assert len(result.comments) == 1
    assert result.comments[0].location == "billing.py:13"
    assert result.comments[0].severity == "warning"
    assert result.input_tokens == 7


def test_the_prompt_contains_the_diff_and_the_related_code(repo: dict) -> None:
    provider = ScriptedProvider("[]")
    diff = SIMPLE_DIFF.replace("src/pay/billing.py", "billing.py")
    review_diff(diff, repo["root"], provider=provider,
                persist_dir=repo["persist_dir"], bm25_dir=repo["bm25_dir"])

    prompt = provider.prompts[0]
    assert "if entry.value is not None:" in prompt
    assert "around the change" in prompt
    assert "   13 +" in prompt, "line numbers must reach the model"


def test_one_call_per_hunk(repo: dict) -> None:
    """Small prompts, independent retrieval, and one bad reply cannot sink the rest."""
    diff = SIMPLE_DIFF.replace("src/pay/billing.py", "billing.py") + """\
diff --git a/report.py b/report.py
--- a/report.py
+++ b/report.py
@@ -4,3 +4,3 @@
 def render(ledger):
-    return f'{ledger.total():.2f}'
+    return f'{ledger.total()}'
"""
    provider = ScriptedProvider("garbage, not json", '[{"line": 6, "severity": "note", "message": "ok"}]')
    result = review_diff(diff, repo["root"], provider=provider,
                         persist_dir=repo["persist_dir"], bm25_dir=repo["bm25_dir"])

    assert len(provider.prompts) == 2
    assert result.hunks_reviewed == 2
    assert len(result.comments) == 1, "the good hunk survives the bad one"
    assert any("no JSON array" in s for s in result.skipped)


def test_max_hunks_stops_early_and_says_so(repo: dict) -> None:
    diff = SIMPLE_DIFF.replace("src/pay/billing.py", "billing.py") + """\
diff --git a/report.py b/report.py
--- a/report.py
+++ b/report.py
@@ -4,3 +4,3 @@
 def render(ledger):
-    return f'{ledger.total():.2f}'
+    return f'{ledger.total()}'
"""
    provider = ScriptedProvider("[]", "[]")
    result = review_diff(diff, repo["root"], provider=provider, max_hunks=1,
                         persist_dir=repo["persist_dir"], bm25_dir=repo["bm25_dir"])

    assert result.hunks_reviewed == 1
    assert len(provider.prompts) == 1
    assert any("beyond --max-hunks" in s for s in result.skipped)


def test_identical_comments_are_deduplicated(repo: dict) -> None:
    same = '[{"line": 13, "severity": "note", "message": "The same observation twice."}]'
    diff = SIMPLE_DIFF.replace("src/pay/billing.py", "billing.py")
    result = review_diff(diff + diff, repo["root"], provider=ScriptedProvider(same, same),
                         persist_dir=repo["persist_dir"], bm25_dir=repo["bm25_dir"])
    assert len(result.comments) == 1
