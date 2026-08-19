"""The review's output type.

One comment is (file, line, severity, message) — the shape a code host expects
for an inline comment, so the result can be posted as-is later rather than
reshaped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

# Ordered worst-first: the reviewer sorts by this, and a caller can threshold on
# it ("only show me warnings and above") by slicing the list.
SEVERITIES: tuple[str, ...] = ("blocker", "warning", "suggestion", "note")
SEVERITY_RANK = {name: rank for rank, name in enumerate(SEVERITIES)}

# What each level is for, included verbatim in the prompt so the model and the
# reader mean the same thing by them.
SEVERITY_GUIDE = {
    "blocker": "a real defect: wrong behaviour, a crash, a security or data-loss risk",
    "warning": "likely to cause a problem — a missed edge case, a swallowed error",
    "suggestion": "a clear improvement that is not a defect",
    "note": "an observation worth reading, no action implied",
}


@dataclass(frozen=True)
class ReviewComment:
    """One inline comment, anchored to a line of the new file."""

    file: str
    line: int
    severity: str
    message: str

    def __post_init__(self) -> None:
        if self.severity not in SEVERITY_RANK:
            raise ValueError(f"severity {self.severity!r} not in {SEVERITIES}")

    @property
    def location(self) -> str:
        return f"{self.file}:{self.line}"

    def to_dict(self) -> dict[str, str | int]:
        return {
            "file": self.file,
            "line": self.line,
            "severity": self.severity,
            "message": self.message,
        }

    def __str__(self) -> str:
        return f"{self.severity.upper():<10} {self.location}  {self.message}"


@dataclass
class ReviewResult:
    """Every comment on a diff, plus what it cost to produce."""

    comments: list[ReviewComment] = field(default_factory=list)
    hunks_reviewed: int = 0
    files_reviewed: int = 0
    skipped: list[str] = field(default_factory=list)
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0

    def sorted(self) -> list[ReviewComment]:
        """Worst first, then by position, so the important thing is read first."""
        return sorted(
            self.comments,
            key=lambda c: (SEVERITY_RANK[c.severity], c.file, c.line),
        )

    def of_severity(self, *levels: str) -> list[ReviewComment]:
        return [c for c in self.sorted() if c.severity in levels]

    @property
    def blocking(self) -> list[ReviewComment]:
        return self.of_severity("blocker")

    def counts(self) -> dict[str, int]:
        return {s: sum(1 for c in self.comments if c.severity == s) for s in SEVERITIES}

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(
            {
                "summary": {
                    "files": self.files_reviewed,
                    "hunks": self.hunks_reviewed,
                    "comments": len(self.comments),
                    "by_severity": self.counts(),
                    "model": self.model,
                },
                "skipped": self.skipped,
                "comments": [c.to_dict() for c in self.sorted()],
            },
            indent=indent,
        )

    def summary(self) -> str:
        counts = ", ".join(f"{n} {s}" for s, n in self.counts().items() if n)
        return (
            f"{len(self.comments)} comments ({counts or 'none'}) "
            f"over {self.hunks_reviewed} hunks in {self.files_reviewed} files"
        )
