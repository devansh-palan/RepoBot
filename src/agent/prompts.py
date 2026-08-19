"""Prompt construction for grounded Q&A.

The whole value of RAG is that the answer is traceable to retrieved code, so
these prompts do two jobs: forbid outside knowledge, and make citation the
default rather than a request. Prompt text lives here, apart from the retrieval
and call logic, so it can be diffed and ablated on its own.
"""

from __future__ import annotations

from collections.abc import Sequence

from src.index import SearchResult

SYSTEM_PROMPT = """\
You are a code comprehension assistant. You answer questions about a codebase \
using only the source code excerpts provided to you.

Rules:

1. Answer only from the provided excerpts. They are the entire evidence you \
have. Do not use general knowledge about libraries, frameworks, or common \
patterns to fill a gap, and do not guess at code you cannot see.

2. Cite every claim with a file path and line range, like \
`src/ingest/walker.py:13-38`. Put the citation directly after the claim it \
supports, not collected at the end. A sentence about the code without a \
citation is an error.

3. If the excerpts do not contain the answer, say so plainly and name what is \
missing — for example, "the excerpts show where chunks are written but not \
where they are deleted." A partial answer with an explicit gap is far more \
useful than a confident guess. Do not pad it with what the code might do.

4. Quote code only when the exact text matters, and keep quotes short. Prefer \
explaining what the code does and citing where it lives.

5. Match the length of the answer to the question. A question about one \
function deserves a few sentences, not an essay.
"""


def format_context(results: Sequence[SearchResult]) -> str:
    """Render retrieved chunks as numbered, citable excerpts.

    Each excerpt is labelled with the exact `path:start-end` string the model is
    asked to cite, so citing correctly is copying rather than constructing.
    """
    blocks = []
    for n, result in enumerate(results, start=1):
        chunk = result.chunk
        blocks.append(
            f"[{n}] {chunk.location}  {chunk.kind} {chunk.symbol}  ({chunk.language})\n"
            f"```\n{chunk.code}\n```"
        )
    return "\n\n".join(blocks)


def build_user_prompt(question: str, results: Sequence[SearchResult]) -> str:
    """Assemble the excerpts and the question into one user turn.

    Question last: it is the volatile part, and keeping it after the excerpts
    leaves the long prefix stable if prompt caching is added later.
    """
    return (
        f"Here are {len(results)} excerpts retrieved from the codebase, "
        f"most relevant first.\n\n"
        f"{format_context(results)}\n\n"
        f"---\n\n"
        f"Question: {question}"
    )


NO_CONTEXT_ANSWER = (
    "Retrieval found no code for that question. The repository may not be "
    "indexed, or the question may be about code that is not in it."
)


REFLECTION_PROMPT = """\
You are checking whether an answer about a codebase is fully supported by the \
code excerpts it was given. You are not judging whether the answer is well \
written or whether it is true of the wider world — only whether the excerpts \
below actually support every claim it makes.

Reply with exactly three lines, nothing else:

GROUNDED: yes|no
REASON: <one sentence>
QUERY: <a different search query that would find the missing code, or NONE>

Mark it `no` when the answer states something the excerpts do not show — \
behaviour that is described but not visible, a function said to call another \
when no such call appears, or a claim carrying a citation that does not contain \
it. A partial answer that clearly says what it could not determine is \
`yes`: admitting a gap is not the same as inventing a fact.

The QUERY line matters. If the answer is not grounded, name what is missing in \
the words the code itself would use — a likely function or symbol name — not a \
rephrasing of the original question. Searching the same thing again returns the \
same excerpts and the retry is wasted.
"""


def build_reflection_prompt(question: str, answer: str, context: str) -> str:
    """Assemble the critic's turn: the evidence, the answer, and the question."""
    return (
        f"Excerpts the answer was given:\n\n{context}\n\n"
        f"---\n\nQuestion asked: {question}\n\n"
        f"---\n\nAnswer to check:\n\n{answer}"
    )


RETRY_NOTE = """\

An earlier attempt at this question was rejected as not fully grounded:
  {reason}
Additional excerpts have been retrieved since. Answer again using everything \
below, and leave out any claim the excerpts still do not support.
"""


# Distinct from RETRY_NOTE on purpose. When the only failure was missing
# citations, the content was fine — telling the model to "leave out unsupported
# claims" makes it hedge a correct answer into a thinner one (measured: the
# generic note cost explanatory accuracy while fixing factual). This note asks
# for the same answer with its receipts attached, nothing less.
CITE_RETRY_NOTE = """\

An earlier attempt at this question was rejected for one reason only: it made \
claims without citing any file and line, so nothing could be checked. The \
content itself was not judged wrong. Give the same answer again, this time \
attaching a citation like `src/ingest/walker.py:13-38` directly after each \
claim, copied from the excerpt labels above. Do not shorten the answer or drop \
claims — cite them.
"""
