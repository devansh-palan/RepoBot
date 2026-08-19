"""Code-aware tokenization for the keyword index.

Prose tokenizers split on whitespace and punctuation, which leaves `parse_jwt`
and `parseJwt` as two opaque, unrelated terms — so a search for "parse jwt"
matches neither. Splitting identifiers into their parts is what makes BM25
useful on source code rather than merely present.
"""

from __future__ import annotations

import re

from src import config

# Runs of identifier characters. Everything else (punctuation, operators,
# whitespace) is a separator and is discarded.
_WORD = re.compile(r"[A-Za-z0-9_]+")

# One camelCase / PascalCase segment. The first branch takes a run of capitals
# not followed by a lowercase letter, which is what keeps acronyms whole:
# `parseHTTPResponse` -> parse, HTTP, Response rather than parse, H, T, T, P...
_SEGMENT = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+")


def split_identifier(word: str) -> list[str]:
    """Split one identifier into lowercase parts.

    `parse_jwt` and `parseJwt` both become ['parse', 'jwt'], which is the whole
    point: the two naming conventions become the same query.
    """
    parts: list[str] = []
    for piece in word.split("_"):
        parts.extend(_SEGMENT.findall(piece))
    return [part.lower() for part in parts]


def tokenize(text: str) -> list[str]:
    """Tokenize code or a question into BM25 terms.

    A multi-part identifier also emits its parts joined together, so an exact
    search for `parse_jwt` outranks a chunk that merely mentions `parse` and
    `jwt` separately. Joining rather than keeping the original spelling means
    `parse_jwt` and `parseJwt` produce the same `parsejwt` term and match each
    other exactly.
    """
    tokens: list[str] = []
    for word in _WORD.findall(text):
        parts = split_identifier(word)
        if len(parts) > 1:
            tokens.append("".join(parts))
        tokens.extend(parts)
    return [token for token in tokens if token not in config.BM25_STOPWORDS]
