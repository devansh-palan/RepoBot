"""Cloning, file walking, and tree-sitter chunking.

Turns a repository into chunks that each carry code, language, file path,
symbol name, and start/end lines. Language behaviour comes entirely from
config.LANGUAGE_REGISTRY.
"""

from .chunker import MissingGrammarError, chunk_file, chunk_source
from .clone import CloneError, clone_github_repo, clone_path_for, parse_github_ref
from .docs import chunk_doc, is_doc_file, qualifies_as_doc
from .models import MODULE_SYMBOL, Chunk
from .pipeline import ingest_repo
from .walker import collect_source_files

__all__ = [
    "MODULE_SYMBOL",
    "Chunk",
    "CloneError",
    "MissingGrammarError",
    "chunk_doc",
    "chunk_file",
    "chunk_source",
    "clone_github_repo",
    "is_doc_file",
    "qualifies_as_doc",
    "clone_path_for",
    "collect_source_files",
    "ingest_repo",
    "parse_github_ref",
]
