"""Health check: every module under src/ imports cleanly, and config is coherent.

This grows automatically as modules are added — no list to keep in sync.
"""

from __future__ import annotations

import importlib
import pkgutil

import pytest

import src
from src import config


def _module_names() -> list[str]:
    """Every importable module and subpackage under src/, recursively."""
    return ["src"] + [
        m.name for m in pkgutil.walk_packages(src.__path__, prefix="src.")
    ]


@pytest.mark.parametrize("name", _module_names())
def test_module_imports(name: str) -> None:
    importlib.import_module(name)


def test_registry_is_seeded_with_python() -> None:
    spec = config.spec_for_path("example.py")
    assert spec is not None
    assert spec.grammar_module == "tree_sitter_python"
    assert spec.function_nodes and spec.class_nodes
    assert spec.test_command[0] == "python"


def test_unsupported_extension_returns_none() -> None:
    assert config.spec_for_path("README.md") is None


def test_registry_entries_are_well_formed() -> None:
    for ext, spec in config.LANGUAGE_REGISTRY.items():
        assert ext.startswith("."), ext
        assert ext == ext.lower(), ext
        assert spec.grammar_module, ext
        assert spec.grammar_entrypoint, ext
        assert spec.name_fields, ext
        assert spec.test_command, ext
        # class_nodes is legitimately empty for Go and C, which declare no
        # member-owning types, but a language must recognise *something*.
        assert spec.symbol_nodes(), ext


def test_chunk_bounds_are_sane() -> None:
    # Overlap must be a strict fraction of the window or splitting never advances.
    assert config.CHUNK_OVERLAP_LINES < config.MAX_CHUNK_LINES


def test_final_top_k_does_not_exceed_candidate_pool() -> None:
    assert config.FINAL_TOP_K <= config.VECTOR_TOP_K + config.BM25_TOP_K
