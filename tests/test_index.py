"""Embedding cache, Chroma store, and vector retrieval.

These tests load the real embedding model once and index a real (tiny) repo.
Mocking the encoder would leave the parts most likely to break — vector shapes,
metadata round-tripping, cosine sign conventions — untested.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src import config
from src.index import (
    cache_key,
    chunk_id,
    collection_name,
    document_text,
    embed_chunks,
    embed_query,
    embed_texts,
    get_collection,
    index_repo,
    load_cache,
    load_model,
    prune_to,
    save_cache,
    search,
    upsert_chunks,
)
from src.ingest import Chunk, ingest_repo

SAMPLE = """\
\"\"\"Money helpers.\"\"\"


def parse_gitignore(repo_path):
    \"\"\"Read .gitignore and return the patterns it declares.\"\"\"
    return []


def compound_interest(principal, rate, years):
    \"\"\"Grow a principal at a fixed annual rate.\"\"\"
    return principal * (1 + rate) ** years


class HttpRetryClient:
    \"\"\"Issues HTTP requests and retries on failure with backoff.\"\"\"

    def fetch(self, url):
        return url
"""


@pytest.fixture(scope="module")
def repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A one-file repository to index."""
    root = tmp_path_factory.mktemp("sample_repo")
    (root / "money.py").write_text(SAMPLE, encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def indexed(repo: Path, tmp_path_factory: pytest.TempPathFactory):
    """Index the sample repo into a throwaway Chroma directory."""
    store = tmp_path_factory.mktemp("chroma")
    cache = tmp_path_factory.mktemp("emb_cache")
    stats = index_repo(repo, persist_dir=str(store), cache_dir=cache)
    return stats, get_collection(repo, str(store))


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------


def test_cache_key_depends_on_text_and_model() -> None:
    assert cache_key("a", "m") == cache_key("a", "m")
    assert cache_key("a", "m") != cache_key("b", "m")
    assert cache_key("a", "m1") != cache_key("a", "m2")


def test_cache_key_cannot_alias_across_the_separator() -> None:
    """"ab"+"c" and "a"+"bc" must not collide."""
    assert cache_key("c", "ab") != cache_key("bc", "a")


def test_cache_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "c.npz"
    original = {"k1": np.ones(4, dtype=np.float32), "k2": np.zeros(4, dtype=np.float32)}
    save_cache(path, original)

    restored = load_cache(path)
    assert set(restored) == set(original)
    assert np.array_equal(restored["k1"], original["k1"])


def test_empty_cache_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "c.npz"
    save_cache(path, {})
    assert load_cache(path) == {}


def test_missing_and_corrupt_caches_are_treated_as_empty(tmp_path: Path) -> None:
    assert load_cache(tmp_path / "nope.npz") == {}

    corrupt = tmp_path / "bad.npz"
    corrupt.write_bytes(b"not an npz file")
    assert load_cache(corrupt) == {}


# --------------------------------------------------------------------------
# Embedding
# --------------------------------------------------------------------------


def test_configured_dim_matches_the_real_model() -> None:
    """EMBEDDING_DIM is used to shape empty results; a stale value fails late.

    Swapping EMBEDDING_MODEL without updating EMBEDDING_DIM would otherwise
    only surface on an edge case, long after the index was built.
    """
    assert load_model().get_embedding_dimension() == config.EMBEDDING_DIM


def test_embeddings_have_the_configured_shape() -> None:
    vectors = embed_texts(["def a(): pass", "def b(): pass"])
    assert vectors.shape == (2, config.EMBEDDING_DIM)
    assert vectors.dtype == np.float32


def test_embeddings_are_normalised() -> None:
    """Cosine similarity is only a dot product if the vectors are unit length."""
    norms = np.linalg.norm(embed_texts(["one", "two", "three"]), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_embedding_is_deterministic() -> None:
    assert np.allclose(embed_query("how do we retry"), embed_query("how do we retry"))


def test_document_text_includes_path_and_symbol() -> None:
    chunk = Chunk(
        code="return 1",
        language="Python",
        file_path="src/a.py",
        symbol="Widget.render",
        kind="function",
        start_line=1,
        end_line=1,
    )
    text = document_text(chunk)
    assert "src/a.py" in text
    assert "Widget.render" in text
    assert "return 1" in text


def test_unchanged_chunks_are_not_re_embedded(repo: Path, tmp_path: Path) -> None:
    """The point of the cache: a second run sends nothing to the model."""
    chunks = ingest_repo(repo)
    cache_file = tmp_path / "cache.npz"

    first_vectors, first = embed_chunks(chunks, cache_file)
    assert first.computed > 0
    assert first.cached == 0

    second_vectors, second = embed_chunks(chunks, cache_file)
    assert second.computed == 0
    assert second.cached == len(chunks)
    assert second.hit_rate == 1.0
    assert np.array_equal(first_vectors, second_vectors)


def test_changed_chunks_are_re_embedded(repo: Path, tmp_path: Path) -> None:
    chunks = ingest_repo(repo)
    cache_file = tmp_path / "cache.npz"
    embed_chunks(chunks, cache_file)

    edited = replace(chunks[0], code=chunks[0].code + "\n# changed")
    _, stats = embed_chunks([edited, *chunks[1:]], cache_file)
    assert stats.computed == 1


def test_duplicate_chunks_are_embedded_once(repo: Path, tmp_path: Path) -> None:
    chunks = ingest_repo(repo)
    vectors, stats = embed_chunks(list(chunks) + list(chunks), tmp_path / "c.npz")
    assert stats.total == 2 * len(chunks)
    assert stats.computed == len(chunks)
    assert np.array_equal(vectors[: len(chunks)], vectors[len(chunks) :])


def test_embedding_no_chunks_is_not_an_error(tmp_path: Path) -> None:
    vectors, stats = embed_chunks([], tmp_path / "c.npz")
    assert vectors.shape == (0, config.EMBEDDING_DIM)
    assert stats.total == 0


# --------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------


def test_collection_name_is_chroma_legal(tmp_path: Path) -> None:
    name = collection_name(tmp_path)
    assert 3 <= len(name) <= 63
    assert re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]*[a-zA-Z0-9]", name)


def test_collection_name_is_stable_and_path_specific(tmp_path: Path) -> None:
    a = tmp_path / "repo"
    b = tmp_path / "nested" / "repo"
    (b).mkdir(parents=True)
    a.mkdir()
    assert collection_name(a) == collection_name(a)
    assert collection_name(a) != collection_name(b), "same basename must not collide"


def test_chunk_ids_are_unique_for_a_real_repo(repo: Path) -> None:
    chunks = ingest_repo(repo)
    ids = [chunk_id(c) for c in chunks]
    assert len(ids) == len(set(ids))


def test_indexing_stores_every_chunk(repo: Path, indexed) -> None:
    stats, collection = indexed
    assert stats.chunks == len(ingest_repo(repo))
    assert collection.count() == stats.chunks


def test_reindexing_is_idempotent(repo: Path, tmp_path: Path) -> None:
    store, cache = str(tmp_path / "chroma"), tmp_path / "cache"
    first = index_repo(repo, persist_dir=store, cache_dir=cache)
    second = index_repo(repo, persist_dir=store, cache_dir=cache)

    assert second.chunks == first.chunks
    assert second.embedded == 0, "second run should hit the cache entirely"
    assert second.pruned == 0
    assert get_collection(repo, store).count() == first.chunks


def test_deleted_code_is_pruned(indexed) -> None:
    """A chunk that ingest no longer produces must leave the collection."""
    _, collection = indexed
    before = collection.count()

    ghost = Chunk(
        code="def gone(): pass",
        language="Python",
        file_path="deleted.py",
        symbol="gone",
        kind="function",
        start_line=1,
        end_line=1,
    )
    upsert_chunks(collection, [ghost], embed_texts([document_text(ghost)]))
    assert collection.count() == before + 1

    keep = [i for i in collection.get(include=[])["ids"] if i != chunk_id(ghost)]
    assert prune_to(collection, keep) == 1
    assert collection.count() == before


def test_upsert_rejects_mismatched_vectors(indexed) -> None:
    """Silently zipping to the shorter list would corrupt the index."""
    _, collection = indexed
    with pytest.raises(ValueError):
        upsert_chunks(collection, [], np.zeros((3, config.EMBEDDING_DIM), dtype=np.float32))


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------


def test_a_broken_ingest_does_not_wipe_the_index(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pruning to zero chunks would delete everything.

    A missing grammar wheel or a chunker regression makes ingest return [], and
    silently destroying a good index over it is the worst outcome available.
    """
    store, cache = str(tmp_path / "chroma"), tmp_path / "cache"
    first = index_repo(repo, persist_dir=store, cache_dir=cache)
    assert first.chunks > 0

    monkeypatch.setattr("src.index.pipeline.ingest_repo", lambda _: [])
    with pytest.warns(UserWarning, match="no chunks"):
        broken = index_repo(repo, persist_dir=store, cache_dir=cache)

    assert broken.pruned == 0
    assert get_collection(repo, store).count() == first.chunks


def test_switching_embedding_model_rebuilds_the_collection(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two models can share a dimension without sharing a vector space.

    Chroma would accept the mixture silently, so the collection is stamped with
    the model that built it and dropped when that no longer matches.
    """
    store = str(tmp_path / "chroma")
    index_repo(repo, persist_dir=store, cache_dir=tmp_path / "cache")
    assert get_collection(repo, store).count() > 0

    monkeypatch.setattr(config, "EMBEDDING_MODEL", "some/other-model")
    with pytest.warns(UserWarning, match="unusable"):
        rebuilt = get_collection(repo, store)

    assert rebuilt.count() == 0, "stale vectors must not survive a model swap"
    assert rebuilt.metadata["embedding_model"] == "some/other-model"


def test_search_returns_ranked_results(indexed) -> None:
    _, collection = indexed
    results = search(collection, "how do we grow money over time", k=3)

    assert len(results) == 3
    assert [r.rank for r in results] == [1, 2, 3]
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert all(-1.0 <= s <= 1.0 for s in scores)


def test_search_finds_the_semantically_right_chunk(indexed) -> None:
    """A question sharing no identifiers with the code should still match it."""
    _, collection = indexed
    top = search(collection, "retrying a failed network request", k=1)[0]
    assert top.chunk.symbol.startswith("HttpRetryClient")


def test_search_round_trips_chunk_metadata(indexed) -> None:
    _, collection = indexed
    chunk = search(collection, "compound interest", k=1)[0].chunk

    assert chunk.language == "Python"
    assert chunk.file_path == "money.py"
    assert chunk.kind in {"function", "class", "type", "module"}
    assert 1 <= chunk.start_line <= chunk.end_line
    assert chunk.code


def test_search_respects_k(indexed) -> None:
    _, collection = indexed
    assert len(search(collection, "money", k=1)) == 1
    assert search(collection, "money", k=0) == []
    # k larger than the collection returns everything, not an error.
    assert len(search(collection, "money", k=999)) == collection.count()


def test_search_on_an_empty_collection_returns_nothing(tmp_path: Path) -> None:
    empty = tmp_path / "empty_repo"
    empty.mkdir()
    assert search(get_collection(empty, str(tmp_path / "chroma")), "anything") == []
