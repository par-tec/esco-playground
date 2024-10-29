"""
Module for Testing Vector and Local Database Operations with ESCO Skills Data.

This module defines tests for creating, loading, and managing vector indices with `VectorDB` and
`LocalDB` using a subset of ESCO skills data. It includes context managers for handling temporary
database instances and pytest functions to validate indexing, search functionality, and persistence.

"""

from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import pytest

import esco
from esco import LocalDB
from esco.vector import VectorDB

TESTDIR = Path(__file__).parent
DATADIR = TESTDIR / "data"

skills_10 = esco.load_table("skills")[:10]
_vector_idx_configs = [
    {"path": -1, "collection_name": "deleteme-esco-skills"},
    {"url": "http://qdrant:6333", "collection_name": "deleteme-esco-skills"},
]


def LocalDBShort(**kwargs):
    """
    Creates a shortened LocalDB instance containing only the first 10 skills.
    """
    db = LocalDB(**kwargs)
    db.skills = db.skills[:10]
    return db


@contextmanager
def TmpVectorIdx(**kwargs):
    """
    Context manager for creating and automatically closing a temporary VectorDB instance.
    Initializes a VectorDB for use within the context and ensures it is closed upon exit.
    """
    idx = VectorDB(**kwargs)
    yield idx
    idx.close()


@contextmanager
def TmpLocalDB(**kwargs):
    """
    Context manager for a temporary LocalDB instance with automatic cleanup.

    Creates a LocalDB instance (limited to the first 10 skills) and ensures that the associated
    vector index collection is deleted and the database is closed when exiting the context.
    """
    db = LocalDBShort(**kwargs)
    yield db
    db.vector_idx.qdrant.client.delete_collection(
        collection_name=db.vector_idx.config["collection_name"]
    )
    db.close()


@pytest.mark.parametrize("vector_idx_config", _vector_idx_configs)
def test_can_create_idx_with_path(tmpdir, request, vector_idx_config):
    """
    Test to ensure a VectorDB index can be created using a specified path.

    If no valid path is provided in the configuration, a temporary path is assigned.
    Verifies that the index is created successfully and returns non-empty search results,
    then closes the index.
    """
    if vector_idx_config.get("path") == -1:
        vector_idx_config |= {"path": tmpdir / f"deleteme-{uuid4()}"}

    idx = VectorDB(skills=skills_10, force_recreate=True, config=vector_idx_config)
    ret = idx.search("haskell")
    assert ret
    idx.close()


def test_can_load_idx_from_disk(tmpdir, request):
    """
    Tests loading a VectorDB index from disk to verify persistence.

    Creates a VectorDB instance with a specified path, performs a search to confirm
    successful indexing, and closes it. Then reopens the index from disk and verifies
    that the same search returns non-empty results, confirming data persistence.
    """
    # When I create a vector database
    idx = VectorDB(
        skills=skills_10,
        force_recreate=True,
        config={
            "path": tmpdir
            / f"deleteme-qdrant-esco-{request.node.name}-{VectorDB.MODEL_NAME}",
            "collection_name": "esco-skills",
        },
    )
    ret = idx.search("haskell")
    idx.close()
    assert ret

    # I can load it again
    idx2 = VectorDB(
        skills=skills_10,
        force_recreate=False,
        config=idx.config,
    )
    ret = idx2.search("haskell")
    assert ret
    idx2.close()


@pytest.mark.parametrize(
    "vector_idx_config",
    _vector_idx_configs,
)
def test_localdb_can_load_existing_idx(tmpdir, request, vector_idx_config):
    """
    Tests loading an existing VectorDB index in a LocalDB instance.

    Creates a VectorDB with the given configuration, ensuring it is set up properly,
    and closes it. Then, it loads the index within a temporary LocalDB context
    and verifies that a search returns non-empty results, confirming successful loading.
    """
    if vector_idx_config.get("path") == -1:
        vector_idx_config |= {"path": tmpdir / f"deleteme-{uuid4()}"}

    # When I have an existing vector database...
    idx = VectorDB(
        skills=skills_10,
        force_recreate=True,
        config=vector_idx_config,
    )
    idx.close()

    # .. I can load it again.
    with TmpLocalDB(vector_idx_config=idx.config) as db:
        ret = db.search_neural("haskell")
        assert ret


@pytest.mark.parametrize(
    "vector_idx_config",
    _vector_idx_configs,
)
def test_localdb_can_create_idx(tmpdir, request, vector_idx_config):
    """
    Tests the creation of a VectorDB index within a LocalDB instance.

    If the provided vector index configuration lacks a valid path, a temporary path is assigned.
    A new vector index is created in the LocalDB, and a search is performed to verify
    successful indexing by checking for non-empty results.
    """
    if vector_idx_config.get("path") == -1:
        vector_idx_config |= {"path": tmpdir / f"deleteme-{uuid4()}"}

    with TmpLocalDB() as db:
        db.create_vector_idx(
            vector_idx_config=vector_idx_config,
        )
        ret = db.search_neural("haskell")
        assert ret


@pytest.mark.parametrize(
    "vector_idx_config",
    _vector_idx_configs,
)
def test_localdb_can_recreate_idx(tmpdir, request, vector_idx_config):
    """
    Tests the recreation of a VectorDB index in a LocalDB instance.

    If no valid path is provided in the vector index configuration, a temporary path is assigned.
    An existing vector index is created, and after modifying the skills in the LocalDB,
    the vector index is recreated. The test verifies that the removed entry is no longer
    present in the index and that a search for it returns no results.
    """
    if vector_idx_config.get("path") == -1:
        vector_idx_config |= {"path": tmpdir / f"deleteme-{uuid4()}"}

    # When I have an existing vector database...
    ldb = LocalDBShort()
    ldb.create_vector_idx(vector_idx_config=vector_idx_config)
    ldb.close()

    # .. I can load it...
    db = LocalDBShort(
        vector_idx_config=ldb.vector_idx.config,
    )
    db.validate()
    assert len(db.vector_idx.scroll(limit=10000)[0]) == 10

    # .. modify the skills ..
    db.skills = db.skills[1:]

    # .. and recreate the vector index.
    db.create_vector_idx()

    # Then the removed entry is not in the index anymore.
    assert len(db.vector_idx.scroll(limit=10000)[0]) == 9
    ret = db.search_neural("haskell")
    assert not ret
