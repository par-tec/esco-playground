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
    db = LocalDB(**kwargs)
    db.skills = db.skills[:10]
    return db


@contextmanager
def TmpVectorIdx(**kwargs):
    idx = VectorDB(**kwargs)
    yield idx
    idx.close()


@contextmanager
def TmpLocalDB(**kwargs):
    db = LocalDBShort(**kwargs)
    yield db
    db.vector_idx.qdrant.client.delete_collection(
        collection_name=db.vector_idx.config["collection_name"]
    )
    db.close()


@pytest.mark.parametrize("vector_idx_config", _vector_idx_configs)
def test_can_create_idx_with_path(tmpdir, request, vector_idx_config):
    if vector_idx_config.get("path") == -1:
        vector_idx_config |= {"path": tmpdir / f"deleteme-{uuid4()}"}

    idx = VectorDB(skills=skills_10, force_recreate=True, config=vector_idx_config)
    ret = idx.search("haskell")
    assert ret
    idx.close()


def test_can_load_idx_from_disk(tmpdir, request):
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
