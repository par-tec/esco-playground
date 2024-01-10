import time
from itertools import chain
from pathlib import Path

import pytest
import yaml
from langchain.schema import Document

import esco
from esco.vector import VectorDB

TESTDIR = Path(__file__).parent
DATADIR = TESTDIR / "data"

skills = esco.load_table("skills")
models = {
    "all-MiniLM-L12-v2": {
        "score_threshold": 0.3,
    },
    "paraphrase-albert-small-v2": {"score_threshold": 0.25},
    "paraphrase-MiniLM-L3-v2": {
        "score_threshold": 0.25,
    },
}
documents = [
    Document(page_content=t, metadata={"label": l, "uri": i})
    for t, l, i in zip(skills.text.values, skills.label.values, skills.index.values)
]


@pytest.fixture
def vector_db():
    d = VectorDB(force_recreate=True, skills=skills)
    yield d
    d.close()


def _tokenizers():
    import nltk
    import spacy

    yield "nltk", nltk.sent_tokenize
    yield "spacy", lambda doc: [str(x) for x in spacy.load("en_core_web_sm")(doc).sents]


def test_create_db_path(tmpdir):
    d = VectorDB(
        skills=skills,
        force_recreate=True,
        config={
            "path": tmpdir / f"deleteme-qdrant-esco-{VectorDB.MODEL_NAME}",
            "collection_name": "esco-skills",
        },
    )
    ret = d.search("python")
    assert ret
    d.close()


def test_create_db_url(tmpdir):
    d = VectorDB(
        skills=skills,
        force_recreate=True,
        config={
            "url": "http://qdrant:6333",
            "collection_name": f"deleteme-esco-skills-{VectorDB.MODEL_NAME}",
        },
    )
    ret = d.search("python")
    assert ret
    d.close()


@pytest.mark.skip("You need to provision your data to run this test.")
@pytest.mark.parametrize("cv_path", DATADIR.glob("*-en.txt"))
@pytest.mark.parametrize("tokenizer", _tokenizers())
def test_models(cv_path, tokenizer, vector_db):
    tokenizer_name, tokenizer = tokenizer
    sentences = tokenizer(cv_path.read_text())
    neural_cv = []
    ts_start = time.time()
    for sentence in sentences:
        txt = str(sentence).strip()
        if not txt:
            continue
        if len(txt.split()) < 5:
            continue

        neural_cv.append(
            {
                "text": txt,
                "skills": [
                    {
                        "id": x[0].metadata["id"],
                        "label": x[0].metadata["label"],
                        "score": x[1],
                    }
                    for x in vector_db.search(txt)
                ],
            }
        )

    results = list(chain(*(x["skills"] for x in neural_cv)))
    Path(f"{cv_path.stem}-esco-neural-{vector_db.model_name}.yaml").write_text(
        yaml.dump(neural_cv)
    )

    assert results
    x = {}
    for d in results:
        id_ = d["id"]
        if id_ not in x:
            x[id_] = d | {"count": 1}
        else:
            x[id_]["score"] = max(d["score"], x[id_]["score"])
            x[id_]["count"] += 1
    stats = {
        "st_model": vector_db.model_name,
        "cv_path": cv_path.name,
        "tokenizer": tokenizer_name,
        "count": len({x["label"] for x in results}),
        "max_score": max(x.values(), key=lambda x: x["score"])["score"],
        "max_label": max(x.values(), key=lambda x: x["count"])["label"],
        "min_score": min(x.values(), key=lambda x: x["score"])["score"],
        "min_label": min(x.values(), key=lambda x: x["count"])["label"],
        "elapsed": time.time() - ts_start,
    }
    print(stats)
