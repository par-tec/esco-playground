import logging
from pathlib import Path
import time
import yaml

import nltk
import pytest

from esco import LocalDB
from esco.ner import Ner
from esco.vector import VectorDB
from esco.cv import EscoCV

TESTDIR = Path(__file__).parent
DATADIR = TESTDIR / "data"
log = logging.getLogger(__name__)

# Ensure punkt is installed.
nltk.download("punkt")


@pytest.fixture(scope="module")
def esco_db(tmpdir):
    db = LocalDB()
    db.create_vector_idx(
        {
            "path": tmpdir / "esco-skills",
            "collection_name": "esco-skills",
        }
    )
    yield db
    db.close()


@pytest.fixture(
    scope="module",
    params=[
        #        "en_core_web_trf_esco_ner",
        Path(TESTDIR / ".." / "generated" / "en_core_web_trf_esco_ner").as_posix(),
    ],
)
def esco_ner(esco_db, request):
    TESTDIR / ".." / "generated" / "en_core_web_trf_esco_ner"
    nlp_e = Ner(
        db=esco_db,
        model_name_or_path=request.param,
        tokenizer=nltk.sent_tokenize,
    )

    yield nlp_e


def test_ner_class_has_db(esco_ner):
    assert esco_ner
    assert isinstance(esco_ner.db, LocalDB)
    assert isinstance(esco_ner.db.vector_idx, VectorDB)


def test_ner_recognizes_entities_in_string(esco_ner):
    cv = esco_ner(
        """I am a software developer with 5 years of experience in Python and Java. """
    )
    skills = cv.skills()
    assert len(skills) >= 2


def test_ner_can_recognize_some_entities_in_cv(esco_ner):
    text = TESTDIR / "data" / "rpolli.txt"
    doc = esco_ner.model(text.read_text())
    assert doc.ents
    assert len(doc.ents) >= 40


def test_esco_cv(esco_ner):
    # When I parse a CV...
    cv_path = TESTDIR / "data" / "rpolli.txt"

    cv = EscoCV(
        ner=esco_ner,
        text=cv_path.read_text(),
    )

    # ... then I should be able to extract skills from it.
    ner_skills = cv.ner_skills()
    assert ner_skills

    skill_sentences = cv.skills_by_sentence()
    Path(f"{cv_path.stem}-skill-sentences.yaml").write_text(yaml.dump(skill_sentences))

    t0 = time.time()
    v_skills = cv.skills()
    log.warning("skills: %s", time.time() - t0)
    Path(f"{cv_path.stem}-skills.yaml").write_text(yaml.dump(v_skills))
