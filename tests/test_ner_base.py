"""
Module for Testing Named Entity Recognition (NER) with ESCO Skills Data.

This module contains tests to validate the functionality of NER using the ESCO dataset
within a local database (LocalDB). It ensures that entities can be recognized in text,
particularly in CVs, by leveraging the Ner class and the VectorDB for skill indexing.
"""

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
    """
    Fixture to create a LocalDB instance with a vector index for testing.

    This fixture initializes a LocalDB, creates a vector index with a specified path,
    and yields the database instance for use in tests. The database is automatically
    closed after the tests are completed.
    """
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
    """
    Fixture to initialize the NER model with the LocalDB instance.

    This fixture sets up the NER model using the provided LocalDB and the specified model
    path from the request parameters. It yields the NER model for use in tests.
    """
    TESTDIR / ".." / "generated" / "en_core_web_trf_esco_ner"
    nlp_e = Ner(
        db=esco_db,
        model_name_or_path=request.param,
        tokenizer=nltk.sent_tokenize,
    )

    yield nlp_e


def test_ner_class_has_db(esco_ner):
    """
    Tests that the NER model is properly initialized with a LocalDB instance.

    Asserts that the NER model is not None and verifies that its database is an instance
    of LocalDB, which in turn should have a vector index of type VectorDB.
    """
    assert esco_ner
    assert isinstance(esco_ner.db, LocalDB)
    assert isinstance(esco_ner.db.vector_idx, VectorDB)


def test_ner_recognizes_entities_in_string(esco_ner):
    """
    Tests the NER model's ability to recognize entities in a given string.

    Provides a sample string describing a software developer's experience and
    verifies that the NER model extracts at least two skills from it.
    """
    cv = esco_ner(
        """I am a software developer with 5 years of experience in Python and Java. """
    )
    skills = cv.skills()
    assert len(skills) >= 2


def test_ner_can_recognize_some_entities_in_cv(esco_ner):
    """
    Tests the NER model's ability to recognize entities in a CV.

    Reads a CV from a specified file, processes it with the NER model, and asserts that
    entities are recognized. Verifies that at least 40 entities are identified in the document.
    """
    text = TESTDIR / "data" / "rpolli.txt"
    doc = esco_ner.model(text.read_text())
    assert doc.ents
    assert len(doc.ents) >= 40


def test_esco_cv(esco_ner):
    """
    Tests the extraction of skills from a CV using the ESCO NER model.

    Parses a CV from a specified file and utilizes the NER model to extract skills.
    Verifies that skills are extracted successfully and saves the sentences containing
    skills and the extracted skills into YAML files for further inspection.
    Also logs the time taken for the extraction of skills.
    """
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
