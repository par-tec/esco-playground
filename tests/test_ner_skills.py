"""
Module for Testing ESCO Named Entity Recognition (NER) for Skills Extraction.

This module contains tests that validate the functionality of the NER model
in recognizing skills from text, specifically focusing on parsing CVs.
"""

import logging
from pathlib import Path

import nltk
import pytest
import yaml

from esco import LocalDB
from esco.cv import EscoCV
from esco.ner import Ner

TESTDIR = Path(__file__).parent
DATADIR = TESTDIR / "data"
TESTFILE = TESTDIR / "data" / Path(__file__).with_suffix(".yaml").name.replace("_", "-")
log = logging.getLogger(__name__)

# Ensure punkt is installed.
nltk.download("punkt")


@pytest.fixture(scope="module")
def esco_db(tmpdir):
    """
    Fixture to create a LocalDB instance with a vector index for testing.

    This fixture initializes a LocalDB instance, creates a vector index at a specified
    temporary path, and yields the database instance for use in tests. The database is
    automatically closed after the tests are completed.
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

    This fixture loads the NER model using the provided LocalDB and the specified model
    path from the request parameters. It logs the loading process and yields the NER model
    for use in tests.
    """
    log.info("Loading model %s", request.param)
    nlp_e = Ner(
        db=esco_db,
        model_name_or_path=request.param,
        tokenizer=nltk.sent_tokenize,
    )
    log.info("Model loaded")
    yield nlp_e


@pytest.mark.parametrize("testcase", yaml.safe_load(TESTFILE.read_text())["tests"])
def test_recognize_skills_in_text(testcase, esco_ner):
    # Given a text and a set of expected skills...
    text, expected_skills = testcase["text"], testcase["skills"]
    cv = EscoCV(
        ner=esco_ner,
        text=text,
    )

    # When I extract the skills from the text...
    extracted_skills = cv.skills()
    log.info("Skills: %s", len(extracted_skills))

    # ... then I should get at least the skills I expect.
    assert set(extracted_skills) >= set(expected_skills)

    # ... and the ner-skills should be a subset of the extracted skills too.
    extracted_ner_skills = {
        k: {**v, "uri": k}
        for k, v in extracted_skills.items()
        if v.get("source") == "ner"
    }
    expected_ner_skills = {
        k: v for k, v in expected_skills.items() if v.get("source") == "ner"
    }
    assert set(extracted_ner_skills) == set(expected_ner_skills)
