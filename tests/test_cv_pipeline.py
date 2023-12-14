from pathlib import Path

import pytest
import yaml

from api import tasks

DATADIR = Path(__file__).parent / "data"
recognizer_yaml = DATADIR / "recognizer.yaml"
cvs = yaml.safe_load(recognizer_yaml.read_text())["cvs"]


@pytest.fixture(scope="module")
def recognizer():
    return tasks.Recognizer()


def test_recognize_entities(recognizer):
    text = (DATADIR / "rpolli.txt").read_text()
    result = recognizer.recognize_entities(text)
    assert isinstance(result, dict), "The result should be a dict (JSON)."
    assert "entities" in result
    assert "count" in result
    assert result["count"] > 10
    raise NotImplementedError


@pytest.mark.parametrize("cv", cvs)
def test_infer_skills_from_products(recognizer, cv):
    entities = cv["entities"]
    skills = recognizer.infer_skills(entities)
    assert len(skills) > 10
    assert next(iter(skills)).startswith("http://")

    raise NotImplementedError


@pytest.mark.parametrize("cv", cvs)
def test_infer_skills_from_skills(recognizer, cv):
    skills = cv["esco_skills"]
    recognizer.infer_skills_from_skills(skills)

    raise NotImplementedError
