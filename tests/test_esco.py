from pathlib import Path

import pytest
import spacy

import esco

TESTDIR = Path(__file__).parent


@pytest.mark.parametrize("table", ["skills", "occupations"])
def test_datafile(table):
    data = esco.load_esco_js(table=table)
    assert len(data) > 1000


def test_load_occupations():
    occupations = esco.load_occupations(source="json")
    assert len(occupations) > 70
    raise NotImplementedError


def test_load_skills():
    skills = esco.load_skills(source="json")
    assert len(skills) > 1000
    raise NotImplementedError


def test_esco_ner():
    model_file = TESTDIR / ".." / "generated" / "en_core_web_trf_esco_ner"
    nlp_e = spacy.load(model_file.as_posix())
    text = TESTDIR / "data" / "rpolli.txt"
    nlp_e(text.read_text())


def test_find_esco(products: set):
    skills = esco.load_skills()
    skills[skills.apply(lambda x: bool(x.allLabel & products), axis=1)]
