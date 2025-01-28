"""
Unit tests for the ESCO LocalDB functionality.

This module contains tests for loading occupations and skills, retrieving labels,
and searching for skills in the ESCO LocalDB. It uses pytest for testing framework
and parametrize feature for varied input cases.
"""

from pathlib import Path

import pytest

import esco

TESTDIR = Path(__file__).parent


@pytest.fixture
def db(tmpdir):
    """
    Fixture to create and yield an instance of the ESCO LocalDB.

    This fixture provides a fresh LocalDB instance for use in tests.
    """
    yield esco.LocalDB()


def test_load_occupations(db):
    """
    Test loading occupations from the ESCO LocalDB.

    Asserts that the number of loaded occupations is greater than 70.
    """
    occupations = db.load_occupations()
    assert len(occupations) > 70


def test_load_skills(db):
    """
    Test loading skills from the ESCO LocalDB.

    Asserts that the number of loaded skills is greater than 800.
    """
    skills = db.load_skills()
    assert len(skills) > 800


def test_get_label(db):
    """
    Test retrieving labels from the ESCO LocalDB.

    Asserts that the correct label is returned for given ESCO IDs.
    """
    assert db
    assert db.get_label("esco:b0096dc5-2e2d-4bc1-8172-05bf486c3968")
    assert db.get_label(
        "http://data.europa.eu/esco/skill/b0096dc5-2e2d-4bc1-8172-05bf486c3968"
    )


def test_get_missing_return_none(db):
    """
    Test retrieval of a nonexistent entry.

    Asserts that requesting a nonexistent ESCO ID returns None.
    """
    assert db.get("esco:nonexistent") is None


def test_get_skill(db):
    """
    Test retrieving a skill from the ESCO LocalDB.

    Asserts that the skill has a description and its type is 'skill'.
    """
    skill = db.get("esco:b0096dc5-2e2d-4bc1-8172-05bf486c3968")
    assert skill["description"]
    assert skill["skillType"] == "skill"


@pytest.mark.parametrize(
    "products,expected_results",
    [({"ansible", "JBoss", "Bash"}, 3), ({"agile", "scrum", "kanban"}, 1)],
)
def test_search_skill_label(db, products, expected_results):
    """
    Test searching for skills by product labels.

    Asserts that the number of skills returned meets or exceeds the expected results.
    """
    skills = db.search_products(products)
    assert len(skills) >= expected_results
