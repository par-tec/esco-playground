import pytest

import esco.sparql


@pytest.fixture
def sparql():
    yield esco.sparql.SparqlClient(
        url="http://virutoso:8890/sparql",
    )


def test_can_load_esco_from_sparql(sparql):
    assert sparql
    ret = sparql.load_esco()
    assert len(ret) > 5000
    assert "narrowers" in ret.columns


def test_can_load_skills_from_sparql(sparql):
    assert sparql
    ret = sparql.load_skills()
    assert len(ret) > 900
    assert "narrowers" in ret.columns


def test_can_load_occupations_from_sparql(sparql):
    assert sparql
    ret = sparql.load_occupations()
    assert len(ret) > 70
