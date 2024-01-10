import pytest

import esco.sparql


@pytest.fixture
def sparql():
    yield esco.sparql.SparqlClient(
        url="http://virtuoso:8890/sparql",
    )


def test_can_load_skills_from_sparql(sparql):
    assert sparql
    ret = sparql.load_esco()
    assert len(ret) > 5000


def test_can_load_occupations_from_sparql(sparql):
    assert sparql
    ret = sparql.load_occupations()
    assert len(ret) > 70
