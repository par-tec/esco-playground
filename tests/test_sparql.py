import io

import pandas as pd
import pytest

import esco.sparql


@pytest.fixture
def sparql():
    yield esco.sparql.SparqlClient(
        url="http://virtuoso:8890/sparql",
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


def test_if_skills_exists_in_json(sparql):
    assert sparql
    ret = sparql.load_skills()
    categories = categories or [  # Defaults to ICT professionals and technicians.
        "http://data.europa.eu/esco/isco/C25",
        "http://data.europa.eu/esco/isco/C35",
    ]

    categories = "\n".join([f"<{uri}>" for uri in categories])

    res = sparql.query(
        """

    SELECT DISTINCT ?s ?label WHERE {

    VALUES ?category { """
        + categories
        + """ }

    ?uri a esco:Occupation ;
        (esco:relatedEssentialSkill | esco:relatedOptionalSkill) ?s ;
        skos:broaderTransitive* ?category  ;
        iso-thes:status "released"
    .

    # Get current skill labels associated
    #  with the occupation.
    ?s iso-thes:status "released" .



    FILTER (lang(?label) = "en")
                    }"""
    )
    df = pd.read_csv(io.StringIO(res.decode()))

    for skill in df["s"]:
        assert skill in ret.uri

    raise NotImplementedError
