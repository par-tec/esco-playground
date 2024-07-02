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
    assert len(ret) > 8000
    assert "narrowers" in ret.columns


def test_can_load_skills_from_sparql(sparql):
    assert sparql
    ret = sparql.load_skills()
    assert len(ret) > 1300
    assert "narrowers" in ret.columns


def test_can_load_occupations_from_sparql(sparql):
    assert sparql
    ret = sparql.load_occupations()
    assert len(ret) > 70


@pytest.fixture
def query_essential_skills():
    categories = [  # Defaults to ICT professionals and technicians.
        "http://data.europa.eu/esco/isco/C25",
        "http://data.europa.eu/esco/isco/C35",
    ]

    categories = "\n".join([f"<{uri}>" for uri in categories])

    q = (
        """

    SELECT DISTINCT ?s ?label WHERE {

    VALUES ?category { """
        + categories
        + """ }

    ?uri a esco:Occupation ;
        esco:relatedEssentialSkill ?s ;
        skos:broaderTransitive* ?category  ;
        iso-thes:status "released"
    .

    # Get current skill labels associated
    #  with the occupation.
    ?s iso-thes:status "released" ;
    skos:prefLabel ?label
    .

    FILTER (lang(?label) = "en")
                    }"""
    )
    return q


def test_if_skills_exists_in_sparql(sparql, query_essential_skills):
    res = sparql.query(query_essential_skills)
    df = pd.read_csv(io.StringIO(res.decode()))

    all_ict_occupations_skills = set(df["s"].values)

    isco_essential_skills = set(sparql.load_isco().s.values)

    missing_essential_skills = all_ict_occupations_skills - isco_essential_skills

    # verifica se il df non è
    ret = sparql.load_skills()
    missing_skills = all_ict_occupations_skills - set(ret.index)
    assert not missing_essential_skills, [
        df[df.s == x].label for x in missing_essential_skills
    ]
    assert not missing_skills, [df[df.s == x].label for x in missing_skills]


def test_if_skills_exists_in_json(sparql, query_essential_skills):
    db = esco.LocalDB()
    df = db.load_skills()
    all_skills_from_json = set(df.index)

    res = sparql.query(query_essential_skills)
    df = pd.read_csv(io.StringIO(res.decode()))

    all_ict_occupations_skills = set(df["s"].values)

    missing_skills = all_ict_occupations_skills - all_skills_from_json
    assert not missing_skills, [df[df.s == x].label for x in missing_skills]
