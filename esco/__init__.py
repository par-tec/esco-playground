import io
import logging
from pathlib import Path

import pandas as pd

# Load a larger pipeline with vectors
from SPARQLWrapper import CSV, SPARQLWrapper

log = logging.getLogger(__name__)


def load_esco_js(table="skills"):
    """Load the skills from the JSON file."""
    if table == "skills":
        fpath = "esco.json.gz"
        idx = "s"
    elif table == "occupations":
        fpath = "esco_o.json.gz"
        idx = "o"
    else:
        raise ValueError(f"Unknown table {table}")
    df = pd.read_json(Path(__file__).parent / fpath, orient="record")
    return df.set_index(idx)


# Sparql
#
def sparql_query(query, url="http://localhost:18890/sparql"):
    sparql = SPARQLWrapper(url)
    query = (
        """
        prefix jolux: <http://data.legilux.public.lu/resource/ontology/jolux#>
        prefix adms:  <http://www.w3.org/ns/adms#>
        prefix skosXl: <http://www.w3.org/2008/05/skos-xl#>
        prefix owl:   <http://www.w3.org/2002/07/owl#>
        prefix skosxl: <http://www.w3.org/2008/05/skos-xl#>
        prefix org:   <http://www.w3.org/ns/org#>
        prefix xsd:   <http://www.w3.org/2001/XMLSchema#>
        prefix iso-thes: <http://purl.org/iso25964/skos-thes#>
        prefix skos:  <http://www.w3.org/2004/02/skos/core#>
        prefix rdfs:  <http://www.w3.org/2000/01/rdf-schema#>
        prefix vocab-adms: <https://www.w3.org/TR/vocab-adms/#>
        prefix at:    <http://publications.europa.eu/ontology/authority/>
        prefix dct:   <http://purl.org/dc/terms/>
        prefix rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        prefix esco:  <http://data.europa.eu/esco/model#>
        prefix rov:   <http://www.w3.org/ns/regorg#>
        prefix etms:  <http://data.europa.eu/esco/etms/model/#>
        prefix dcat:  <http://www.w3.org/ns/dcat#>
        prefix euvoc: <http://publications.europa.eu/ontology/euvoc#>
        prefix prov:  <http://www.w3.org/ns/prov#>
        prefix foaf:  <http://xmlns.com/foaf/0.1/>
        prefix qdr:   <http://data.europa.eu/esco/qdr#>
        """
        + query
    )
    sparql.setQuery(query)
    sparql.setReturnFormat(CSV)
    results = sparql.query().convert()
    return results


def load_esco(categories=None):
    categories = categories or [
        "http://data.europa.eu/esco/isced-f/06",
        "http://data.europa.eu/esco/skill/243eb885-07c7-4b77-ab9c-827551d83dc4",
    ]

    categories = "\n".join([f"<{uri}>" for uri in categories])

    res = sparql_query(
        """

    SELECT DISTINCT * WHERE {

    VALUES ?category { """
        + categories
        + """ }

    ?s a esco:Skill ;
        skos:prefLabel ?label ;
        skos:broaderTransitive* ?category  ;
        esco:skillType _:skillType ;
        iso-thes:status "released"
    .

    _:skillType skos:prefLabel ?skillType .

    OPTIONAL {
        ?s skos:altLabel ?altLabel .
        ?s dct:description _:description .

        _:description
        esco:nodeLiteral ?description;
        esco:language "en"^^xsd:language
        .
    }

    # ?description contains "cloud" .
    # FILTER regex(CONCAT(?description," ", ?label), "program", "i") .
    FILTER (lang(?label) = "en") .
    FILTER(lang(?altLabel) = "en")
    FILTER(lang(?skillType) = "en")
                    }"""
    )
    df = pd.read_csv(io.StringIO(res.decode()))
    return df


def load_skills(source="sparql"):
    # Concatenate the values label, altLabel and description in the `text` column separated by "; "
    df = load_esco() if source == "sparql" else load_esco_js()
    skills = df.groupby("s").agg(
        {
            "label": lambda x: x.iloc[0],
            "altLabel": lambda x: list(set(x)),
            "description": lambda x: x.iloc[0],
            "skillType": lambda x: x.iloc[0],
        }
    )
    # Add a lowercase text field for semantic search.
    skills["text"] = skills.apply(
        lambda x: "; ".join([x.label] + x.altLabel + [x.description]).lower(), axis=1
    )
    # .. and a set of all the labels for each skill.

    skills["allLabel"] = skills.apply(
        lambda x: {t.lower() for t in x.altLabel} | {x.label.lower()}, axis=1
    )
    return skills


def infer_skills_from_products(skills, product_labels: list):
    """
    Infer skills from a set of products.
    """
    product_labels = {p.lower() for p in product_labels}
    ret = skills[skills.apply(lambda x: bool(x.allLabel & product_labels), axis=1)]
    # return only the s and the label
    return ret[["label"]].to_dict(orient="index")


def infer_skills_from_skill(skill_uri: str):
    """
    Infer skills from a set of skills.
    """
    query = f"""
    SELECT DISTINCT (?parent AS ?s) ?label WHERE {{

    ?parent a esco:Skill ;
        skos:prefLabel ?label

    .

    <{skill_uri}> skos:broaderTransitive+ ?parent .

    FILTER (lang(?label) = 'en')
    }}"""

    res = sparql_query(query, url="http://localhost:18890/sparql")
    df = pd.read_csv(io.StringIO(res.decode()))
    return df.groupby(df.s).agg(lambda x: x.iloc[0]).to_dict(orient="index")


def load_isco(categories=None):
    categories = categories or [  # Defaults to ICT professionals and technicians.
        "http://data.europa.eu/esco/isco/C25",
        "http://data.europa.eu/esco/isco/C35",
    ]

    categories = "\n".join([f"<{uri}>" for uri in categories])

    res = sparql_query(
        """

    SELECT DISTINCT * WHERE {

    VALUES ?category { """
        + categories
        + """ }

    ?o a esco:Occupation ;
        skos:prefLabel ?label ;
        esco:relatedEssentialSkill ?s ;
        skos:broaderTransitive* ?category  ;
        iso-thes:status "released"
    .

    # Get current skill labels associated
    #  with the occupation.
    ?s skos:prefLabel ?skill ;
        esco:skillType ?skillType ;
        iso-thes:status "released"
        .
    FILTER (lang(?skill) = "en")

    # If an occupation lacks a description,
    #   don't skip it.
    OPTIONAL {
        ?o skos:altLabel ?altLabel .
        ?o dct:description _:description .

        _:description
        esco:nodeLiteral ?description;
        esco:language "en"^^xsd:language
        .
    }

    FILTER (lang(?label) = "en")
    FILTER(lang(?altLabel) = "en")
                    }"""
    )
    df = pd.read_csv(io.StringIO(res.decode()))
    return df


def load_occupations(source="sparql"):
    occupations = load_isco() if source == "sparql" else load_esco_js("occupations")
    o = occupations.groupby("o").apply(
        lambda x: pd.Series(
            {
                "label": x.label.iloc[0],
                "altLabel": list(set(x.altLabel)),
                "description": x.description.iloc[0],
                "skill": list(set(x.skill.values)),
                "skill_": list(set(x[x.skillType.str.endswith("skill")].skill.values)),
                "knowledge_": list(
                    set(x[x.skillType.str.endswith("knowledge")].skill.values)
                ),
                "s": list(set(x.s.values)),
            }
        )
    )
    # Add a lowercase text field for semantic search.
    o["text"] = o.apply(
        lambda x: "; ".join([x.label] + x.altLabel + [x.description]).lower(), axis=1
    )
    # .. and a set of all the labels for each skill.

    o["allLabel"] = o.apply(
        lambda x: {t.lower() for t in x.altLabel} | {x.label.lower()}, axis=1
    )
    return o
