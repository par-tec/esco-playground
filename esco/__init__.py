import io
import logging
from pathlib import Path

import pandas as pd

# Load a larger pipeline with vectors
from SPARQLWrapper import CSV, SPARQLWrapper

log = logging.getLogger(__name__)


def load_esco_js():
    """Load the skills from the JSON file."""
    df = pd.read_json(Path(__file__).parent / "esco.json.gz", orient="record")
    df.index = df.s
    return df


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
        skos:broader* ?category  ;
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
    skills = df.groupby(df.s).agg(
        {
            "altLabel": lambda x: list(x),
            "label": lambda x: x.iloc[0],
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
