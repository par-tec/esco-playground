import io
import logging

import pandas as pd
import spacy

# Load a larger pipeline with vectors
from SPARQLWrapper import CSV, SPARQLWrapper

log = logging.getLogger(__name__)


#
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
    esco:skillType _:skillType
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


def load_skills():
    # Concatenate the values label, altLabel and description in the `text` column separated by "; "
    df = load_esco()
    skills = df.groupby(df.s).agg(
        {
            "altLabel": lambda x: list(x),
            "label": lambda x: x.iloc[0],
            "description": lambda x: x.iloc[0],
            "skillType": lambda x: x.iloc[0],
        }
    )
    skills["text"] = skills.apply(
        lambda x: "; ".join([x.label] + x.altLabel + [x.description]), axis=1
    )
    skills["text"] = skills["text"].str.lower()
    skills["allLabel"] = skills.apply(
        lambda x: {t.lower() for t in x.altLabel} | {x.label.lower()}, axis=1
    )
    return skills


def make_pattern(kn: dict):
    """Given an ESCO skill entry in the dataframe, create a pattern for the matcher.

    The entry has the following fields:
    - label: the preferred label
    - altLabel: a list of alternative labels
    - the skillType: e.g. knowledge, skill, ability

    The logic uses some euristic to decide whether to use the preferred label or the alternative labels.
    """
    label = kn["label"]
    pattern = [{"LOWER": label.lower()}] if len(label) > 3 else [{"TEXT": label}]
    patterns = [pattern]
    altLabel = [kn["altLabel"]] if isinstance(kn["altLabel"], str) else kn["altLabel"]
    for alt in altLabel:
        if len(alt) <= 3:
            candidate = [{"TEXT": alt}]
        elif len(alt.split()) > 1:
            candidate = [{"LOWER": x} for x in alt.lower().split()]
        else:
            candidate = [{"LOWER": alt.lower()}]
        if candidate not in patterns:
            patterns.append(candidate)
    pattern_identifier = (
        f"{kn['skillType'][:2]}_{label.replace(' ', '_')}".upper().translate(
            str.maketrans("", "", "()")
        )
    )
    return pattern_identifier, patterns


def esco_matcher():
    skills = load_skills()
    # Create the patterns for the matcher
    return dict(make_pattern(kni) for kni in skills.to_dict(orient="records"))


def infer_skills_from_products(skills, product_labels: list):
    """
    Infer skills from a set of products.
    """
    product_labels = {p.lower() for p in product_labels}
    ret = skills[skills.apply(lambda x: bool(x.allLabel & product_labels), axis=1)]
    # return only the s and the label
    return ret[["label"]].to_dict(orient="index")


def infer_skills_from_skills(skill_uri: str):
    """
    Infer skills from a set of skills.
    """
    query = f"""
    SELECT DISTINCT * WHERE {{

    ?parent a esco:Skill ;
    skos:prefLabel ?label
    .

    <{skill_uri}> skos:broader* ?parent .

    FILTER (lang(?label) = 'en')
    }}"""

    print(query)
    res = sparql_query(query, url="http://localhost:18890/sparql")
    df = pd.read_csv(io.StringIO(res.decode()))
    return df[["label"]].to_dict(orient="index")


def main():
    """Generate the esco matching model."""
    log.info("Generating the esco matcher")
    m = esco_matcher()
    esco_p = [
        {
            "label": "ESCO",
            "pattern": pattern,
        }
        for k, p in m.items()
        for pattern in p
    ]
    log.info("Loading the spacy model")
    nlp_e = spacy.load("en_core_web_trf")
    ruler = nlp_e.add_pipe("entity_ruler", after="ner")
    ruler.add_patterns(esco_p)
    log.info("Saving the model")
    nlp_e.to_disk("generated/en_core_web_trf_esco_ner")
    log.info("Done")
