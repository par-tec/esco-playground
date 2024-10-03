"""
ESCO Matcher and NER Model Generator

    This script generates a matcher for ESCO skills and a Named Entity Recognition (NER) model.
    It retrieves skills and occupations from a SPARQL endpoint, creates text embeddings,
    and builds patterns for matching using spaCy. The generated models and patterns are saved
    for later use.

    Usage:
        Run the script with command-line options to generate the matcher, embeddings, and NER model.
"""

import json
import logging
from pathlib import Path

import click
import pandas as pd
import spacy

from esco import to_curie
from esco.sparql import SparqlClient

log = logging.getLogger(__name__)


def make_pattern(id_: str, kn: dict):
    """Given an ESCO skill entry in the dataframe, create a pattern for the matcher.

    The entry has the following fields:
    - label: the preferred label
    - altLabel: a list of alternative labels
    - the skillType: e.g. knowledge, skill, ability

    The logic uses some euristic to decide whether to use
    the preferred label or the alternative labels.
    """
    label = kn["label"]
    pattern = [{"LOWER": label.lower()}] if len(label) > 3 else [{"TEXT": label}]
    patterns = [pattern]
    altLabel = [kn["altLabel"]] if isinstance(kn["altLabel"], str) else kn["altLabel"]
    for alt in altLabel:
        if len(alt) <= 3:
            candidate = [{"TEXT": alt}]
        elif 1 < len(alt.split()) < 4:
            candidate = [{"LOWER": x} for x in alt.lower().split()]
        else:
            candidate = [{"LOWER": alt.lower()}]
        if candidate not in patterns:
            patterns.append(candidate)

    return to_curie(id_), patterns


def esco_matcher(skills):
    """
    Create the patterns for the matcher
    """
    return dict(
        make_pattern(id_, kni) for id_, kni in skills.to_dict(orient="index").items()
    )


@click.command()
@click.option("--esco", default=True, help="Generate the esco matcher")
@click.option("--embeddings", default=True, help="Generate the text embeddings")
@click.option("--ner", default=True, help="Generate the NER model")
@click.option("--sparql", default="http://virtuoso:8890/sparql", help="Sparql URL")
def main(esco, embeddings, ner, sparql):  # pylint: disable=too-many-locals
    """Generate the esco matching model."""
    outdir = Path("generated")
    model_dir = outdir / "en_core_web_trf_esco_ner"
    meta_json = model_dir / "meta.json"
    sparql = SparqlClient(url=sparql)

    log.info("Update the esco.json.gz file")
    esco = sparql.load_esco()
    esco.to_json("esco/esco.json.gz", orient="records", compression="gzip")

    occupations_file = "esco/esco_o.json.gz"
    log.info(f"Update the occupations file: {occupations_file}.")
    occupations = sparql.load_occupations()
    occupations.reset_index().to_json(
        occupations_file, orient="records", compression="gzip"
    )
    df_occupations = pd.read_json(occupations_file, compression="gzip")
    if "uri" not in df_occupations.columns:
        raise ValueError("Missing uri")

    if embeddings:
        skills_file = "esco/esco_s.json.gz"
        log.info("Generate the text embeddings")
        from langchain_community.embeddings import SentenceTransformerEmbeddings

        f = SentenceTransformerEmbeddings(model_name="all-MiniLM-L12-v2")
        skills = sparql.load_skills()
        log.info(f"Generating the embeddings for {len(skills)} skills")
        skills["vector"] = f.embed_documents(skills.text.values)
        skills.reset_index().to_json(skills_file, orient="records", compression="gzip")

        log.info("Validate text embeddings")
        df_esco_skills = pd.read_json(skills_file, compression="gzip")
        if {"uri", "vector"} - set(df_esco_skills.columns):
            raise ValueError("Missing uri or vector")

    if not ner:
        log.warning("Skipping the model generation")
        return

    log.info("Loading the skills from the SPARQL endpoint")
    skills = sparql.load_skills()
    log.info(f"Loaded {len(skills)} skills")

    log.info("Generating the esco matcher")
    m = esco_matcher(skills)

    log.info("Validate the matcher")
    nlp_test = spacy.blank("en")
    m1 = spacy.matcher.Matcher(nlp_test.vocab, validate=True)
    for pid, patterns in m.items():
        m1.add(pid, patterns)

    log.info("Generating the patterns")
    esco_p = [
        {"label": "ESCO", "pattern": pattern, "id": k}
        for k, p in m.items()
        for pattern in p
    ]
    outdir.mkdir(exist_ok=True)
    (outdir / "esco_patterns.json").write_text(json.dumps(esco_p, indent=2))
    log.info("Loading the spacy model")
    nlp_e = spacy.load("en_core_web_trf")
    ruler = nlp_e.add_pipe("entity_ruler", after="ner")
    ruler.add_patterns(esco_p)
    log.info("Saving the model")
    nlp_e.to_disk(model_dir.as_posix())

    log.info("Update meta.json")
    meta = json.loads(meta_json.read_text())
    metadata = json.loads(Path("model/meta.json").read_text())
    meta |= metadata
    meta_json.write_text(json.dumps(meta, indent=2))
    log.info("Done")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
