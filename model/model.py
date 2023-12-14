import json
import logging
from pathlib import Path

import spacy

from esco import load_skills

log = logging.getLogger(__name__)


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
    Path("generated/esco_patterns.json").write_text(json.dumps(esco_p, indent=2))
    log.info("Loading the spacy model")
    nlp_e = spacy.load("en_core_web_trf")
    ruler = nlp_e.add_pipe("entity_ruler", after="ner")
    ruler.add_patterns(esco_p)
    log.info("Saving the model")
    nlp_e.to_disk("generated/en_core_web_trf_esco_ner")
    log.info("Done")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
