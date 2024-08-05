import logging

import pytest
import spacy

from esco import LocalDB, to_curie


def make_pattern(id_: str, sk: dict, generate_pattern_from_label):
    """Given an ESCO skill entry in the dataframe, create a pattern for the matcher.

    The entry has the following fields:
    - label: the preferred label
    - altLabel: a list of alternative labels
    - the skillType: e.g. knowledge, skill, ability

    The logic uses some euristic to decide whether to use the preferred label or the alternative labels.
    """
    if sk["skillType"] != "skill":
        raise ValueError(f"Expected a skill, got {sk['skillType']}")

    label = sk["label"]
    pattern = [{"LOWER": label.lower()}] if len(label) > 3 else [{"TEXT": label}]
    patterns = [pattern]
    altLabel = [sk["altLabel"]] if isinstance(sk["altLabel"], str) else sk["altLabel"]
    for alt in altLabel:
        candidate = generate_pattern_from_label(alt)
        # Avoid duplicates.
        if candidate not in patterns:
            patterns.append(candidate)

    return to_curie(id_), patterns


def esco_matcher(skills, generate_pattern_from_label):
    # Create the patterns for the matcher
    return dict(
        make_pattern(id_, ski, generate_pattern_from_label=generate_pattern_from_label)
        for id_, ski in skills.to_dict(orient="index").items()
        if ski["skillType"] == "skill"
    )


@pytest.fixture
def generate_pattern_from_label(nlp):
    def f(label):
        if len(label) <= 3:
            return [{"TEXT": label}]

        if 1 < len(label.split()) < 4:
            return [{"LOWER": x} for x in label.lower().split()]

        if nlp is None:
            return [{"LOWER": label.lower()}]

        nlp(label)
        logging.warning(f"Generating pattern for: {label}")
        # If the label is longer than 3 characters and has more than 3 words,
        #   use spacy to generate the pattern.
        raise NotImplementedError

    yield f  #


@pytest.fixture
def skills():
    """Return the first 10 skills from the local database."""
    db = LocalDB()
    skills = [
        "collaborate with engineers",
        "deploy cloud resource",
        "design cloud architecture",
        "design cloud networks",
        "plan migration to cloud",
        "automate cloud tasks",
        "coordinate engineering teams",
        "design database in the cloud",
        "design for organisational complexity",
        "develop with cloud services",
        "do cloud refactoring",
    ]
    yield db.skills[db.skills.label.str.lower().isin(skills)]


@pytest.fixture(params=(None, "merge_noun_chunks"))
def nlp(request):
    n = spacy.load("en_core_web_trf")
    if request.param is not None:
        n.add_pipe(request.param)
    logging.warning(f"Using {n}")
    yield n


def test_esco_matcher(
    skills,
    generate_pattern_from_label,
):
    esco_matcher(skills, generate_pattern_from_label)
    raise NotImplementedError
