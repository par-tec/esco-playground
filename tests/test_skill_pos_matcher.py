import logging

import pytest
import spacy

from esco import LocalDB, to_curie

log = logging.getLogger(__name__)


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


def find_root(doc):
    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            return token
    return None


def find_dobj(token):
    # Iterate over the children of the token
    # and return the first child with the dependency label 'dobj'
    # If you find 'adp' or 'prep' as a child, follow the link to the child of the child.
    for child in token.children:
        if child.dep_ in ("dobj", "pobj"):
            return child
        elif child.dep_ in ("prep", "adp"):
            return find_dobj(child)
    return None


def get_verb_obj_from_label(label, nlp):
    """
    Given a label, return the root verb and the direct object.

    This function should be improved to handle more complex cases,
    such as pobj and compound nouns.
    """
    log.debug(f"Processing label: {label}")
    doc = nlp(label)
    root = find_root(doc)
    if root is None:
        log.warning(f"Prepend 'to'... {doc}")
        doc = nlp(f"to {label}")
        root = find_root(doc)
    if root is None:
        log.error(f"Cannot find root for {doc}")
        return None, None

    dobj = find_dobj(root)
    if dobj is None:
        return None, None
    return root, dobj


@pytest.fixture
def generate_pattern_from_label(nlp_with_label):
    name, nlp = nlp_with_label

    def f(label):
        if len(label) <= 3:
            return [{"TEXT": label}]

        if 1 < len(label.split()) < 3:
            return [{"LOWER": x} for x in label.lower().split()]

        if nlp is None:
            # If no nlp is provided, return the label as is.
            return [{"LOWER": label.lower()}]

        # If the label is longer than 3 characters and has more than 3 words,
        #   use spacy to generate the pattern.
        logging.warning(f"Generating pattern for: {label}")
        root, obj = get_verb_obj_from_label(label, nlp)
        if root is None or obj is None:
            return [{"LOWER": label.lower()}]

        return [
            {
                "LEMMA": root.lemma_,
                "POS": "VERB",
            },
            {
                "LEMMA": obj.lemma_,
                "POS": "NOUN",
                # "DEP": "dobj",
            },
        ]

    yield name, f


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
def nlp_with_label(request):
    n = spacy.load("en_core_web_trf")
    if request.param is not None:
        n.add_pipe(request.param)
    logging.warning(f"Using {n}")
    yield request.param, n


@pytest.fixture
def nlp_e(skills, generate_pattern_from_label):
    name, generate_pattern_from_label = generate_pattern_from_label
    nlp = spacy.load("en_core_web_trf")
    patterns = esco_matcher(skills, generate_pattern_from_label)
    assert patterns
    assert len(patterns) >= 9
    ruler = nlp.add_pipe("entity_ruler", after="ner")
    ruler.add_patterns(
        [
            {"label": "ESCO", "pattern": pattern, "id": k}
            for k, p in patterns.items()
            for pattern in p
        ]
    )
    yield name, nlp


@pytest.mark.parametrize(
    "text", ["Create networks in the cloud.", "Create cloud networks."]
)
def test_esco_matcher(nlp_e, text):
    name, nlp_e = nlp_e
    doc = nlp_e(text)
    assert doc.ents
    assert len(doc.ents) > 0
    ent_labels = {ent.label_ for ent in doc.ents}
    assert "ESCO" in ent_labels
    # raise NotImplementedError