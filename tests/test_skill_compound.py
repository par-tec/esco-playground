import logging
from pathlib import Path

import pytest
import spacy
import yaml

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


def find_root(doc, nlp):
    for prefix in ("", "to "):
        doc = nlp(prefix + doc.text)
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                return token


def find_obj(token):
    for child in token.children:
        if child.dep_ in ("prep"):
            return find_obj(child)

        if child.dep_ in ("dobj", "pobj", "nsubj"):
            return child
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


def find_compound(token):
    if token:
        yield token.lemma_
        if token.children:
            for child in token.children:
                if child.dep_ == "compound":
                    yield from find_compound(child)


@pytest.fixture
def generate_pattern_from_label(named_nlp):
    name, nlp = named_nlp

    def f_compound(label):
        default = [{"LOWER": label.lower()}]

        if len(label) <= 3:
            return [{"TEXT": label}]

        if 1 < len(label.split()) < 3:
            return [{"LOWER": x} for x in label.lower().split()]

        if nlp is None:
            # If no nlp is provided, return the label as is.
            return default

        # If the label is longer than 3 characters and has more than 3 words,
        #   use spacy to generate the pattern.
        logging.warning(f"Generating dependency pattern for: {label}")
        doc = nlp(label)

        verb = find_root(doc, nlp)
        if verb is None:
            return default

        obj = find_obj(verb)
        if obj is None:
            return default

        compound = list(find_compound(obj))

        if len(compound) != 2:
            log.warning(
                f"Compound is <2 or >3: {compound} for {label}. Use displacy or other tools to implement this case."
            )
            return default
        return [
            {"RIGHT_ID": "base", "RIGHT_ATTRS": {"LEMMA": obj.lemma_}},
            {
                "LEFT_ID": "base",
                "RIGHT_ID": "compound",
                "REL_OP": ">>",
                "RIGHT_ATTRS": {"LEMMA": compound[1]},
            },
        ]

    f = f_compound
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


@pytest.fixture(
    params=(
        None,
        # "merge_noun_chunks",  # This is not needed for the dependency matcher, because we are resolving dependencies between tokens instead of merging them.  # noqa: E501
        "merge_entities",
    )
)
def named_nlp(request):
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
    matcher = spacy.matcher.DependencyMatcher(nlp.vocab)
    for name, patterns in patterns.items():
        for pattern in patterns:
            if not pattern[0].get("RIGHT_ID"):
                continue

            matcher.add(
                name, [pattern]
            )  # Does it add the pattern to the matcher, or does it override the previous one?

    yield name, nlp, matcher


DATADIR = Path(__file__).parent / "data"
TESTFILE_YAML = DATADIR / "test_cloud_skill.yml"
TESTCASES = yaml.safe_load(TESTFILE_YAML.read_text())["tests"]


@pytest.mark.parametrize(
    "text,expected_skills",
    [(tc["text"], tc["skills"]) for tc in TESTCASES],
)
def test_esco_dependency_matcher(nlp_e, text, expected_skills):
    name, nlp_e, matcher = nlp_e
    doc = nlp_e(text)

    actual_skills = find_skills(doc, matcher)
    assert set(actual_skills) >= set(expected_skills)


def find_skills(doc, matcher):
    matches = matcher(doc)
    assert matches
    skills = dict()
    for match_id, token_ids in matches:
        s, e = sorted(token_ids)
        matched_span = doc[s:e]
        skill_id = doc.vocab.strings[match_id]
        log.warning(f"Identified {skill_id} in {matched_span}")
        skills[skill_id] = {"uri": skill_id, "text": matched_span}
    return skills


@pytest.mark.parametrize(
    "text",
    [
        "create networks in the cloud",
        "Create cloud networks",
        "Design networks in cloud environments",
    ],
)
def test_esco_dependency_matcher_working(nlp_e, text):
    name, nlp_e, matcher = nlp_e
    doc = nlp_e(text)

    matches = find_skills(doc, matcher)
    assert matches


@pytest.mark.parametrize(
    "text",
    [
        "Create networks in the cloud",
        "create networks in the cloud.",
    ],
)
def test_esco_dependency_matcher_should_process_punctuation(nlp_e, text):
    name, nlp_e, matcher = nlp_e
    doc = nlp_e(text)
    assert matcher._patterns
    matches = find_skills(doc, matcher)
    assert matches
