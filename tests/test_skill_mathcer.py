import logging
from collections import defaultdict

import pytest
import spacy
import yaml
from spacy.matcher import Matcher

log = logging.getLogger(__name__)


CUSTOMER_SATISFACTION_TEXTS = [
    "I have a long experience in guaranteeing customer satisfaction ",
    "I have a long experience in providing satisfaction ",
    "I have a long experience in assure customer satisfaction and promise the customer satisfaction ",
    "Dynamic and results-driven professional with a proven record of enhancing customer satisfaction and retention. Proficient in managing customer relations, identifying client needs, and providing tailored solutions. Skilled in resolving complex customer service issues, leading to a significant improvement in customer ratings. Adept at training and leading teams to deliver exceptional customer service. Committed to creating a positive and productive environment that promotes excellence and fosters customer loyalty.",
]

CUSTOMER_SATISFACTION_PATTERNS = dict(
    pattern=[
        {"LEMMA": {"IN": ["guarantee", "assure", "ensure", "provide", "promise"]}},
        {"POS": "DET", "OP": "?"},
        {"TEXT": "customer", "OP": "?"},
        {"TEXT": "satisfaction"},
    ],
    pattern1=[
        {"TEXT": "customer"},
        {"TEXT": "satisfaction"},
        {
            "LEMMA": {"IN": ["guarantee", "assure", "ensure", "provide", "promise"]},
            "OP": "?",
        },
    ],
    pattern2=[
        {
            "LEMMA": {"IN": ["guarantee", "assure", "ensure", "provide", "promise"]},
            "OP": "?",
        },
        {"POS": "DET", "OP": "?"},
        {"TEXT": "customer", "OP": "?"},
        {"TEXT": "satisfaction"},
        {
            "LEMMA": {"IN": ["guarantee", "assure", "ensure", "provide", "promise"]},
            "OP": "?",
        },
    ],
)


@pytest.fixture
def nlp():
    return spacy.load("en_core_web_trf")


@pytest.fixture(params=CUSTOMER_SATISFACTION_PATTERNS.items())
def customer_satisfaction_matcher(request, nlp):
    m = Matcher(nlp.vocab)
    m_name, m_pattern = request.param
    m.add(m_name, [m_pattern])
    yield m_name, m


@pytest.fixture(scope="module")
def stats():
    """
    This fixture is used to collect and print the stats of the number of matches.
    """
    s = defaultdict(int)
    yield s
    print("\n\n", yaml.safe_dump(dict(s)))


def test_can_match_customer_satisfaction(nlp, customer_satisfaction_matcher, stats):
    docs = nlp.pipe(CUSTOMER_SATISFACTION_TEXTS)
    m_name, customer_satisfaction_matcher = customer_satisfaction_matcher
    for doc in docs:
        matches = customer_satisfaction_matcher(doc)
        matched_spans = []
        for match_id, start, end in matches:
            matched_span = doc[start:end]
            log.info(matched_span)
            matched_spans.append(matched_span.text)
        stats[m_name] += bool(len(matched_spans))
    for k, v in stats.items():
        assert v > 0


CLOUD_PATTERNS = dict(
    pattern=[
        {
            "LEMMA": {
                "IN": [
                    "design",
                    "deploy",
                    "plan",
                    "develop",
                    "manage",
                    "automate",
                    "implement",
                    "migrate",
                ]
            },
            "OP": "+",
        },
        {"POS": "ADJ", "OP": "?"},
        {"POS": "NOUN", "OP": "?"},
        {"POS": "DET", "OP": "?"},
        {"LOWER": "cloud"},
        {"POS": "NOUN", "OP": "?"},
    ]
)
CLOUD_TEXTS = [
    "manage and maintain company IT infrastructure implement hybrid cloud solutions and design cloud architecture and plan migration cloud optimize",
]


@pytest.mark.parametrize("text", CLOUD_TEXTS)
@pytest.mark.parametrize("pattern", CLOUD_PATTERNS.values())
def test_can_match_cloud(nlp, text, pattern):
    doc = nlp(text)
    matcher = Matcher(nlp.vocab)
    matcher.add("MatcherSkill", [pattern])
    matches = matcher(doc)
    matched_spans = []
    for match_id, start, end in matches:
        matched_span = doc[start:end]
        log.info(matched_span)
        matched_spans.append(matched_span.text)
    assert matched_spans
