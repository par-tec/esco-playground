import pytest
import spacy
from spacy.matcher import Matcher


@pytest.fixture
def nlp():
    return spacy.load("en_core_web_trf")


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

CUSTOMER_SATISFACTION_TEXTS = [
    "I have a long experience in guaranteeing customer satisfaction ",
    "I have a long experience in providing satisfaction ",
    "I have a long experience in assure customer satisfaction and promise the customer satisfaction ",
]


@pytest.mark.parametrize("text", CUSTOMER_SATISFACTION_TEXTS)
@pytest.mark.parametrize("pattern", CUSTOMER_SATISFACTION_PATTERNS.values())
def test_can_match_customer_satisfaction(nlp, text, pattern):
    doc = nlp(text)
    matcher = Matcher(nlp.vocab)
    matcher.add("MatcherSkill", [pattern])
    matches = matcher(doc)
    for match_id, start, end in matches:
        matched_span = doc[start:end]
        print(matched_span)
        assert matched_span.text == " ".join(pattern)


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
    for match_id, start, end in matches:
        matched_span = doc[start:end]
        print(matched_span)
        assert matched_span.text == " ".join(pattern)
