import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_trf")
matcher = Matcher(nlp.vocab)
matched_sents = []

"""
'assure customer satisfaction',
'customer satisfaction guarantee',
'ensure customer satisfaction',
'guarantee customer satisfaction',
'guaranteeing customer satisfaction',
'promise customer satisfaction',
'provide customer satisfaction',
'to guarantee customer satisfaction']
"""

pattern = [
    {"LEMMA": {"IN": ["guarantee", "assure", "ensure", "provide", "promise"]}},
    {"POS": "DET", "OP": "?"},
    {"TEXT": "customer", "OP": "?"},
    {"TEXT": "satisfaction"},
]
pattern1 = [
    {"TEXT": "customer"},
    {"TEXT": "satisfaction"},
    {
        "LEMMA": {"IN": ["guarantee", "assure", "ensure", "provide", "promise"]},
        "OP": "?",
    },
]
pattern2 = [
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
]
matcher.add("MatcherSkill", [pattern2])  # add pattern
doc = nlp("I have a long experience in guaranteeing customer satisfaction ")
doc1 = nlp("I have a long experience in providing satisfaction ")
doc2 = nlp(
    "I have a long experience in assure customer satisfaction and promise the customer satisfaction "
)
matches = matcher(doc)
matches1 = matcher(doc1)
matches2 = matcher(doc2)
print("*************test_1*************")
for match_id, start, end in matches:
    matched_span = doc[start:end]
    print(matched_span)
print("*************test_2*************")
for match_id, start, end in matches1:
    matched_span = doc1[start:end]
    print(matched_span)
print("*************test_3*************")
for match_id, start, end in matches2:
    matched_span = doc2[start:end]
    print(matched_span)

# cloud position

matched_sents = []

pattern3 = [
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
matcher.add("CloudSkill", [pattern3])
doc = nlp(
    "manage and maintain company IT infrastructure implement hybrid cloud solutions and design cloud architecture and plan migration cloud optimize"
)
matches = matcher(doc)

for match_id, start, end in matches:
    matched_span = doc[start:end]
    print(matched_span)
