import pytest
import spacy
from spacy.language import Language
from spacy.matcher import Matcher
from spacy.tokens import Span

import model


@pytest.mark.skip(reason="Superseeded by entity_recognizer")
def test_add_esco_spacy_pipeline():
    nlp = spacy.load("en_core_web_trf")
    matcher = spacy.matcher.Matcher(nlp.vocab)

    # Define the custom component
    @Language.component("esco_component")
    def esco_component_function(doc):
        # Apply the matcher to the doc
        matches = matcher(doc)
        # Create a Span for each match and assign the label "ESCO"
        spans = [
            Span(doc, start, end, label="ESCO") for match_id, start, end in matches
        ]
        # Overwrite the doc.ents with the matched spans
        # FIXME: there are overlaps in the spans
        doc.ents = spans  # tuple(spans)
        return doc

    nlp.add_pipe("esco_component", after="ner")


def test_esco_matcher():
    m = model.esco_matcher()
    validate_patterns = False
    if validate_patterns:
        # If patterns are not valid, the matcher will raise an error.

        nlp_test = spacy.blank("en")

        m1 = Matcher(nlp_test.vocab, validate=True)
        for pid, patterns in m.items():
            m1.add(pid, patterns)
