import spacy
from connexion import ProblemException

nlp_e = spacy.load("en_core_web_trf_esco_ner")


def recognize_entities(body, user=None):
    if "text" not in body:
        raise ProblemException(
            status=400,
            title="Missing text",
            detail="The request body must contain a text field",
        )
    text = body["text"]

    doc = nlp_e(text)
    return {
        "entities": [
            {
                "start": e.start_char,
                "end": e.end_char,
                "label": e.label_,
                "text": e.text,
            }
            for e in doc.ents
            if e.label_ in ("ESCO", "PRODUCT", "LANGUAGE", "LAW")
            and e.start_char
            > 100  # Ignore the first part of the CV, since it may contain only personal data.
        ],
        "count": len(doc.ents),
    }, 200
