import spacy
from connexion import ProblemException

import esco

nlp_e = spacy.load("en_core_web_trf_esco_ner")
skills = esco.load_skills(source="js")
skills["uri"] = skills.index
skills["id"] = skills.index.str.split("/").str[-1]
RETURN_FIELDS = [
    "uri",
    "label",
    "description",
    "altLabel",
    "skillType",
]


def list_skills(limit=10, q=None, user=None):
    if q:
        ret = skills[skills.text.str.contains(q, case=False)]
    else:
        ret = skills
    ret = ret[RETURN_FIELDS].head(limit)

    return {
        "count": len(ret),
        "items": ret.to_dict(orient="records"),
    }, 200


def show_skill(uri, user=None):
    ret = skills[skills.id.str.startswith(uri)]
    if len(ret) == 0:
        raise ProblemException(
            status=404,
            title="Skill not found",
            detail=f"The skill {uri} was not found in the database",
        )
    ret = ret[RETURN_FIELDS].head(10).to_dict(orient="records")
    if len(ret) > 1:
        raise ProblemException(
            status=400,
            title="Multiple skills found",
            detail=f"At least {len(ret)} skills found for {uri}",
            ext={"skills": ret},
        )
    return ret[0], 200


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
