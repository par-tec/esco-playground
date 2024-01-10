import logging
from functools import reduce

import spacy

import esco

log = logging.getLogger(__name__)


class Recognizer:
    """
    This is a spacy-aware esco skill recognizer.
    """

    def __init__(self, config=None):
        if config:
            self.model = config["model"]
            self.labels = config["labels"]
            return

        self.model = spacy.load("en_core_web_trf_esco_ner")
        self.model = spacy.load("generated/en_core_web_trf_esco_ner")
        self.labels = ("ESCO", "PRODUCT", "LANGUAGE", "LAW")
        self.skills = esco.load_skills(source="sparql")
        self.db = esco.DB(source="sparql")

    def recognize_entities(self, text, context=None):
        """
        Recognize entities in the text using the NER model.
        """

        doc = self.model(text)
        return {
            "entities": [
                {
                    "start": e.start_char,
                    "end": e.end_char,
                    "label": e.label_,
                    "text": e.text,
                    "id": e.ent_id_,
                }
                for e in doc.ents
                if e.label_ in self.labels
                and e.start_char
                > 100  # Ignore the first part of the CV, since it may contain only personal data.
            ],
            "count": len(doc.ents),
        }

    @staticmethod
    def entity_counter(entities: list, context=None):
        """
        @return a dict of entities with the number of occurrencies.
        { "identifier": {"label": "PRODUCT", "count": 1, "id": ID, "text": text}}}
        """
        counter = {}
        for e in entities:
            k = e.get("id") if e.get("id") else e["text"]
            if k not in counter:
                counter[k] = {
                    "label": e["label"],
                    "count": 1,
                    "id": e.get("id"),
                    "text": e["text"],
                }
            else:
                counter[k]["count"] += 1
        return counter

    def search_skills(self, entities: list, context=None) -> dict:
        """
        Infer skills from a set of entities.
        """
        entities = self.entity_counter(entities)
        ret = {}
        for k, e in entities.items():
            if e["label"] == "ESCO":
                uri = esco.from_curie(k)
                label = self.db.get_label(uri)
                ret[uri] = {"label": label, "count": e["count"]}
            elif e["label"] == "PRODUCT":
                product_label = k.lower()
                skills = self.db.search_products(
                    {
                        product_label,
                    }
                )
                for uri, s in skills.items():
                    ret[uri] = {"label": s["label"], "count": e["count"]}
            else:
                log.debug(f"Ignoring other labels: {e['label']}")
        return ret

    def infer_skills_from_skill(self, skill_uri: str, context=None):
        """
        Infer skills from a set of skills.
        """
        return esco.infer_skills_from_skill(skill_uri)

    def infer_skills_from_skills(self, skills: list, context=None):
        """
        Infer skills from a set of skills.
        """
        inferred_skills = (
            esco.infer_skills_from_skill(skill_uri) for skill_uri in skills
        )
        aggregated_skills = reduce(lambda x, y: x | y, inferred_skills, {})
        return aggregated_skills
