from functools import reduce

import spacy

import esco


class Recognizer:
    def __init__(self, config=None):
        if config:
            self.model = config["model"]
            self.labels = config["labels"]
            return

        self.model = spacy.load("en_core_web_trf_esco_ner")
        self.labels = ("ESCO", "PRODUCT", "LANGUAGE", "LAW")
        self.skills = esco.load_skills()

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
                }
                for e in doc.ents
                if e.label_ in self.labels
                and e.start_char
                > 100  # Ignore the first part of the CV, since it may contain only personal data.
            ],
            "count": len(doc.ents),
        }

    def infer_skills(self, entities: list, context=None):
        """
        Infer skills from a set of entities.
        """
        product_labels = {
            e["text"].lower()
            for e in entities
            if e["label"]
            in (
                "ESCO",
                "PRODUCT",
            )
        }
        return esco.infer_skills_from_products(
            self.skills, product_labels=product_labels
        )

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
