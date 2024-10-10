"""
Extracts and analyzes skills from CV text using ESCO classifications.

Features:
- Entity counting and skill extraction via NER and neural search
- Integration with ESCO database
- Sentence-level skill analysis

Note: Avoid personal data in CV text to prevent NER confusion.

Usage: Import EscoCV class to process CV text and extract skills.
"""

import logging

import esco

log = logging.getLogger(__name__)


def entity_counter(entities: list):
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


class EscoCV:
    """
    A CV skill extractor.

    The text should not contain personal data,
    since this may confuse the NER model
    (e.g., the text "address: Java street" may be recognized as a skill).
    """

    def __init__(self, ner, text=None, doc=None) -> None:
        self.ner = ner

        if doc:
            self.doc = doc
            self.text = doc.text
        else:
            self.text = text
            self.doc = self.ner.model(text)

        self._all_skills = None
        self._ner_skills = None
        self.sentences = []

    def entities(self):
        """
        @return a dict containing entities and their count.
        The dict has two keys:
        - 'entities': a list of dicts, each representing an entity with:
            - 'start': start character index
            - 'end': end character index
            - 'label': entity label
            - 'text': entity text
            - 'id': entity ID
        - 'count': total number of entities found
        """
        return {
            "entities": [
                {
                    "start": e.start_char,
                    "end": e.end_char,
                    "label": e.label_,
                    "text": e.text,
                    "id": e.ent_id_,
                }
                for e in self.doc.ents
                if e.label_ in self.ner.labels
            ],
            "count": len(self.doc.ents),
        }

    def ner_skills(self, force=False) -> dict:
        """
        Infer skills from a set of entities.

        @return a dict of skills with the number of occurrencies, e.g.

          {
            uri: {count: 1, label: "label"},
            ...
          }
        """
        if self._ner_skills and not force:
            return self._ner_skills
        ret = {}
        ents = entity_counter(self.entities()["entities"])
        for k, e in ents.items():
            if e["label"] == "ESCO":
                uri = esco.from_curie(k)
                label = self.ner.db.get_label(uri)
                ret[uri] = {"label": label, "count": e["count"], "source": "ner"}
            elif e["label"] == "PRODUCT":
                product_label = k.lower()
                skills = self.ner.db.search_products(
                    {
                        product_label,
                    }
                )
                for skill in skills:
                    ret[skill["uri"]] = {"label": skill["label"], "count": e["count"]}
            else:
                log.debug("Ignoring other labels: %s", e["label"])
        self._ner_skills = ret
        return ret

    def skills_by_sentence(self, force=False):
        """
        @param text: the text to search for.
        @param params: additional parameters to pass to the neural database.
            Currently supported:
            - k: the number of entries to retrieve
            - score_threshold: the score threshold
        @return a dict of skills related to a set of product labels.
        """
        if self.sentences and not force:
            return self.sentences

        if self.ner.tokenizer:
            sentences = self.ner.tokenizer(self.text)
        else:
            sentences = (str(t) for t in self.doc.sents)

        for sentence in sentences:
            txt = str(sentence).strip()
            if not txt:
                continue
            if len(txt.split()) < 5:
                continue

            skills_ = [
                s
                for s in self.ner.db.search_neural(
                    txt,
                    k=7,
                )
                if self.ner.db.get(s["uri"])["skillType"] == "skill"
            ]
            ret = {
                "text": txt,
                "skills": skills_,
            }
            self.sentences.append(ret)
        return self.sentences

    def skills(self, force=False):
        """
        @param force: if True, force recalculation even if results are cached.
        @return a dict of skills, where:
            - Key: skill URI
            - Value: dict containing:
                - 'label': skill label
                - 'count': number of occurrences
                - 'score': relevance score (if available)
        Note:
            - Skills with labels shorter than 5 characters are skipped.
            - For duplicate skills, count is incremented and score is maximized.
        """
        if self._all_skills and not force:
            return self._all_skills
        ner_skills = {} | self.ner_skills(force=force)
        for sentence in self.skills_by_sentence(force=force):
            for skill in sentence["skills"]:
                uri = skill["uri"]

                if len(skill["label"]) < 5:
                    log.debug(
                        "Skipping %s. Too short for neural search.", skill["label"]
                    )
                    continue
                if uri not in ner_skills:
                    skill.setdefault("count", 1)
                    ner_skills[uri] = skill
                else:
                    ner_skills[uri]["count"] += 1
                    ner_skills[uri]["score"] = max(
                        skill.get("score", 0), ner_skills[uri].get("score", 0)
                    )
        self._all_skills = ner_skills
        return ner_skills
