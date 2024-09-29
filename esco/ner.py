"""
ESCO Skill Recognition Module

Provides NLP-based recognition of ESCO skills in text using spaCy and an ESCO database.
Main component is the Ner class, a spaCy-aware ESCO skill recognizer.
"""

import logging
from typing import Collection

import spacy

from esco.cv import EscoCV

log = logging.getLogger(__name__)


class Ner:
    """
    This is a spacy-aware esco skill recognizer.

    It uses a spacy model to recognize entities in a text,
    and then it uses the esco database to infer skills from the entities.
    """

    def __init__(
        self,
        db,
        model_name_or_path: str = "en_core_web_trf",
        labels: tuple = ("ESCO", "PRODUCT", "LANGUAGE", "LAW"),
        tokenizer=None,
    ):
        self.db = db
        self.model = spacy.load(model_name_or_path)
        self.labels = labels
        self.tokenizer = tokenizer

    def pipe(self, texts) -> Collection[EscoCV]:
        """
        @param texts a list of texts
        """
        for doc in self.model.pipe(texts):
            yield EscoCV(ner=self, doc=doc)

    def __call__(self, text: str) -> EscoCV:
        """
        @param text a string
        """
        return EscoCV(ner=self, text=text)
