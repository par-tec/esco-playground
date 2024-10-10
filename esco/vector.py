"""
Manages a vector database for ESCO embeddings using Qdrant,
providing functionalities for storage, retrieval, and search.
"""

from pathlib import Path

import pandas as pd
from langchain.schema import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores.qdrant import Qdrant
from qdrant_client.models import Batch

MODEL_PARAMETERS = {
    "paraphrase-albert-small-v2": {"score_threshold": 0.25, "k": 10},
    "all-MiniLM-L12-v2": {"score_threshold": 0.3, "k": 10},
}


class ReadOnlyQdrant(Qdrant):
    """
    ReadOnlyQdrant class extends the Qdrant class to provide read-only
    functionality, disabling the addition of new texts to the database.
    """

    @staticmethod
    def add_texts(*args, **kwargs):  # pylint: disable=unused-argument
        return


class VectorDB:
    """
    A database containing the embeddings of ESCO.

    By default, the database is created with the following parameters:
        - path: all-MiniLM-L12-v2
        - collection_name: esco

    To connect to a remote qdrant instance, use the following parameters:
        - url: http://localhost:18890
        - collection_name: esco
    """

    MODEL_NAME = "all-MiniLM-L12-v2"

    def __init__(
        self, force_recreate=False, model_params=None, skills=None, config=None
    ) -> None:
        self.model_name = self.MODEL_NAME
        self.model_params = model_params or MODEL_PARAMETERS.get(self.model_name)
        self.embedding_function = SentenceTransformerEmbeddings(
            model_name=self.model_name
        )
        self.config = config or {
            "path": f"qdrant-esco-{self.model_name}",
            "collection_name": "esco-skills",
        }

        if idx_path := self.config.get("path"):
            if not force_recreate and not Path(idx_path).exists():
                raise FileNotFoundError(f"Vector database not found at {idx_path}")
            Path(idx_path).mkdir(parents=True, exist_ok=True)

        one_document = [
            Document(
                page_content=skill["text"],
                metadata={"label": skill["label"], "uri": uri},
            )
            for uri, skill in skills[:1].to_dict(orient="index").items()
        ]
        self.qdrant = ReadOnlyQdrant.from_documents(
            one_document,
            self.embedding_function,
            force_recreate=force_recreate,
            **self.config,
        )
        if force_recreate:
            self._recreate(skills)

    def _recreate(self, skills: pd.DataFrame):
        """
        Recreates the vector database by upserting skill embeddings from the
        provided DataFrame, raising an error if the operation fails.
        """
        points = Batch(
            ids=[x.split("/")[-1] for x in skills.index.values],
            payloads=[
                {"metadata": {"label": label, "uri": i}, "page_content": t}
                for t, label, i in zip(
                    skills.text.values, skills.label.values, skills.index.values
                )
            ],
            vectors=skills.vector.values,
        )
        ret = self.qdrant.client.upsert(
            collection_name=self.config["collection_name"], points=points
        )
        if ret.status.value != "completed":
            raise ValueError(f"Could not add points to Qdrant: {ret}")
        return ret

    def scroll(self, limit=10000):  # pylint: disable=unused-argument
        """
        Retrieves a specified number of documents from the vector database,
        defaulting to 10,000, using the Qdrant client.
        """
        return self.qdrant.client.scroll(
            collection_name=self.config["collection_name"], limit=10000
        )

    def search(self, text, **params):
        """
        Performs a similarity search in the vector database using the provided text
        and additional parameters, returning the matching documents with their
        metadata and similarity scores.
        """
        params = {**self.model_params, **params}
        return [
            {
                "uri": x[0].metadata.get("uri") or x[0].metadata["id"],
                "label": x[0].metadata["label"],
                "score": x[1],
            }
            for x in self.qdrant.similarity_search_with_score(text, **params)
        ]

    def close(self):
        """
        function to close client
        """
        self.qdrant.client.close()
