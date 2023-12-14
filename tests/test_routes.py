from pathlib import Path

import pytest
from connexion import ProblemException

from api.routes import recognize_entities

DATADIR = Path(__file__).parent / "data"


def test_recognize_entities_with_text():
    body = {"text": (DATADIR / "rpolli.txt").read_text()}
    result = recognize_entities(body)
    assert isinstance(result, str), "The result should be a string (JSON)."
    raise NotImplementedError("The test is incomplete")


def test_recognize_entities_without_text():
    body = {}
    with pytest.raises(ProblemException) as exc_info:
        recognize_entities(body)
    assert exc_info.value.status == 400
    assert exc_info.value.title == "Missing text"
    assert exc_info.value.detail == "The request body must contain a text field"
