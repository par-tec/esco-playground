"""
Test utilities for creating and managing temporary directories.
"""

import shutil
from pathlib import Path
from tempfile import mkdtemp

import pytest


@pytest.fixture(scope="module")
def tmpdir():
    """
    Create a temporary directory for testing.

    Yields:
        Path: The path to the temporary directory.
    """
    dpath = Path(mkdtemp())
    if dpath.is_dir():
        shutil.rmtree(dpath.as_posix())
    dpath.mkdir(exist_ok=False, parents=True)
    yield dpath
    shutil.rmtree(dpath.as_posix())
