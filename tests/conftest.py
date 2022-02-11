from pathlib import Path

import pytest

from SOXMOS import SOXMOSFile


@pytest.fixture
def test_file():
    return SOXMOSFile(Path("tests") / "test_data.dat")
