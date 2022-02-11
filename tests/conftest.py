from pathlib import Path

import pytest

from SOXMOS import SOXMOSFile


@pytest.fixture
def test_file():
    return SOXMOSFile(Path(__file__).parent / "test_data.dat")
