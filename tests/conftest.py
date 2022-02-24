from pathlib import Path

import pytest

from SOXMOS import SOXMOSFile


@pytest.fixture
def test_path(shared_datadir):
    return shared_datadir / "test_data.dat"

@pytest.fixture
def test_file(test_path):
    return SOXMOSFile(test_path)
