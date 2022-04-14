# -*- coding: utf-8 -*-
from pathlib import Path

import pytest

from SOXMOS import SOXMOSFile

savgol_settings = {
    (19.5177, 34.3053): {"window_length": 13, "polyorder": 4},
    (75.7912, 101.6328): {"window_length": 14, "polyorder": 4},
}

@pytest.fixture
def test_path(shared_datadir):
    return shared_datadir / "test_data.dat"


@pytest.fixture
def test_file(test_path):
    return SOXMOSFile(test_path, savgol_settings)
