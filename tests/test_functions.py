# -*- coding: utf-8 -*-
from SOXMOS.SOXMOSFile import parse_config, parse_dataframe, parse_dataset

savgol_settings = {
    (19.5177, 34.3053): {"window_length": 13, "polyorder": 4},
    (75.7912, 101.6328): {"window_length": 14, "polyorder": 4},
}

def test_dataset_dataframe_parses(test_path, shared_datadir):
    config = parse_config(test_path)
    df = parse_dataframe(test_path, config)
    ds = parse_dataset(df, config, savgol_settings=savgol_settings)

    contents = shared_datadir / "saved_dataset.txt"
    print(str(ds))
    assert str(ds) == contents.read_text().strip()
