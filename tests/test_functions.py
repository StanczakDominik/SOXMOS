from SOXMOS.SOXMOSFile import parse_config, parse_dataframe, parse_dataset

def test_dataset_dataframe_parses(test_path, shared_datadir):
    config = parse_config(test_path)
    df = parse_dataframe(test_path, config)
    ds = parse_dataset(df, config, savgol_settings=dict(window_length=13, polyorder=2))

    contents = shared_datadir / "saved_dataset.txt"
    print(str(ds))
    assert str(ds) == contents.read_text().strip()

