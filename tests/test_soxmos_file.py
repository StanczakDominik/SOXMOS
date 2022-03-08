# -*- coding: utf-8 -*-
def test_dataset_sizes(test_file):
    ds = test_file.dataset
    assert ds.sizes == {"ch": 2, "pixel": 1024, "Time": 50}


def test_dataset_variables(test_file):
    ds = test_file.dataset
    for var in ["Count", "FilteredCount", "Rough_wavelength"]:
        assert var in ds


def test_config(test_file):
    print(test_file.config)
    for key in [
        "name",
        "shotno",
        "date",
        "dimno",
        "dimname",
        "dimsize",
        "dimunit",
        "valno",
        "valname",
        "valunit",
    ]:
        assert key in test_file.config["Parameters"]


def test_description(test_file):
    assert test_file.description == "SOXMOSTestShot #66642069 @'02/04/2005 21:37'"


def test_spectrogram_actually_plots(test_file, tmp_path):
    test_file.plot_spectrogram().fig.savefig(tmp_path / f"spectrogram.png")
    test_file.plot_spectrogram(vmax=1000).fig.savefig(
        tmp_path / f"spectrogram_max1000.png"
    )


def test_spectrum_actually_plots(test_file, tmp_path):
    for it in range(5, 45, 10):
        time = test_file.dataset.isel(Time=it).Time.item()
        test_file.plot_spectrum(time).fig.savefig(tmp_path / f"spectrum_t_{time}.png")


def test_global_timetrace_actually_plots(test_file, tmp_path):
    test_file.plot_global_timetrace().fig.savefig(tmp_path / f"global_timetrace.png")
