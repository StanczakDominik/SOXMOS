import pathlib
import pandas as pd
import configparser
import itertools
from functools import cached_property
import numpy as np
import matplotlib.pyplot as plt
import xarray
from scipy import signal


class SOXMOSFile:
    def __init__(
        self, path: pathlib.Path, *, savgol_window_length=13, savgol_polyorder=4
    ):
        self.path = path
        self.config = parse_config(path)
        self.savgol_settings = dict(
            window_length=savgol_window_length, polyorder=savgol_polyorder
        )

    def __repr__(self, *args, **kwargs):
        c = self.config
        p = c["Parameters"]
        return f"{__class__.__name__}({p['Name']} #{p['ShotNo']} @{p['Date']})"

    def plot_spectrogram(self):
        self.dataset.set_coords("Rough_wavelength").Count.plot(
            row="ch", x="Rough_wavelength", y="Time"
        )
        plt.suptitle(self)

    def plot_spectrum(self, time):
        self.dataset.sel(Time=time, method="nearest").set_coords(
            "Rough_wavelength"
        ).FilteredCount.plot.line(x="Rough_wavelength", hue="ch")
        plt.suptitle(self.path)
        plt.tight_layout()
        plt.ylabel(f"Filtered count, Savgol params: {self.savgol_settings}")

    @property
    def dataframe(self):
        df = pd.read_table(
            self.path, comment="#", header=None, sep=",\s", engine="python"
        )
        columns = list(
            itertools.chain.from_iterable(
                [
                    list_from_config(self.config["Parameters"][category_name])
                    for category_name in ["DimName", "ValName"]
                ]
            )
        )
        df.columns = columns
        return df

    @cached_property
    def dataset(self):
        ds = self.dataframe.to_xarray()
        # https://stackoverflow.com/questions/70861487/turn-1d-indexed-xarray-variables-into-3d-coordinates/70873363#70873363
        ds = (
            ds.assign_coords(
                {
                    "index": pd.MultiIndex.from_arrays(
                        [ds.Time.values, ds.ch.values, ds.pixel.values],
                        names=["Time", "ch", "pixel"],
                    )
                }
            )
            .drop_vars(["Time", "ch", "pixel"])
            .unstack("index")
        )
        ds["FilteredCount"] = xarray.apply_ufunc(
            signal.savgol_filter,
            ds.Count,
            kwargs=self.savgol_settings,
            input_core_dims=[["pixel"]],
            output_core_dims=[["pixel"]],
        )
        ds["FilteredCount"].attrs.update(self.savgol_settings)
        ds = ds.set_coords("Rough_wavelength")
        ds["Time"].attrs["units"] = "s"
        ds["Rough_wavelength"].attrs["units"] = "nm"
        a = ds.attrs
        for key, value in self.config["Comments"].items():
            a[key] = value
        for key in ["Name", "ShotNo", "Date"]:
            a[key] = self.config["Parameters"][key]
        return ds


def list_from_config(field):
    return field.replace("'", "").split(", ")


def parse_config(path):
    # TODO inefficient to read this twice
    lines = path.read_text().splitlines()
    parsedlines = list(filter(lambda line: line.startswith("# "), lines))
    parsedlines = [line[2:] for line in parsedlines]
    configstr = "\n".join(parsedlines)
    config = configparser.ConfigParser()
    config.read_string(configstr)
    return config
