# -*- coding: utf-8 -*-
import configparser
import itertools
import os
import pathlib
from functools import cached_property

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray
from joblib import Memory
from scipy import signal

memflag = os.environ.get("TEST_SOXMOS", False)
location = None if memflag else "./cachedir"
memory = Memory(location, verbose=0)


class SOXMOSFile:
    def __init__(
        self, path: pathlib.Path, *, savgol_window_length=13, savgol_polyorder=4
    ):
        self.path = path
        self.config = parse_config(path)
        self.savgol_settings = dict(
            window_length=savgol_window_length, polyorder=savgol_polyorder
        )

    @property
    def description(self):
        c = self.config
        p = c["Parameters"]
        return f"{p['Name']} #{p['ShotNo']} @{p['Date']}"

    @property
    def shotid(self):
        c = self.config
        p = c["Parameters"]
        return p["ShotNo"]

    def __repr__(self, *args, **kwargs):
        return f"{self.__class__.__name__}({self.description})"

    def plot_spectrogram(self, *, vmax=None):
        fig = self.dataset.FilteredCount.plot(
            x="Rough_wavelength",
            y="Time",
            col="ch",
            sharex=False,
            vmax=vmax,
            robust=True,
        )
        plt.suptitle(self)
        return fig

    def plot_spectrum(self, time):
        plot = self.dataset.sel(Time=time, method="nearest").FilteredCount.plot.line(
            x="Rough_wavelength", col="ch", sharex=False
        )
        plt.suptitle(f"{self}\nSavgol params: {self.savgol_settings}")
        plt.tight_layout()
        return plot

    def plot_global_timetrace(self):
        # for ch in [1, 2]:
        ds = self.dataset
        arr = ds.Count.sum(dim="pixel")
        arr.name = r"$\sum_{pixel} count(time, pixel, ch)$"
        plot = arr.plot(x="Time", col="ch", sharey=False)
        return plot

    @cached_property
    def dataset(self):
        return parse_everything(self.path, self.savgol_settings)

    @property
    def WL_resolution(self):
        return self.dataset.Rough_wavelength.diff(dim="pixel").max(dim="pixel")


def list_from_config(field):
    return field.replace("'", "").split(", ")


@memory.cache  # pragma: no cover
def parse_config(path):
    parsedlines = []
    with path.open() as f:
        line = f.readline()
        while line.startswith("#"):
            parsedlines.append(line[2:])
            line = f.readline()

    configstr = "\n".join(parsedlines)
    config = configparser.ConfigParser()
    config.read_string(configstr)
    return config


@memory.cache  # pragma: no cover
def parse_dataframe(path, config):
    df = pd.read_table(path, comment="#", header=None, sep=r",\s+", engine="python")
    columns = list(
        itertools.chain.from_iterable(
            [
                list_from_config(config["Parameters"][category_name])
                for category_name in ["DimName", "ValName"]
            ]
        )
    )
    df.columns = columns
    return df


@memory.cache  # pragma: no cover
def parse_dataset(dataframe, config, savgol_settings):
    ds = dataframe.to_xarray()
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
    # verify that wavelengths are flat (constant with time)
    WLs = ds.Rough_wavelength
    unique_WLs = np.unique(WLs).size == (WLs.ch.size * WLs.pixel.size)
    assert unique_WLs
    ds["Rough_wavelength"] = ds["Rough_wavelength"].isel(Time=0)
    ds = ds.set_coords("Rough_wavelength")

    ds["Count"] = ds.Count.where(ds.Count.median("Time") != 0, np.nan)
    ds["FilteredCount"] = xarray.apply_ufunc(
        signal.savgol_filter,
        ds.Count,
        kwargs=savgol_settings,
        input_core_dims=[["pixel"]],
        output_core_dims=[["pixel"]],
    )
    ds["FilteredCount"].attrs.update(savgol_settings)
    ds["Time"].attrs["units"] = "s"
    ds["Rough_wavelength"].attrs["units"] = "nm"
    a = ds.attrs
    for key, value in config["Comments"].items():
        a[key] = value
    for key in ["Name", "ShotNo", "Date"]:
        a[key] = config["Parameters"][key]
    return ds


def parse_everything(path, savgol_settings):
    config = parse_config(path)
    dataframe = parse_dataframe(path, config)
    dataset = parse_dataset(dataframe, config, savgol_settings)
    return dataset
