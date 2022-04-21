# -*- coding: utf-8 -*-
import configparser
import itertools
import os
import pathlib
from functools import cached_property

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional
import xarray
from joblib import Memory
from scipy import signal

memflag = os.environ.get("TEST_SOXMOS", False)
location = None if memflag else "./cachedir"
memory = Memory(location, verbose=0)

savgol_settings = {
    (19.5177, 34.3053): {"savgol_window_length": 13, "savgol_polyorder": 4},
    (75.7912, 101.6328): {"savgol_window_length": 14, "savgol_polyorder": 4},
    (1.3272, 3.5807): {"savgol_window_length": 13, "savgol_polyorder": 4},
    (2.6977, 5.4807): {"savgol_window_length": 13, "savgol_polyorder": 4},
    (25.1769, 32.0582): {"savgol_window_length": 13, "savgol_polyorder": 4},
}


def _add_lines(fig, lines, direction, unit="s", ls="--", alpha=0.3):
    for i, (key, value) in enumerate(lines.items()):
        for ax in fig.axes[0]:
            if direction == "vertical":
                func = ax.axvline
            elif direction == "horizontal":
                func = ax.axhline
            else:  # pragma: no cover
                raise ValueError(direction)
            func(
                value,
                c=f"C{i}",
                ls=ls,
                alpha=alpha,
                label=f"{key} at {value:.3f}{unit}",
            )

import dataclasses
@dataclasses.dataclass
class SOXMOSFile:
    dataset: xarray.Dataset
    savgol_settings: dict
    path: Optional[pathlib.Path] = None

    @classmethod
    def from_file(cls, path: pathlib.Path, savgol_settings: dict):
        config = parse_config(path)
        dataset = parse_from_path(path, config, savgol_settings)
        return cls(dataset, savgol_settings, path)

        
    @classmethod
    def from_web(cls, shotno, savgol_settings: dict):
        dataset = parse_from_web(shotno, savgol_settings)
        return cls(dataset, savgol_settings)
    
    @property
    def description(self):
        c = self.dataset.attrs
        return f"{c['Name']} #{c['ShotNo']} @{c['Date']}"

    @property
    def shotid(self):
        c = self.dataset.attrs
        return c["ShotNo"]

    def __repr__(self, *args, **kwargs):
        return f"{self.__class__.__name__}({self.description})"

    @cached_property
    def Lambda_interval(self):
        Lambda = self.dataset.Rough_wavelength
        min1, min2 = Lambda.min(dim="pixel").data
        max1, max2 = Lambda.max(dim="pixel").data
        return {1: (min1, max1), 2: (min2, max2)}

    def plot_spectrogram(self, *, vmax=None, **special_lines):
        fig = self.dataset.FilteredCount.plot(
            x="Rough_wavelength",
            y="Time",
            col="ch",
            sharex=False,
            vmax=vmax,
            robust=True,
        )
        _add_lines(fig, special_lines, "horizontal")
        plt.suptitle(self)
        return fig

    def plot_spectrum(self, time, **special_lines):
        plot = self.dataset.sel(Time=time, method="nearest").FilteredCount.plot.line(
            x="Rough_wavelength", col="ch", sharex=False
        )
        plt.suptitle(f"{self}\nSavgol params: {self.savgol_settings}")
        plt.tight_layout()
        _add_lines(plot, special_lines, "vertical")
        return plot

    def plot_global_timetrace(self, **special_lines):
        # for ch in [1, 2]:
        ds = self.dataset
        arr = ds.Count.sum(dim="pixel")
        arr.name = r"$\sum_{pixel} count(time, pixel, ch)$"
        plot = arr.plot(x="Time", col="ch", sharey=False)
        _add_lines(plot, special_lines, "vertical")
        return plot


    @property
    def WL_resolution(self):
        return self.dataset.Rough_wavelength.diff(dim="pixel").max(dim="pixel")


def list_from_config(field):
    return field.replace("'", "").split(", ")


# +
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
    for ds_key in ds:
        da = ds[ds_key]
        flat_dims = {dim: not (da.diff(dim).any().item()) for dim in da.dims}

        for key, value in flat_dims.items():
            if not value:
                continue
            print(f"Flattening {value} at {key}")
            ds[da.name] = da.isel({key: 0})
    parameters = config["Parameters"]
    def parse(s):
        return s.strip(",").replace("'", "").split(", ")
    
    dimname = parse(parameters["dimname"])
    dimunit = parse(parameters["dimunit"])
    dimsize = list(map(int, parse(parameters["dimsize"])))
    valname = parse(parameters["valname"])
    valunit = parse(parameters["valunit"])
    
    for name, unit, size in zip(dimname, dimunit, dimsize):
        ds[name].attrs["units"] = unit
        assert ds.sizes[name] == size
    for name, unit in zip(valname, valunit):
        ds[name].attrs["units"] = unit

    a = ds.attrs
    a.update(config["Comments"])
    for key in ["Name", "ShotNo", "Date"]:
        a[key] = config["Parameters"][key]
        
    #####
    ds = ds.set_coords("Rough_wavelength")

    ds["Count"] = ds.Count.where(ds.Count.median("Time") != 0, np.nan)

        
    #####
    def filter_helper(group):
        Lambda = group.Rough_wavelength
        interval = (Lambda.min().item(), Lambda.max().item())
        local_savgol_settings = savgol_settings[interval]

        return xarray.apply_ufunc(
            signal.savgol_filter,
            group.Count,
            kwargs=local_savgol_settings,
            input_core_dims=[["pixel"]],
            output_core_dims=[["pixel"]],
        )

    ds["FilteredCount"] = ds.groupby("ch").apply(filter_helper)
    ds["FilteredCount"].attrs.update(savgol_settings)
    ds["FilteredCount"].attrs["units"] = ds["Count"].attrs["units"]
    return ds



@memory.cache  # pragma: no cover
def parse_from_path(path, config, savgol_settings):
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
    return parse_dataset(df, config, savgol_settings)


@memory.cache
def parse_from_web(shotno, savgol_settings):
    import requests
    import io
    diagnostic = "soxmos"
    url = f"https://exp.lhd.nifs.ac.jp/opendata/LHD/webapi.fcgi?cmd=getfile&diag={diagnostic}&shotno={shotno}&subno=1"
    result = requests.get(url)
    parsedlines = []
    for line in result.content.splitlines():
        if line.startswith(b"#"):
            parsedlines.append(line[2:].decode())

    configstr = "\n".join(parsedlines)
    config = configparser.ConfigParser()
    config.read_string(configstr)
    return parse_from_path.func(io.BytesIO(result.content), config, savgol_settings)
