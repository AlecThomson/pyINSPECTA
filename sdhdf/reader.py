"""Reading of SDHDF bands into xarray Datasets with SDHDF 4.0 names."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import astropy.units as u
import dask.array as da
import h5py
import numpy as np
import pandas as pd
import xarray as xr
from dask.array.core import normalize_chunks

from sdhdf.attributes import decode_attrs, get_str_list
from sdhdf.constants import DATA_DIMS
from sdhdf.definition import definition
from sdhdf.logger import logger
from sdhdf.schema import canonical_dim, find_child, find_group, polarisation_labels
from sdhdf.tables import SDHDFTable


class LazyH5Array:
    """A picklable HDF5 dataset that opens the file only when indexed."""

    def __init__(
        self, filename: str | Path, path: str, shape: tuple[int, ...], dtype: Any
    ) -> None:
        self.filename = str(filename)
        self.path = path
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.ndim = len(self.shape)

    def __getitem__(self, key: Any) -> np.ndarray:
        with h5py.File(self.filename, "r") as h5:
            values: np.ndarray = h5[self.path][key]
            return values

    def __dask_tokenize__(self) -> tuple[str, str, str, int | None]:
        try:
            mtime: int | None = Path(self.filename).stat().st_mtime_ns
        except OSError:
            mtime = None
        return (type(self).__name__, self.filename, self.path, mtime)


def read_lazy_array(dataset: h5py.Dataset, filename: Path) -> da.Array | np.ndarray:
    """A dask array reading ``dataset`` on demand, in chunks along the first axis."""
    if dataset.size == 0:
        empty: np.ndarray = dataset[()]
        return empty
    chunks = normalize_chunks(
        ("auto",) + (-1,) * (dataset.ndim - 1),
        shape=dataset.shape,
        dtype=dataset.dtype,
        limit="128MiB",
    )
    return da.from_array(
        LazyH5Array(filename, dataset.name, dataset.shape, dataset.dtype),
        chunks=chunks,
        asarray=True,
        meta=np.empty((0,) * dataset.ndim, dtype=dataset.dtype),
    )


def _valid_labels(attrs: dict[str, Any], ndim: int) -> list[str] | None:
    labels = get_str_list(attrs, "DIMENSION_LABELS")
    if labels is None or len(labels) != ndim:
        return None
    labels = [canonical_dim(label) for label in labels]
    if len(set(labels)) != ndim or not set(labels) <= {*DATA_DIMS, "beam"}:
        return None
    return labels


def _array_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    """Attributes to keep on an array, dropping layout bookkeeping."""
    return {
        k: v for k, v in attrs.items() if k not in ("DIMENSION_LABELS", "CLASS", "NAME")
    }


def infer_data_dims(
    shape: tuple[int, ...], attrs: dict[str, Any], name: str = "data"
) -> list[str]:
    """Dimension names of a data-like array.

    Uses ``DIMENSION_LABELS`` when intact, otherwise the SDHDF ordering
    ``(time, [beam,] polarisation, frequency, bin)``.
    """
    labels = _valid_labels(attrs, len(shape))
    if labels is not None:
        return labels
    defaults = {
        5: ["time", "beam", "polarisation", "frequency", "bin"],
        4: list(DATA_DIMS),
        3: ["time", "polarisation", "frequency"],
        2: ["time", "frequency"],
        1: ["frequency"],
    }
    if len(shape) not in defaults:
        msg = f"Cannot interpret dimensions of '{name}' with shape {shape}"
        raise ValueError(msg)
    logger.debug(
        f"No usable DIMENSION_LABELS for '{name}'; assuming {defaults[len(shape)]}"
    )
    return defaults[len(shape)]


def infer_aux_dims(
    shape: tuple[int, ...],
    attrs: dict[str, Any],
    data_sizes: dict[str, int],
    name: str,
) -> list[str] | None:
    """Dimension names of a flags or weights array, or None if it can't match the data.

    Flags and weights may have any subset of the data dimensions, and size-1
    dimensions apply to all of that axis (e.g. one flag for all polarisations).
    """
    candidates: list[list[str]] = []
    labels = _valid_labels(attrs, len(shape))
    if labels is not None:
        candidates.append(labels)
    data_dims = list(data_sizes)
    if len(shape) == len(data_dims):
        candidates.append(data_dims)
    candidates += {
        5: [["time", "beam", "polarisation", "frequency", "bin"]],
        4: [list(DATA_DIMS)],
        3: [["time", "polarisation", "frequency"]],
        2: [["time", "frequency"]],
        1: [["frequency"], ["time"]],
    }.get(len(shape), [])
    for dims in candidates:
        if all(
            dim in ("beam", *data_sizes) and size in (1, data_sizes.get(dim, 1))
            for dim, size in zip(dims, shape, strict=True)
        ):
            return dims
    logger.warning(
        f"Could not match '{name}' with shape {shape} to data dimensions "
        f"{data_sizes}; ignoring it."
    )
    return None


def _squeeze_beam(arr: xr.DataArray) -> xr.DataArray:
    if "beam" not in arr.dims:
        return arr
    if arr.sizes["beam"] != 1:
        logger.warning(
            f"'{arr.name}' has {arr.sizes['beam']} beams in one beam group; "
            "using the first."
        )
    return arr.isel(beam=0, drop=True)


def read_flags_or_weights(
    dataset: h5py.Dataset | None,
    data_sizes: dict[str, int],
    filename: Path,
    name: str,
) -> xr.DataArray | None:
    """Read flags or weights, dropping size-1 axes that apply to the whole axis."""
    if not isinstance(dataset, h5py.Dataset):
        return None
    attrs = decode_attrs(dataset.attrs)
    dims = infer_aux_dims(dataset.shape, attrs, data_sizes, name)
    if dims is None:
        return None
    arr = _squeeze_beam(
        xr.DataArray(read_lazy_array(dataset, filename), dims=dims, name=name)
    )
    # time and frequency stay as dimensions even when the data have one sample
    squeeze = [
        d
        for d in arr.dims
        if arr.sizes[d] == 1
        and (d not in ("time", "frequency") or data_sizes.get(str(d), 1) != 1)
    ]
    if squeeze:
        arr = arr.isel(dict.fromkeys(squeeze, 0), drop=True)
    arr.attrs = _array_attrs(attrs)
    return arr


def read_frequency(
    dataset: h5py.Dataset | None, n_time: int, n_chan: int, name: str
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    """Read a frequency axis.

    Returns:
        tuple: Frequency per channel, frequency per integration and channel
            (None if constant in time), and the decoded attributes.
    """
    if not isinstance(dataset, h5py.Dataset):
        logger.warning(f"No frequency axis for '{name}'; using channel numbers.")
        return np.arange(n_chan, dtype=float), None, {"UNIT": None}
    attrs = decode_attrs(dataset.attrs)
    freq = np.asarray(dataset[()], dtype=np.float64)
    if freq.ndim == 1:
        return freq, None, attrs
    freq = freq.reshape(-1, freq.shape[-1])
    if freq.shape[0] == 1 or np.all(freq == freq[0]):
        return freq[0], None, attrs
    if freq.shape[0] != n_time:
        logger.warning(
            f"Frequency axis of '{name}' has {freq.shape[0]} rows for {n_time} "
            "integrations; using the first row."
        )
        return freq[0], None, attrs
    return freq[0], freq, attrs


def channel_width_hz(freq: np.ndarray, unit: str) -> np.ndarray:
    """Width of each channel in Hz, from the spacing of ``freq`` in ``unit``."""
    if freq.size < 2:
        return np.ones_like(freq)
    width_hz: np.ndarray = (np.abs(np.gradient(freq)) * u.Unit(unit)).to_value(u.Hz)
    return width_hz


def time_coordinates(
    table: SDHDFTable | None, n_time: int, name: str
) -> dict[str, tuple[Any, ...]]:
    """Per-integration coordinates from an observation_parameters table.

    ``time`` is the MJD at the integration centre. Columns keep their units
    from the file in a ``units`` attribute.
    """
    coords: dict[str, tuple[Any, ...]] = {}
    if table is None or len(table) != n_time:
        if table is not None:
            logger.warning(
                f"'{name}' has {len(table)} rows for {n_time} integrations; "
                "not attaching per-integration metadata."
            )
        coords["time"] = ("time", np.arange(n_time, dtype=float))
        return coords
    frame = table.table
    for col in frame.columns:
        values = frame[col].to_numpy()
        if values.ndim == 1:
            attrs = {"units": table.units[col]} if col in table.units else {}
            coords[str(col)] = ("time", values, attrs)
    if "MJD" in frame.columns and "FRACTIONAL_MJD" in frame.columns:
        mjd = frame["MJD"].to_numpy(dtype=float) + frame["FRACTIONAL_MJD"].to_numpy(
            dtype=float
        )
        coords["time"] = ("time", mjd, {"units": "d"})
    else:
        coords["time"] = ("time", np.arange(n_time, dtype=float))
    return coords


def _integration_time_s(
    coords: dict[str, tuple[Any, ...]], requested_s: float, n_time: int
) -> np.ndarray:
    if "INTEGRATION_TIME" in coords:
        _, values, *rest = coords["INTEGRATION_TIME"]
        unit = rest[0].get("units", "s") if rest else "s"
        integration_time_s: np.ndarray = (
            np.asarray(values, dtype=float) * u.Unit(unit)
        ).to_value(u.s)
        if np.all(np.isfinite(integration_time_s)) and np.all(integration_time_s > 0):
            return integration_time_s
    return np.full(n_time, requested_s if requested_s > 0 else 1.0)


def build_dataset(
    variables: dict[str, h5py.Dataset],
    freq_dataset: h5py.Dataset | None,
    flags_dataset: h5py.Dataset | None,
    weights_dataset: h5py.Dataset | None,
    table: SDHDFTable | None,
    band_row: dict[str, Any],
    filename: Path,
    name: str,
) -> xr.Dataset:
    """Build a Dataset from data-like HDF5 datasets that share time and frequency axes."""
    arrays: dict[str, xr.DataArray] = {}
    for var, dset in variables.items():
        attrs = decode_attrs(dset.attrs)
        dims = infer_data_dims(dset.shape, attrs, f"{name}/{var}")
        arr = _squeeze_beam(
            xr.DataArray(read_lazy_array(dset, filename), dims=dims, name=var)
        )
        missing = [d for d in DATA_DIMS if d not in arr.dims]
        if missing:
            arr = arr.expand_dims(missing)
        arr = arr.transpose(*DATA_DIMS)
        if var == "binned":
            # The noise diode phase bins differ in number from the on/off bins
            arr = arr.rename(bin="cal_bin")
        arr.attrs = _array_attrs(attrs)
        arr.attrs["units"] = arr.attrs.get("UNIT") or "counts"
        arrays[var] = arr

    first = next(iter(arrays.values()))
    n_time, n_pol, n_chan = (
        first.sizes[d] for d in ("time", "polarisation", "frequency")
    )
    for var, arr in arrays.items():
        if any(
            arr.sizes[d] != first.sizes[d]
            for d in ("time", "polarisation", "frequency")
        ):
            msg = f"'{name}/{var}' has shape {dict(arr.sizes)}, inconsistent with {dict(first.sizes)}"
            raise ValueError(msg)

    freq, freq_2d, freq_attrs = read_frequency(freq_dataset, n_time, n_chan, name)
    if freq.size != n_chan:
        logger.warning(
            f"Frequency axis of '{name}' has {freq.size} channels, data has "
            f"{n_chan}; using channel numbers."
        )
        freq, freq_2d = np.arange(n_chan, dtype=float), None
    # MHz is the SDHDF default when a file doesn't record the unit
    freq_unit = freq_attrs.get("UNIT") or "MHz"

    coords: dict[str, Any] = time_coordinates(table, n_time, name)
    coords["frequency"] = (
        "frequency",
        freq,
        {"units": freq_unit, "FRAME": freq_attrs.get("FRAME") or "topocentric"},
    )
    if freq_2d is not None:
        coords["frequency_per_time"] = (
            ("time", "frequency"),
            freq_2d,
            {"units": freq_unit},
        )
    coords["polarisation"] = (
        "polarisation",
        polarisation_labels(band_row.get("POLARISATION_TYPE"), n_pol),
    )

    sizes = {d: first.sizes[d] for d in DATA_DIMS}
    flags = read_flags_or_weights(flags_dataset, sizes, filename, f"{name}/flags")
    if flags is None:
        flags = xr.DataArray(
            da.zeros((n_time, n_chan), dtype=bool), dims=("time", "frequency")
        )
    else:
        flags = flags != 0
    flags.attrs = {"description": "True where flagged"}

    weights = read_flags_or_weights(weights_dataset, sizes, filename, f"{name}/weights")
    if weights is None:
        integration_time_s = _integration_time_s(
            coords, float(band_row.get("REQUESTED_INTEGRATION_TIME", 1.0)), n_time
        )
        weights = xr.DataArray(
            da.outer(
                da.from_array(integration_time_s),
                da.from_array(channel_width_hz(freq, freq_unit)),
            ).astype(np.float32),
            dims=("time", "frequency"),
            attrs={
                "WEIGHT_TYPE": "Product of the channel bandwidth (Hz) and integration time (s)"
            },
        )

    ds = xr.Dataset({**arrays, "flags": flags, "weights": weights}, coords=coords)
    ds["time"].attrs = {"units": "d", "description": "MJD (UTC) at integration centre"}
    return ds


def _read_table(
    group: h5py.Group | None, kind: str, tables: dict[str, tuple[str, ...]]
) -> SDHDFTable | None:
    dset = find_child(group, tables[kind])
    if not isinstance(dset, h5py.Dataset):
        return None
    return SDHDFTable.from_dataset(dset, kind)


def read_band(
    h5: h5py.File,
    beam_label: str,
    band_label: str,
    band_row: dict[str, Any] | None = None,
    calibrator_band_row: dict[str, Any] | None = None,
) -> tuple[xr.Dataset, xr.Dataset | None]:
    """Read one band of an open SDHDF file.

    Args:
        h5 (h5py.File): Open SDHDF file.
        beam_label (str): Beam group name (e.g. ``"beam_0"``).
        band_label (str): Band group name (e.g. ``"band_SB0"``).
        band_row (dict, optional): This band's row of ``band_parameters``.
        calibrator_band_row (dict, optional): This band's row of
            ``calibrator_band_parameters``.

    Returns:
        tuple: The astronomy Dataset, and the calibrator Dataset (None if the
            file has no noise diode data).
    """
    filename = Path(os.fsdecode(h5.filename))
    group = h5[beam_label][band_label]
    name = f"{beam_label}/{band_label}"
    band_row = band_row or {}
    meta = find_group(group, "metadata")
    band_tables = definition().band_tables
    astronomy_datasets = definition().astronomy_datasets
    calibrator_datasets = definition().calibrator_datasets

    astro = find_group(group, "astronomy_data")
    data = find_child(astro, astronomy_datasets["data"])
    if not isinstance(data, h5py.Dataset):
        msg = f"No astronomy data found in '{name}'"
        raise KeyError(msg)
    astronomy = build_dataset(
        variables={"data": data},
        freq_dataset=find_child(astro, astronomy_datasets["frequency"]),
        flags_dataset=find_child(astro, astronomy_datasets["flags"]),
        weights_dataset=find_child(astro, astronomy_datasets["weights"]),
        table=_read_table(meta, "observation_parameters", band_tables),
        band_row=band_row,
        filename=filename,
        name=name,
    )
    astronomy.attrs = {"beam": beam_label, "band": band_label}

    cal = find_group(group, "calibrator_data")
    if cal is None:
        return astronomy, None
    variables = {
        role: dset
        for role in ("on", "off", "binned", "data")
        if isinstance(dset := find_child(cal, calibrator_datasets[role]), h5py.Dataset)
        and dset.size > 0
    }
    if not variables:
        return astronomy, None
    try:
        calibrator = build_dataset(
            variables=variables,
            freq_dataset=find_child(cal, calibrator_datasets["frequency"]),
            flags_dataset=find_child(cal, calibrator_datasets["flags"]),
            weights_dataset=find_child(cal, calibrator_datasets["weights"]),
            table=_read_table(meta, "calibrator_observation_parameters", band_tables),
            band_row=calibrator_band_row or band_row,
            filename=filename,
            name=f"{name}/calibrator_data",
        )
    except ValueError as e:
        logger.warning(f"Could not read calibrator data for '{name}': {e}")
        return astronomy, None
    calibrator.attrs = {"beam": beam_label, "band": band_label}
    return astronomy, calibrator


def table_row(table: SDHDFTable | None, label: str) -> dict[str, Any]:
    """The row of a band_parameters table for a band label."""
    if table is None or len(table) == 0:
        return {}
    frame: pd.DataFrame = table.table
    if "LABEL" in frame.columns:
        match = frame[frame["LABEL"] == label]
        if len(match):
            return {str(k): v for k, v in match.iloc[0].to_dict().items()}
    return {}
