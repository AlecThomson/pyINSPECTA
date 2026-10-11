"""Weighted averaging of SDHDF bands in time and frequency.

Flags are treated as in CASA's ``mstransform``: flagged samples are left out
of the weighted mean ``sum(w (1 - f) x) / sum(w (1 - f))`` unless every sample
in a block is flagged. The output is then the weighted mean of all of them,
with weight ``sum(w)``, and is flagged. Samples left over when the axis length
is not a multiple of the block size are dropped.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np
import xarray as xr
from astropy.units import Quantity

from sdhdf.logger import logger

if TYPE_CHECKING:
    from xarray.computation.rolling import DataArrayCoarsen


def samples_per_width(ds: xr.Dataset, dim: str, width: Quantity) -> int:
    """Number of samples along ``dim`` spanning ``width``.

    The sample spacing is taken in the units recorded in the file: the
    frequency axis for ``frequency``, and ``INTEGRATION_TIME`` (or the MJD
    spacing) for ``time``.
    """
    if dim == "time" and "INTEGRATION_TIME" in ds.coords:
        integration_time = ds["INTEGRATION_TIME"]
        step = float(np.median(integration_time)) * u.Unit(
            integration_time.attrs.get("units", "s")
        )
    else:
        values = np.asarray(ds[dim], dtype=float)
        if values.size < 2:
            return 1
        step = float(np.median(np.abs(np.diff(values)))) * u.Unit(
            ds[dim].attrs.get("units", "MHz" if dim == "frequency" else "d")
        )
    n_samples = max(1, round(float((width / step).decompose())))
    logger.info(f"Averaging {n_samples} samples along '{dim}' for a width of {width}")
    return n_samples


def average_coordinate(coord: xr.DataArray, dim: str, n: int) -> xr.DataArray:
    """Average one coordinate along ``dim``."""
    coord = xr.DataArray(coord.variable, name=coord.name)  # without other coords
    if coord.name == "INTEGRATION_TIME" and dim == "time":  # summed, not averaged
        return coord.coarsen({dim: n}, boundary="trim").sum()
    if np.issubdtype(coord.dtype, np.number) and not np.issubdtype(
        coord.dtype, np.integer
    ):
        return coord.coarsen({dim: n}, boundary="trim").mean()
    # Integers, strings and objects: take the middle sample of each block
    n_out = coord.sizes[dim] // n
    idx = np.arange(n_out) * n + n // 2
    return coord.isel({dim: idx})


def average_coordinates(
    ds: xr.Dataset, out: xr.Dataset, dim: str, n: int
) -> xr.Dataset:
    """Assign the block-averaged coordinates along ``dim`` of ``ds`` to ``out``."""
    coords = {}
    for name, coord in ds.coords.items():
        if dim not in coord.dims:
            continue
        new = average_coordinate(coord, dim, n)
        coords[name] = (coord.dims, new.data, coord.attrs)
    out = out.assign_coords(coords)
    if dim == "time" and {"MJD", "FRACTIONAL_MJD"} <= set(out.coords):
        mjd = np.asarray(out["time"], dtype=float)
        out = out.assign_coords(
            MJD=("time", np.floor(mjd).astype(np.int64), out["MJD"].attrs),
            FRACTIONAL_MJD=("time", mjd - np.floor(mjd), out["FRACTIONAL_MJD"].attrs),
        )
    return out


def average_dimension(ds: xr.Dataset, dim: str, n: int, method: str) -> xr.Dataset:
    """Weighted average of ``n`` consecutive samples along ``dim``.

    See the module docstring for the treatment of flags and weights.
    """
    if n <= 1:
        return ds
    if ds.sizes[dim] < n:
        msg = f"Cannot average {n} samples along '{dim}' of length {ds.sizes[dim]}"
        raise ValueError(msg)

    flags = ds["flags"]
    weights = (ds["weights"] * xr.ones_like(flags, dtype=np.float64)).astype(np.float64)
    if dim not in weights.dims:
        weights = weights.expand_dims({dim: ds.sizes[dim]}, axis=0)
        flags = flags.expand_dims({dim: ds.sizes[dim]}, axis=0)
    flags = flags.broadcast_like(weights)
    unflagged_weights = weights.where(~flags, 0)

    def blocks(arr: xr.DataArray) -> DataArrayCoarsen:
        arr = arr.drop_vars([c for c in arr.coords if c not in arr.dims])
        return arr.coarsen({dim: n}, boundary="trim")

    all_flagged = blocks(flags).all()
    w_good = blocks(unflagged_weights).sum()
    w_all = blocks(weights).sum()
    new_vars = {}
    for var in ("data", "on", "off", "binned"):  # flags and weights handled below
        if var not in ds:
            continue
        x = ds[var]
        if method == "median":
            good = blocks(x.where(~flags)).median()
            every = blocks(x).median()
        else:
            good = blocks(
                (x * unflagged_weights).astype(np.float64)
            ).sum() / w_good.where(w_good > 0)
            every = blocks((x * weights).astype(np.float64)).sum() / w_all.where(
                w_all > 0
            )
        value = every.where(all_flagged, good)
        # Zero weights: fall back to the unweighted mean of the same samples
        value = value.fillna(
            blocks(x).mean().where(all_flagged, blocks(x.where(~flags)).mean())
        )
        new_vars[var] = value.astype(x.dtype).transpose(*x.dims).assign_attrs(x.attrs)

    out = xr.Dataset(
        {
            **new_vars,
            "flags": all_flagged.assign_attrs(ds["flags"].attrs),
            "weights": w_all.where(all_flagged, w_good)
            .astype(np.float32)
            .assign_attrs(ds["weights"].attrs),
        },
        attrs=ds.attrs,
    )
    out = out.drop_vars([c for c in out.coords if c not in out.dims], errors="ignore")
    keep = {c: ds.coords[c] for c in ds.coords if dim not in ds.coords[c].dims}
    out = out.assign_coords(keep)
    return average_coordinates(ds, out, dim, n)


def average(
    ds: xr.Dataset,
    time: int | Quantity | None = None,
    frequency: int | Quantity | None = None,
    method: str = "mean",
) -> xr.Dataset:
    """Average a band dataset in time and/or frequency.

    Args:
        ds (xr.Dataset): Band dataset (astronomy or calibrator).
        time (int | Quantity, optional): Number of integrations to average, or
            the integration time to average to (e.g. ``10 * u.s``).
        frequency (int | Quantity, optional): Number of channels to average,
            or the channel width to average to (e.g. ``0.5 * u.MHz``).
        method (str, optional): ``"mean"`` (weighted) or ``"median"`` (of
            unflagged samples). Defaults to ``"mean"``.

    Returns:
        xr.Dataset: The averaged dataset.

    Raises:
        TypeError: If ``time`` or ``frequency`` is a plain float, which would
            be ambiguous between a count and a width.
    """
    if method not in ("mean", "median"):
        msg = f"Unknown averaging method '{method}'"
        raise ValueError(msg)
    for dim, value in (("frequency", frequency), ("time", time)):
        if value is None:
            continue
        if isinstance(value, Quantity):
            n = samples_per_width(ds, dim, value)
        elif isinstance(value, int | np.integer):
            n = int(value)
        else:
            msg = (
                f"'{dim}' must be a number of samples (int) or a width with units "
                f"(e.g. astropy Quantity), got {value!r}"
            )
            raise TypeError(msg)
        if dim == "frequency" and "frequency_per_time" in ds.coords:
            ds = ds.drop_vars("frequency_per_time")
            logger.warning(
                "Dropping time-dependent frequencies when averaging channels."
            )
        ds = average_dimension(ds, dim, n, method)
    return ds
