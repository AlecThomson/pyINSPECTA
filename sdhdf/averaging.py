"""Weighted averaging of SDHDF bands in time and frequency.

The weighted mean of ``n`` samples is::

    x = sum(w_i (1 - f_i) x_i) / sum(w_i (1 - f_i))

where ``w`` are the weights and ``f`` the flags. The output weight is
``sum(w_i (1 - f_i))``, and an output sample is flagged only if all of its
input samples were flagged (it then holds the unweighted mean of the inputs,
so that it remains inspectable). Samples left over when the axis length is not
a multiple of ``n`` are dropped.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from sdhdf.logger import logger

#: Per-integration metadata that should be summed (not averaged) in time.
_SUMMED_TIME_COORDS = {"INTEGRATION_TIME"}
#: Variables averaged like data (all others are flags/weights).
_DATA_VARS = ("data", "on", "off", "binned")


def _factor_for_width(ds: xr.Dataset, dim: str, width: float) -> int:
    values = np.asarray(ds[dim], dtype=float)
    if values.size < 2:
        return 1
    step = float(np.median(np.abs(np.diff(values))))
    if dim == "time" and "INTEGRATION_TIME" in ds.coords:
        step = float(np.median(ds["INTEGRATION_TIME"]))  # seconds
    factor = max(1, round(width / step))
    logger.info(f"Averaging {factor} samples along '{dim}' for a width of {width}")
    return factor


def _coarsen_coord(coord: xr.DataArray, dim: str, n: int) -> xr.DataArray:
    """Average one coordinate along ``dim``."""
    coord = xr.DataArray(coord.variable, name=coord.name)  # without other coords
    if coord.name in _SUMMED_TIME_COORDS and dim == "time":
        return coord.coarsen({dim: n}, boundary="trim").sum()
    if np.issubdtype(coord.dtype, np.number) and not np.issubdtype(
        coord.dtype, np.integer
    ):
        return coord.coarsen({dim: n}, boundary="trim").mean()
    # Integers, strings and objects: take the middle sample of each block
    n_out = coord.sizes[dim] // n
    idx = np.arange(n_out) * n + n // 2
    return coord.isel({dim: idx})


def _rebuild_coords(ds: xr.Dataset, out: xr.Dataset, dim: str, n: int) -> xr.Dataset:
    coords = {}
    for name, coord in ds.coords.items():
        if dim not in coord.dims:
            continue
        new = _coarsen_coord(coord, dim, n)
        coords[name] = (coord.dims, new.data, coord.attrs)
    out = out.assign_coords(coords)
    if dim == "time" and {"MJD", "FRACTIONAL_MJD"} <= set(out.coords):
        mjd = np.asarray(out["time"], dtype=float)
        out = out.assign_coords(
            MJD=("time", np.floor(mjd).astype(np.int64), out["MJD"].attrs),
            FRACTIONAL_MJD=("time", mjd - np.floor(mjd), out["FRACTIONAL_MJD"].attrs),
        )
    return out


def _average_dim(ds: xr.Dataset, dim: str, n: int, method: str) -> xr.Dataset:
    if n <= 1:
        return ds
    if ds.sizes[dim] < n:
        msg = f"Cannot average {n} samples along '{dim}' of length {ds.sizes[dim]}"
        raise ValueError(msg)

    flags = ds["flags"]
    w_eff = ds["weights"].where(~flags, 0).astype(np.float64)
    if dim not in w_eff.dims:
        w_eff = w_eff.expand_dims({dim: ds.sizes[dim]}, axis=0)

    def blocks(arr: xr.DataArray) -> xr.DataArray:
        arr = arr.drop_vars([c for c in arr.coords if c not in arr.dims])
        return arr.coarsen({dim: n}, boundary="trim")

    w_sum = blocks(w_eff).sum()
    new_vars = {}
    for var in _DATA_VARS:
        if var not in ds:
            continue
        x = ds[var]
        if method == "median":
            good = blocks(x.where(w_eff > 0)).median()
        else:
            num = blocks((x * w_eff).astype(np.float64)).sum()
            good = num / w_sum.where(w_sum > 0)
        fallback = blocks(x).mean()
        new_vars[var] = (
            good.fillna(fallback)
            .astype(x.dtype)
            .transpose(*x.dims)
            .assign_attrs(x.attrs)
        )

    out = xr.Dataset(
        {
            **new_vars,
            "flags": (w_sum <= 0).assign_attrs(flags.attrs),
            "weights": w_sum.astype(np.float32).assign_attrs(ds["weights"].attrs),
        },
        attrs=ds.attrs,
    )
    out = out.drop_vars([c for c in out.coords if c not in out.dims], errors="ignore")
    keep = {c: ds.coords[c] for c in ds.coords if dim not in ds.coords[c].dims}
    out = out.assign_coords(keep)
    return _rebuild_coords(ds, out, dim, n)


def average(
    ds: xr.Dataset,
    time: int | float | None = None,
    frequency: int | float | None = None,
    method: str = "mean",
) -> xr.Dataset:
    """Average a band dataset in time and/or frequency.

    Args:
        ds (xr.Dataset): Band dataset (astronomy or calibrator).
        time (int | float, optional): If int, the number of integrations to
            average. If float, the desired integration time in seconds.
        frequency (int | float, optional): If int, the number of channels to
            average. If float, the desired channel width in the units of the
            frequency axis (usually MHz).
        method (str, optional): ``"mean"`` (weighted) or ``"median"`` (of
            unflagged samples). Defaults to ``"mean"``.

    Returns:
        xr.Dataset: The averaged dataset.
    """
    if method not in ("mean", "median"):
        msg = f"Unknown averaging method '{method}'"
        raise ValueError(msg)
    for dim, value in (("frequency", frequency), ("time", time)):
        if value is None:
            continue
        n = (
            _factor_for_width(ds, dim, float(value))
            if isinstance(value, float)
            else int(value)
        )
        if dim == "frequency" and "frequency_per_time" in ds.coords:
            ds = ds.drop_vars("frequency_per_time")
            logger.warning(
                "Dropping time-dependent frequencies when averaging channels."
            )
        ds = _average_dim(ds, dim, n, method)
    return ds
