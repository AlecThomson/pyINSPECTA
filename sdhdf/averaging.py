"""Weighted averaging of SDHDF bands in time and frequency.

Flags are treated as in CASA's ``mstransform``. The weighted mean of ``n``
samples is::

    x = sum(w_i (1 - f_i) x_i) / sum(w_i (1 - f_i))

where ``w`` are the weights and ``f`` the flags, and the output weight is
``sum(w_i (1 - f_i))``. Flagged samples are excluded unless *all* samples
contributing to an output sample are flagged; the output is then the weighted
mean of all of them, ``sum(w_i x_i) / sum(w_i)``, with weight ``sum(w_i)``,
and is flagged. Samples left over when the axis length is not a multiple of
``n`` are dropped.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from sdhdf.logger import logger


def factor_for_width(ds: xr.Dataset, dim: str, width: float) -> int:
    """Number of samples along ``dim`` to average for a given width.

    For ``time`` the width is in seconds (using ``INTEGRATION_TIME`` where
    recorded); otherwise it is in the units of the coordinate.
    """
    values = np.asarray(ds[dim], dtype=float)
    if values.size < 2:
        return 1
    step = float(np.median(np.abs(np.diff(values))))
    if dim == "time" and "INTEGRATION_TIME" in ds.coords:
        step = float(np.median(ds["INTEGRATION_TIME"]))  # seconds
    factor = max(1, round(width / step))
    logger.info(f"Averaging {factor} samples along '{dim}' for a width of {width}")
    return factor


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

    def blocks(arr: xr.DataArray) -> xr.DataArray:
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
        value = xr.where(all_flagged, every, good)
        # Zero weights: fall back to the unweighted mean of the same samples
        value = value.fillna(
            xr.where(all_flagged, blocks(x).mean(), blocks(x.where(~flags)).mean())
        )
        new_vars[var] = value.astype(x.dtype).transpose(*x.dims).assign_attrs(x.attrs)

    out = xr.Dataset(
        {
            **new_vars,
            "flags": all_flagged.assign_attrs(ds["flags"].attrs),
            "weights": xr.where(all_flagged, w_all, w_good)
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
            factor_for_width(ds, dim, float(value))
            if isinstance(value, float)
            else int(value)
        )
        if dim == "frequency" and "frequency_per_time" in ds.coords:
            ds = ds.drop_vars("frequency_per_time")
            logger.warning(
                "Dropping time-dependent frequencies when averaging channels."
            )
        ds = average_dimension(ds, dim, n, method)
    return ds
