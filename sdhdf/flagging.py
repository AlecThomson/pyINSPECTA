"""SDHDF flagging utilities

All functions take a band Dataset (see :mod:`sdhdf.reader`) and return a new
Dataset whose ``flags`` are the logical OR of the existing flags and any new
ones; existing flags are never cleared.
"""

from __future__ import annotations

import warnings
from importlib import resources

import numpy as np
import pandas as pd
import xarray as xr
from astropy.stats import mad_std, sigma_clip

from sdhdf.logger import logger
from sdhdf.schema import DATA_DIMS

RFI_COLUMNS = [
    "type",
    "observatory label",
    "receiver label",
    "freq0 MHz",
    "freq1 MHz",
    "MJD0",
    "MJD1",
    "text string for label",
]

#: Other names for observatories with bundled RFI tables.
_TELESCOPE_ALIASES = {"murriyang": "parkes", "pks": "parkes"}


class AutoFlagError(Exception):
    def __init__(self, msg):
        super().__init__(msg)
        self.msg = msg


def _telescope_key(telescope: str) -> str:
    key = str(telescope).strip().lower()
    return _TELESCOPE_ALIASES.get(key, key)


def get_persistent_rfi(telescope: str = "Parkes") -> pd.DataFrame:
    """Read the bundled persistent RFI table for a telescope.

    Args:
        telescope (str, optional): Telescope name. Defaults to "Parkes".

    Returns:
        pd.DataFrame: Persistent RFI table.
    """
    rfi_file = (
        resources.files("sdhdf.data.rfi") / f"{_telescope_key(telescope)}_rfi.csv"
    )
    if not rfi_file.is_file():
        msg = f"No persistent RFI table for telescope '{telescope}'."
        raise NotImplementedError(msg)
    with resources.as_file(rfi_file) as path:
        return pd.read_csv(path, sep=",", comment="#", names=RFI_COLUMNS)


def _or_flags(ds: xr.Dataset, new: xr.DataArray) -> xr.Dataset:
    flags = ds["flags"] | new
    flags.attrs = ds["flags"].attrs
    return ds.assign(flags=flags.transpose(*[d for d in DATA_DIMS if d in flags.dims]))


def select_rfi(
    rfi: pd.DataFrame, telescope: str | None = None, receiver: str | None = None
) -> pd.DataFrame:
    """Select the entries of an RFI table relevant to a telescope and receiver."""
    keep = np.ones(len(rfi), dtype=bool)
    if telescope is not None:
        obs = rfi["observatory label"].astype(str).str.strip().str.lower()
        keep &= obs.map(_telescope_key).to_numpy() == _telescope_key(telescope)
    if receiver is not None:
        rcvr = rfi["receiver label"].astype(str).str.strip().str.lower()
        keep &= ((rcvr == "all") | (rcvr == str(receiver).strip().lower())).to_numpy()
    return rfi[keep]


def flag_persistent_rfi(
    ds: xr.Dataset,
    rfi: pd.DataFrame,
    telescope: str | None = None,
    receiver: str | None = None,
) -> xr.Dataset:
    """Flag channels affected by persistent RFI.

    A channel is flagged in an integration if its centre frequency lies within
    an RFI entry's frequency range and the integration's MJD lies within the
    entry's MJD range.

    Args:
        ds (xr.Dataset): Band dataset.
        rfi (pd.DataFrame): RFI table (see :func:`get_persistent_rfi`).
        telescope (str, optional): Only use entries for this observatory.
        receiver (str, optional): Only use entries for this receiver (or "all").

    Returns:
        xr.Dataset: Dataset with updated flags.
    """
    rfi = select_rfi(rfi, telescope=telescope, receiver=receiver)
    freq = ds["frequency"]
    mjd = ds["time"]
    have_mjd = bool(np.all(np.asarray(mjd) > 1000))  # real MJDs, not indices
    new = xr.zeros_like(freq * mjd, dtype=bool)
    for _, row in rfi.iterrows():
        in_band = (freq >= row["freq0 MHz"]) & (freq <= row["freq1 MHz"])
        if not bool(in_band.any()):
            continue
        in_time = (mjd >= row["MJD0"]) & (mjd <= row["MJD1"]) if have_mjd else True
        new = new | (in_band & in_time)
    logger.info(f"Persistent RFI: flagged {float(new.mean()):.1%} of samples")
    return _or_flags(ds, new.transpose("time", "frequency"))


def box_filter(
    spectrum: np.ndarray, sigma: float = 3, n_windows: int = 100
) -> np.ndarray:
    """Flag outliers in a 1-D spectrum by sigma-clipping in windows.

    The spectrum is split into ``n_windows`` contiguous windows (all channels
    are covered), and each window is iteratively sigma-clipped about its median
    using the MAD-estimated standard deviation. NaNs are flagged.

    Args:
        spectrum (np.ndarray): 1-D spectrum.
        sigma (float, optional): Clipping threshold. Defaults to 3.
        n_windows (int, optional): Number of windows. Defaults to 100.

    Returns:
        np.ndarray: Boolean mask, True where flagged.
    """
    spectrum = np.asarray(spectrum)
    mask = np.zeros(spectrum.shape, dtype=bool)
    n_windows = max(1, min(int(n_windows), spectrum.size))
    for idx in np.array_split(np.arange(spectrum.size), n_windows):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clipped = sigma_clip(
                spectrum[idx], sigma=sigma, maxiters=None, stdfunc=mad_std, masked=True
            )
        mask[idx] = np.ma.getmaskarray(clipped)
    return mask


def flag_sigma_clip(
    ds: xr.Dataset,
    sigma: float = 3,
    n_windows: int = 100,
    polarisations: list[str] | None = None,
) -> xr.Dataset:
    """Flag outliers in each spectrum by windowed sigma-clipping.

    Each integration and polarisation is clipped separately (already-flagged
    samples are excluded); a channel is flagged in an integration if it is an
    outlier in any of the chosen polarisations.

    Args:
        ds (xr.Dataset): Band dataset.
        sigma (float, optional): Clipping threshold. Defaults to 3.
        n_windows (int, optional): Windows per spectrum. Defaults to 100.
        polarisations (list[str], optional): Polarisation products to examine.
            Defaults to the direct products (AA, BB, ...) where identifiable,
            else all.

    Returns:
        xr.Dataset: Dataset with updated flags.
    """
    data = ds["data"]
    if polarisations is None:
        labels = [str(p) for p in ds["polarisation"].to_numpy()]
        direct = [
            p for p in labels if p in ("AA", "BB", "AA+BB", "XX", "YY", "RR", "LL", "I")
        ]
        polarisations = direct or labels
    data = data.sel(polarisation=polarisations)
    data = data.where(~ds["flags"])
    data = data.chunk({"frequency": -1}) if data.chunks else data
    mask = xr.apply_ufunc(
        box_filter,
        data,
        input_core_dims=[["frequency"]],
        output_core_dims=[["frequency"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[bool],
        kwargs={"sigma": sigma, "n_windows": n_windows},
    )
    mask = mask.any(dim=[d for d in mask.dims if d not in ("time", "frequency")])
    return _or_flags(ds, mask.transpose("time", "frequency"))
