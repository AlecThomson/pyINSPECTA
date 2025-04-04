"""SDHDF flagging utilities"""

from __future__ import annotations

import warnings
from importlib import resources

import numpy as np
import pandas as pd
from astropy.stats import mad_std, sigma_clip
from xarray import DataArray


class AutoFlagError(Exception):
    def __init__(self, msg):
        self.msg = msg


def get_persistent_rfi(telescope: str = "Parkes") -> pd.DataFrame:
    """Read persistent RFI file
    Returns:
        pd.Dataframe: Persistent RFI data.
    """
    with resources.as_file(resources.files("pyINSPECTA.data.rfi")) as rfi_dir:
        rfi_file = rfi_dir / f"{telescope.lower()}_rfi.csv"

    if not rfi_file.exists():
        msg = (
            f"Persistent RFI file for {telescope} not found at '{rfi_file.absolute()}'."
        )
        raise NotImplementedError(msg)
    return pd.read_csv(
        rfi_file,
        sep=",",
        # skip_blank_lines=True,
        comment="#",
        names=[
            "type",
            "observatory label",
            "receiver label",
            "freq0 MHz",
            "freq1 MHz",
            "MJD0",
            "MJD1",
            "text string for label",
        ],
    )


def box_filter(spectrum: np.ndarray | DataArray, sigma=3, n_windows=100):
    """
    Filter a spectrum using a box filter.
    """
    # Divide spectrum into windows
    spectrum_squeezed = spectrum.squeeze()
    window_size = len(spectrum_squeezed) // n_windows
    dat_filt = np.zeros_like(spectrum_squeezed).astype(bool)

    # Iterate through windows
    for i in range(n_windows):
        _dat = spectrum_squeezed[i * window_size : window_size + i * window_size]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Use sigma clipping to remove outliers
            _dat_filt = sigma_clip(
                _dat, sigma=sigma, maxiters=None, stdfunc=mad_std, masked=True
            )
        dat_filt[i * window_size : window_size + i * window_size] = _dat_filt.mask

    dat_filt = dat_filt.reshape(spectrum.shape)

    if isinstance(spectrum, DataArray):
        return DataArray(dat_filt, dims=spectrum.dims, coords=spectrum.coords)

    return dat_filt
