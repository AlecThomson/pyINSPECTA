from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import data_path

from sdhdf import SDHDF, flagging
from sdhdf.constants import RFI_COLUMNS

V3 = "sdhdf_v3.0.hdf"  # 829-830 MHz band


def _rfi(f0, f1, rcvr="all", mjd0=0, mjd1=999999, obs="Parkes"):
    return pd.DataFrame(
        [[1, obs, rcvr, f0, f1, mjd0, mjd1, "test"]], columns=RFI_COLUMNS
    )


@pytest.fixture
def band():
    return SDHDF(data_path(V3)).subbands[0].astronomy_dataset


def test_bundled_table_does_not_flag_clean_band(band):
    """Regression: previously every channel outside any RFI range was flagged."""
    rfi = flagging.get_persistent_rfi("Parkes")
    out = flagging.flag_persistent_rfi(band, rfi, telescope="Parkes")
    assert not bool(out["flags"].any())


def test_flags_only_overlapping_channels(band):
    out = flagging.flag_persistent_rfi(band, _rfi(829.5, 831.0))
    freq = out["frequency"].to_numpy()
    flags = out["flags"].to_numpy()
    assert flags[:, freq >= 829.5].all()
    assert not flags[:, freq < 829.5].any()


def test_receiver_and_telescope_selection(band):
    assert not flagging.flag_persistent_rfi(
        band, _rfi(829, 831, rcvr="MB"), receiver="UWL"
    )["flags"].any()
    assert flagging.flag_persistent_rfi(
        band, _rfi(829, 831, rcvr="UWL"), receiver="uwl"
    )["flags"].all()
    assert flagging.flag_persistent_rfi(band, _rfi(829, 831), telescope="Murriyang")[
        "flags"
    ].all()
    assert not flagging.flag_persistent_rfi(
        band, _rfi(829, 831, obs="ATCA"), telescope="Parkes"
    )["flags"].any()


def test_mjd_range(band):
    mjd = band["time"].to_numpy()
    out = flagging.flag_persistent_rfi(
        band, _rfi(829, 831, mjd0=mjd[1] - 1e-7, mjd1=mjd[1] + 1)
    )
    assert not out["flags"].isel(time=0).any()
    assert out["flags"].isel(time=1).all()


def test_flags_are_never_cleared(band):
    first = flagging.flag_persistent_rfi(band, _rfi(829.0, 829.2))
    second = flagging.flag_persistent_rfi(first, _rfi(829.8, 830.0))
    assert int(second["flags"].sum()) == int(first["flags"].sum()) + int(
        flagging.flag_persistent_rfi(band, _rfi(829.8, 830.0))["flags"].sum()
    )


def test_box_filter_covers_all_channels():
    spectrum = np.ones(1050)
    spectrum[-3] = 1e6  # in the remainder that used to be skipped
    mask = flagging.box_filter(spectrum, sigma=3, n_windows=100)
    assert mask[-3]
    assert mask.sum() == 1


def test_box_filter_more_windows_than_channels():
    spectrum = np.ones(10)
    spectrum[5] = 100
    assert flagging.box_filter(spectrum, n_windows=100).shape == (10,)


def test_sigma_clip_finds_spike(band):
    data = band["data"].load()
    data[:, :2, 100, :] = 1e9
    out = flagging.flag_sigma_clip(band.assign(data=data), sigma=5, n_windows=8)
    assert out["flags"].dims == ("time", "frequency")
    assert out["flags"].isel(frequency=100).all()


def test_sdhdf_auto_flag_records_history():
    sdhdf = SDHDF(data_path(V3))
    sdhdf.auto_flag_rfi()
    assert list(sdhdf.history["PROCESS"].iloc[-2:]) == [
        "sdhdf.flag_persistent_rfi",
        "sdhdf.auto_flag_rfi",
    ]
