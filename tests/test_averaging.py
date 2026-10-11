from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest
from conftest import data_path

from sdhdf import SDHDF, averaging


@pytest.fixture
def band():
    ds = (
        SDHDF.from_file(data_path("sdhdf_v3.0.hdf"), in_memory=True)
        .subbands[0]
        .astronomy_dataset
    )
    rng = np.random.default_rng(0)
    weights = rng.uniform(0.5, 2.0, ds["weights"].shape).astype(np.float32)
    flags = rng.uniform(size=ds["flags"].shape) < 0.3
    flags[:, :4] = True  # one fully flagged output channel
    return ds.assign(
        weights=(ds["weights"].dims, weights), flags=(ds["flags"].dims, flags)
    )


def test_weighted_frequency_average(band):
    """CASA rule: unflagged weighted mean; fully flagged bins average all inputs."""
    n = 4
    out = averaging.average(band, frequency=n)
    x = band["data"].to_numpy()
    nt, npol, nf, nb = x.shape
    flags = band["flags"].to_numpy()
    w_all = band["weights"].to_numpy()[:, None, :, None]
    w_good = (band["weights"].to_numpy() * ~flags)[:, None, :, None]

    def block_sum(a):
        return a.reshape(a.shape[0], a.shape[1], nf // n, n, a.shape[3]).sum(3)

    all_flagged = flags.reshape(nt, nf // n, n).all(2)[:, None, :, None]
    expected = np.where(
        all_flagged,
        block_sum(x * w_all) / block_sum(w_all),
        block_sum(x * w_good) / np.where(all_flagged, 1, block_sum(w_good)),
    )
    weights = np.where(all_flagged, block_sum(w_all), block_sum(w_good))
    np.testing.assert_allclose(out["data"].to_numpy(), expected, rtol=1e-5)
    np.testing.assert_array_equal(out["flags"].to_numpy(), all_flagged[:, 0, :, 0])
    np.testing.assert_allclose(
        out["weights"].to_numpy(), weights[:, 0, :, 0], rtol=1e-6
    )
    np.testing.assert_allclose(
        out["frequency"].to_numpy(), band["frequency"].to_numpy().reshape(-1, n).mean(1)
    )
    assert out["flags"].to_numpy()[:, 0].all()


@pytest.mark.parametrize("unit", [u.MHz, u.kHz, u.Hz])
def test_frequency_width_quantity(band, unit):
    channel_width_mhz = float(np.median(np.diff(band["frequency"].to_numpy())))
    width = (8 * channel_width_mhz * u.MHz).to(unit)
    averaged = averaging.average(band, frequency=width)
    assert averaged.sizes["frequency"] == band.sizes["frequency"] // 8


def test_time_width_quantity(band):
    integration_time_s = float(band["INTEGRATION_TIME"][0])
    averaged = averaging.average(band, time=2 * integration_time_s * u.s)
    assert averaged.sizes["time"] == 1


@pytest.mark.parametrize("value", [4.0, 0.5])
def test_plain_float_is_ambiguous(band, value):
    with pytest.raises(TypeError, match="number of samples"):
        averaging.average(band, frequency=value)


def test_time_average_metadata(band):
    out = averaging.average(band, time=2)
    assert out.sizes["time"] == 1
    np.testing.assert_allclose(
        out["INTEGRATION_TIME"].to_numpy(), band["INTEGRATION_TIME"].to_numpy().sum()
    )
    np.testing.assert_allclose(out["time"].to_numpy(), band["time"].to_numpy().mean())
    np.testing.assert_allclose(
        out["MJD"].to_numpy() + out["FRACTIONAL_MJD"].to_numpy(), out["time"].to_numpy()
    )
    assert out["UTC"].to_numpy()[0] in band["UTC"].to_numpy()


def test_median(band):
    out = averaging.average(band, frequency=4, method="median")
    assert out.sizes["frequency"] == band.sizes["frequency"] // 4


def test_invalid_factor(band):
    with pytest.raises(ValueError, match="Cannot average"):
        averaging.average(band, time=10)


def test_sdhdf_average_lazy_and_records_history():
    sdhdf = SDHDF.from_file(data_path("sdhdf_v4.0.hdf"))
    sdhdf = sdhdf.average(time=2, frequency=4)
    sb = sdhdf.subbands[0]
    assert sb.astronomy_dataset["data"].chunks is not None  # still lazy
    assert dict(sb.astronomy_dataset["data"].sizes) == {
        "time": 1,
        "polarisation": 4,
        "frequency": 64,
        "bin": 1,
    }
    assert sb.calibrator_dataset.sizes["time"] == 1
    assert sdhdf.history["PROCESS"].iloc[-1] == "sdhdf.average"


def test_decimate_compat():
    sdhdf = SDHDF.from_file(data_path("sdhdf_v4.0.hdf"))
    assert sdhdf.decimate(4).subbands[0].astronomy_dataset.sizes["frequency"] == 64
    freq = sdhdf.subbands[0].astronomy_dataset["frequency"].to_numpy()
    width = 4 * float(np.median(np.diff(freq)))  # float: width in the axis units
    assert sdhdf.decimate(width).subbands[0].astronomy_dataset.sizes["frequency"] == 64


def test_methods_return_new_objects():
    sdhdf = SDHDF.from_file(data_path("sdhdf_v4.0.hdf"))
    n_history = len(sdhdf.history)
    averaged = sdhdf.flag_persistent_rfi().average(frequency=4)
    assert averaged is not sdhdf
    assert sdhdf.subbands[0].astronomy_dataset.sizes["frequency"] == 256
    assert len(sdhdf.history) == n_history
    assert len(averaged.history) == n_history + 2
