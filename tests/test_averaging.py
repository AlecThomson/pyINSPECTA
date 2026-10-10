from __future__ import annotations

import numpy as np
import pytest
from conftest import data_path

from sdhdf import SDHDF, averaging


@pytest.fixture
def band():
    ds = (
        SDHDF(data_path("sdhdf_v3.0.hdf"), in_memory=True).subbands[0].astronomy_dataset
    )
    rng = np.random.default_rng(0)
    weights = rng.uniform(0.5, 2.0, ds["weights"].shape).astype(np.float32)
    flags = rng.uniform(size=ds["flags"].shape) < 0.3
    flags[:, :4] = True  # one fully flagged output channel
    return ds.assign(
        weights=(ds["weights"].dims, weights), flags=(ds["flags"].dims, flags)
    )


def test_weighted_frequency_average(band):
    n = 4
    out = averaging.average(band, frequency=n)
    x = band["data"].to_numpy()
    w = (band["weights"].to_numpy() * ~band["flags"].to_numpy())[:, None, :, None]
    nt, npol, nf, nb = x.shape
    num = (x * w).reshape(nt, npol, nf // n, n, nb).sum(3)
    den = w.reshape(nt, 1, nf // n, n, 1).sum(3)
    expected = np.where(
        den > 0,
        num / np.where(den > 0, den, 1),
        x.reshape(nt, npol, nf // n, n, nb).mean(3),
    )
    np.testing.assert_allclose(out["data"].to_numpy(), expected, rtol=1e-5)
    np.testing.assert_array_equal(out["flags"].to_numpy(), (den == 0)[:, 0, :, 0])
    np.testing.assert_allclose(out["weights"].to_numpy(), den[:, 0, :, 0], rtol=1e-6)
    np.testing.assert_allclose(
        out["frequency"].to_numpy(), band["frequency"].to_numpy().reshape(-1, n).mean(1)
    )
    assert out["flags"].to_numpy()[:, 0].all()


def test_frequency_width(band):
    width = float(np.median(np.diff(band["frequency"].to_numpy()))) * 8
    assert (
        averaging.average(band, frequency=width).sizes["frequency"]
        == band.sizes["frequency"] // 8
    )


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
    sdhdf = SDHDF(data_path("sdhdf_v4.0.hdf"))
    sdhdf.average(time=2, frequency=4)
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
    sdhdf = SDHDF(data_path("sdhdf_v4.0.hdf"))
    sdhdf.decimate(4)
    assert sdhdf.subbands[0].astronomy_dataset.sizes["frequency"] == 64
