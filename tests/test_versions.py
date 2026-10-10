from __future__ import annotations

import numpy as np
import pytest
from conftest import data_path

from sdhdf import SDHDF
from sdhdf.schema import DATA_DIMS

EXPECTED_VERSIONS = {
    "sdhdf_v1.9.3.hdf": "1.9.3",
    "sdhdf_v2.0.hdf": "2.0",
    "sdhdf_v2.1.hdf": "2.1",
    "sdhdf_v2.2.hdf": "2.2",
    "sdhdf_v3.0.hdf": "3.0",
    "sdhdf_v4.0.hdf": "4.0",
    "sdhdf_template_v2.9.hdf": "2.9",
    "sdhdf_template_v3.0.hdf": "3.0",
    "sdhdf_template_v4.0.hdf": "4.0",
}


def test_version_detected(any_file):
    sdhdf = SDHDF(any_file)
    assert sdhdf.metadata.version == EXPECTED_VERSIONS[any_file.name]


def test_common_representation(any_file):
    """Every version is read into the same canonical structure."""
    sdhdf = SDHDF(any_file)
    assert sdhdf.beams
    for sb in sdhdf.subbands:
        ds = sb.astronomy_dataset
        assert ds["data"].dims == DATA_DIMS
        assert set(ds["flags"].dims) <= set(DATA_DIMS)
        assert set(ds["weights"].dims) <= set(DATA_DIMS)
        assert ds["flags"].dtype == bool
        assert ds.sizes["frequency"] == ds["frequency"].size
        assert all(isinstance(d, str) and type(d) is str for d in ds["data"].dims)
        assert ds["frequency"].attrs["units"] == "MHz"
        if ds.sizes["polarisation"] == 4:
            assert list(ds["polarisation"].to_numpy()) == ["AA", "BB", "CR", "CI"]
        assert np.all(np.asarray(ds["weights"]) >= 0)


def test_metadata_tables_use_v4_names(test_file):
    sdhdf = SDHDF(test_file)
    ph = sdhdf.metadata.primary_header
    for col in (
        "HEADER_DEFINITION_VERSION",
        "TELESCOPE",
        "RECEIVER",
        "CALIBRATION_MODE",
    ):
        assert col in ph
    ds = sdhdf.subbands[0].astronomy_dataset
    for coord in (
        "MJD",
        "FRACTIONAL_MJD",
        "PARALLACTIC_ANGLE",
        "AZIMUTH_ANGLE",
        "ELEVATION_ANGLE",
    ):
        assert coord in ds.coords, coord
    mjd = ds["MJD"].to_numpy() + ds["FRACTIONAL_MJD"].to_numpy()
    np.testing.assert_allclose(ds["time"].to_numpy(), mjd)
    assert np.all(ds["time"].to_numpy() > 50000)


@pytest.mark.parametrize(
    ("name", "n_cal_chan"),
    [("sdhdf_v2.0.hdf", 128), ("sdhdf_v3.0.hdf", 256), ("sdhdf_v4.0.hdf", 256)],
)
def test_calibrator_data(name, n_cal_chan):
    cal = SDHDF(data_path(name)).subbands[0].calibrator_dataset
    assert cal is not None
    assert {"on", "off", "binned"} <= set(cal.data_vars)
    assert cal.sizes["frequency"] == n_cal_chan
    assert cal.sizes["cal_bin"] == 32


def test_lazy_and_in_memory_agree(test_file):
    lazy = SDHDF(test_file).subbands[0].astronomy_dataset
    eager = SDHDF(test_file, in_memory=True).subbands[0].astronomy_dataset
    assert lazy["data"].chunks is not None
    assert eager["data"].chunks is None
    np.testing.assert_array_equal(lazy["data"].to_numpy(), eager["data"].to_numpy())


def test_lazy_arrays_pickle(test_file):
    import pickle  # noqa: PLC0415

    ds = SDHDF(test_file).subbands[0].astronomy_dataset
    restored = pickle.loads(pickle.dumps(ds))
    np.testing.assert_array_equal(restored["data"].to_numpy(), ds["data"].to_numpy())
