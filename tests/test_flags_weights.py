"""Flags and weights stored with different shapes are matched to the data."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from sdhdf import SDHDF

N_TIME, N_POL, N_CHAN = 2, 4, 256


def _add(path, name, data, labels=None):
    with h5py.File(path, "r+") as h5:
        dset = h5["beam_0/band_SB0/astronomy_data"].create_dataset(name, data=data)
        if labels is not None:
            dset.attrs["DIMENSION_LABELS"] = np.array(labels, dtype=h5py.string_dtype())


@pytest.mark.parametrize(
    ("shape", "labels", "dims"),
    [
        ((N_TIME, N_CHAN), ["time", "frequency"], ("time", "frequency")),
        ((N_TIME, N_CHAN), None, ("time", "frequency")),
        (
            (N_TIME, 1, N_CHAN, 1),
            ["time", "polarisation", "frequency", "bin"],
            ("time", "frequency"),
        ),
        ((N_TIME, N_POL, N_CHAN, 1), None, ("time", "polarisation", "frequency")),
        (
            (N_TIME, N_POL, N_CHAN, 1),
            ["time", "frequency", "", ""],
            ("time", "polarisation", "frequency"),
        ),
    ],
)
def test_flag_shapes(copy_of, shape, labels, dims):
    path = copy_of("sdhdf_v3.0.hdf")
    rng = np.random.default_rng(1)
    flags = (rng.uniform(size=shape) < 0.2).astype(np.uint8)
    _add(path, "flags", flags, labels)
    ds = SDHDF.from_file(path).subbands[0].astronomy_dataset
    assert ds["flags"].dims == dims
    assert ds["flags"].dtype == bool
    np.testing.assert_array_equal(
        ds["flags"].to_numpy(), flags.reshape(ds["flags"].shape).astype(bool)
    )


def test_float_flags(copy_of):
    """The 4.0 template stores flags as float32."""
    path = copy_of("sdhdf_v3.0.hdf")
    flags = np.zeros((N_TIME, N_CHAN), dtype=np.float32)
    flags[:, :10] = 1.0
    _add(path, "flags", flags)
    ds = SDHDF.from_file(path).subbands[0].astronomy_dataset
    assert int(ds["flags"].sum()) == N_TIME * 10


def test_mismatched_flags_ignored(copy_of, caplog):
    path = copy_of("sdhdf_v3.0.hdf")
    _add(path, "flags", np.ones((N_TIME, N_CHAN + 1), dtype=np.uint8))
    ds = SDHDF.from_file(path).subbands[0].astronomy_dataset
    assert not bool(ds["flags"].any())
    assert "Could not match" in caplog.text


def test_default_weights(test_file):
    """Missing weights default to integration time x channel bandwidth (Hz)."""
    ds = SDHDF.from_file(test_file).subbands[0].astronomy_dataset
    with h5py.File(test_file) as h5:
        names: list[str] = []
        h5.visit(names.append)
        has_weights = any("astronomy_data" in n and "weights" in n for n in names)
    if has_weights:
        pytest.skip("file has stored weights")
    chan_bw = np.abs(np.gradient(ds["frequency"].to_numpy())) * 1e6
    t_int = (
        ds["INTEGRATION_TIME"].to_numpy() if "INTEGRATION_TIME" in ds.coords else None
    )
    if t_int is None:
        pytest.skip("no integration time recorded")
    np.testing.assert_allclose(
        ds["weights"].to_numpy(), np.outer(t_int, chan_bw), rtol=1e-6
    )


def test_stored_weights(copy_of):
    path = copy_of("sdhdf_v3.0.hdf")
    weights = np.random.default_rng(2).uniform(size=(N_TIME, N_CHAN)).astype(np.float32)
    _add(path, "weights", weights, ["time", "frequency"])
    ds = SDHDF.from_file(path).subbands[0].astronomy_dataset
    np.testing.assert_array_equal(ds["weights"].to_numpy(), weights)
