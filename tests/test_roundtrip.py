"""Every version can be written as (upgraded to) SDHDF 4.0 without loss."""

from __future__ import annotations

import h5py
import numpy as np
import pytest
from conftest import data_path

from sdhdf import SDHDF, upgrade


def _assert_same(a, b):
    for var in a.data_vars:
        np.testing.assert_array_equal(
            a[var].to_numpy(), b[var].transpose(*a[var].dims).to_numpy(), err_msg=var
        )
    for name, coord in a.coords.items():
        if coord.dtype.kind in "fiu":
            np.testing.assert_allclose(
                coord.to_numpy().astype(float),
                b[name].to_numpy().astype(float),
                err_msg=name,
            )
        else:
            assert list(coord.to_numpy()) == list(b[name].to_numpy()), name


def test_upgrade_roundtrip(any_file, tmp_path):
    out = tmp_path / "upgraded.hdf"
    upgrade(any_file, out)
    original, upgraded = SDHDF(any_file), SDHDF(out)
    assert upgraded.metadata.version_info.recorded == "4.0"
    assert set(original.metadata.tables) <= set(upgraded.metadata.tables)
    assert len(upgraded.history) == len(original.history) + 2  # upgrade + write
    for a, b in zip(original.subbands, upgraded.subbands, strict=True):
        _assert_same(a.astronomy_dataset, b.astronomy_dataset)
        if a.calibrator_dataset is not None:
            _assert_same(a.calibrator_dataset, b.calibrator_dataset)


def test_written_layout_is_v4(tmp_path):
    out = tmp_path / "upgraded.hdf"
    upgrade(data_path("sdhdf_v2.0.hdf"), out)
    with h5py.File(out) as h5:
        for path in (
            "metadata/primary_header",
            "metadata/beam_parameters",
            "configuration/instrument_configuration",
            "beam_0/metadata/band_parameters",
            "beam_0/band_SB0/astronomy_data/data",
            "beam_0/band_SB0/astronomy_data/flags",
            "beam_0/band_SB0/astronomy_data/weights",
            "beam_0/band_SB0/calibrator_data/calibrator_data_on",
            "beam_0/band_SB0/metadata/observation_parameters",
        ):
            assert path in h5, path
        data = h5["beam_0/band_SB0/astronomy_data/data"]
        assert data.ndim == 4
        assert data.attrs["UNIT"].dtype.names == ("description", "unit", "value")
        assert h5["beam_0/band_SB0/astronomy_data/flags"].dtype == np.uint8
        assert h5["beam_0/band_SB0/astronomy_data/frequency"].ndim == 2
        obs = h5["beam_0/band_SB0/metadata/observation_parameters"]
        assert {"MJD", "FRACTIONAL_MJD", "PARALLACTIC_ANGLE"} <= set(obs.dtype.names)


def test_write_refuses_overwrite(tmp_path):
    sdhdf = SDHDF(data_path("sdhdf_v4.0.hdf"))
    out = tmp_path / "out.hdf"
    sdhdf.write(out)
    with pytest.raises(FileExistsError):
        sdhdf.write(out)
    sdhdf.write(out, overwrite=True)


def test_processed_file_roundtrip(tmp_path):
    sdhdf = SDHDF(data_path("sdhdf_v3.0.hdf"))
    sdhdf.flag_persistent_rfi()
    sdhdf.average(frequency=4)
    out = tmp_path / "processed.hdf"
    sdhdf.write(out)
    written = SDHDF(out)
    _assert_same(
        sdhdf.subbands[0].astronomy_dataset, written.subbands[0].astronomy_dataset
    )
    row = written.beams[0].band_parameters.table.iloc[0]
    assert row["NUMBER_OF_CHANNELS"] == 64
    assert list(written.history["PROCESS"].iloc[-3:]) == [
        "sdhdf.flag_persistent_rfi",
        "sdhdf.average",
        "sdhdf.write",
    ]


def test_cli_upgrade(tmp_path):
    from sdhdf.cli import upgrade_main  # noqa: PLC0415

    out = tmp_path / "cli.hdf"
    upgrade_main([str(data_path("sdhdf_v2.1.hdf")), str(out)])
    assert SDHDF(out).metadata.version_info.recorded == "4.0"


def test_observation_parameters_complete(tmp_path):
    """Written observation_parameters always have the columns INSPECTA reads."""
    from sdhdf.writer import REQUIRED_OBS_COLUMNS  # noqa: PLC0415

    out = tmp_path / "template.hdf"
    upgrade(data_path("sdhdf_template_v4.0.hdf"), out)  # template table is empty
    with h5py.File(out) as h5:
        obs = h5["beam_0/band_0/metadata/observation_parameters"]
        assert set(REQUIRED_OBS_COLUMNS) <= set(obs.dtype.names)
