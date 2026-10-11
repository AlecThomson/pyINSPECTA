from __future__ import annotations

import h5py
import pytest
from conftest import data_path

from sdhdf.definition import definition


def test_candidates_are_newest_first():
    """The first candidate of every role is its SDHDF 4.0 name, as written."""
    with h5py.File(data_path("sdhdf_v4.0.hdf")) as h5:
        band = h5["beam_00/band_SB0"]
        for role in ("data", "frequency"):
            assert definition().astronomy_datasets[role][0] in band["astronomy_data"]
        for role in ("on", "off", "binned"):
            assert definition().calibrator_datasets[role][0] in band["calibrator_data"]
        assert definition().band_tables["observation_parameters"][0] in band["metadata"]
        assert definition().groups["configuration"][0] in h5


def test_calibrator_tables_share_renames():
    d = definition()
    assert (
        d.renames_for("calibrator_band_parameters")
        is d.column_renames["band_parameters"]
    )
    assert (
        d.renames_for("calibrator_observation_parameters")["PARA_ANGLE"]
        == "PARALLACTIC_ANGLE"
    )
    assert d.renames_for("no_such_table") == {}


@pytest.mark.parametrize("npol", [1, 2, 4])
def test_default_polarisation_types_exist(npol):
    d = definition()
    assert len(d.polarisation_types[d.default_polarisation_types[npol]]) == npol
