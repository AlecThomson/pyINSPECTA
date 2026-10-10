"""Files with attributes damaged by older processing software must still load."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from sdhdf import SDHDF
from sdhdf.attributes import decode_attr, is_placeholder
from sdhdf.constants import DATA_DIMS


@pytest.mark.parametrize(
    "value",
    ["NOT FOUND", "NOT SET", "not defined", "UNKNOWN", "NULL", "FIX ME 2", "None", ""],
)
def test_placeholders(value):
    assert is_placeholder(value)
    assert is_placeholder(value.encode())
    assert decode_attr(np.bytes_(value)) is None


@pytest.mark.parametrize(
    "value", ["MHz", "0.0", "AABBCRCI", "topocentric", "None of these"]
)
def test_not_placeholders(value):
    assert not is_placeholder(value)


def test_compound_attrs():
    value = np.array(
        [(b"Data unit", b"None", b"MHz")],
        dtype=[("description", "S9"), ("unit", "S4"), ("value", "S3")],
    )
    assert decode_attr(value) == "MHz"
    template = np.array(
        [(b"d", b"None", b"counts")],
        dtype=[("description", "S1"), ("unit", "S4"), ("default", "S6")],
    )
    assert decode_attr(template) == "counts"
    missing = np.array(
        [(b"d", b"None", b"NOT FOUND")],
        dtype=[("description", "S1"), ("unit", "S4"), ("value", "S9")],
    )
    assert decode_attr(missing) is None


def collapse_compound_attrs(obj):
    """Mimic INSPECTA rewriting compound attributes as scalar strings."""
    for key in list(obj.attrs):
        value = obj.attrs[key]
        if (
            isinstance(value, np.ndarray)
            and value.dtype.names
            and "value" in value.dtype.names
        ):
            del obj.attrs[key]
            obj.attrs[key] = np.bytes_(value[0]["value"])


def replace_column(dset: h5py.Dataset, column: str, value: bytes):
    """Replace a table column value, widening the string type if needed."""
    data = dset[()]
    dtype = [
        (
            n,
            f"S{max(len(value), data.dtype[n].itemsize)}"
            if n == column
            else data.dtype[n],
        )
        for n in data.dtype.names
    ]
    new = np.zeros(data.shape, dtype=dtype)
    for n in data.dtype.names:
        new[n] = data[n]
    new[column] = value
    parent, name = dset.parent, dset.name.split("/")[-1]
    attrs = dict(dset.attrs)
    del parent[name]
    out = parent.create_dataset(name, data=new)
    for k, v in attrs.items():
        out.attrs[k] = v


@pytest.fixture
def damaged_v4(copy_of):
    path = copy_of("sdhdf_v4.0.hdf")
    with h5py.File(path, "r+") as h5:
        h5.visititems(lambda _, obj: collapse_compound_attrs(obj))
        band = h5["beam_00/band_SB0"]
        data = band["astronomy_data/data"]
        del data.attrs["DIMENSION_LABELS"]
        data.attrs["DIMENSION_LABELS"] = np.bytes_("NOT FOUND")
        freq = band["astronomy_data/frequency"]
        freq.attrs["UNIT"] = np.bytes_("NOT FOUND")
        freq.attrs["CLASS"] = np.bytes_("FIX ME 2")
        data.attrs["uncertain"] = np.bytes_("not defined")
        cal = band["calibrator_data/calibrator_data_on"]
        del cal.attrs["DIMENSION_LABELS"]
        replace_column(
            h5["metadata/primary_header"], "HEADER_DEFINITION_VERSION", b"NOT FOUND"
        )
        h5["metadata/primary_header"].attrs["HEADER_DEFINITION_VERSION"] = np.bytes_(
            "NOT FOUND"
        )
        replace_column(h5["metadata/primary_header"], "TELESCOPE", b"UNKNOWN")
    return path


def test_damaged_v4_loads(damaged_v4):
    sdhdf = SDHDF(damaged_v4)
    assert sdhdf.metadata.version_info.recorded is None
    assert sdhdf.metadata.version == "4.0"  # inferred from structure
    assert sdhdf.metadata.telescope is None
    sb = sdhdf.subbands[0]
    ds = sb.astronomy_dataset
    assert ds["data"].dims == DATA_DIMS
    assert ds.sizes["polarisation"] == 4
    assert ds["frequency"].attrs["units"] == "MHz"  # default when unit is a placeholder
    assert "uncertain" not in ds["data"].attrs
    assert sb.calibrator_dataset is not None
    assert sb.calibrator_dataset["on"].dims == DATA_DIMS


def test_damaged_file_flags_with_default_telescope(damaged_v4):
    sdhdf = SDHDF(damaged_v4)
    sdhdf.flag_persistent_rfi()  # warns and assumes Parkes
    assert sdhdf.history["PROCESS"].iloc[-1] == "sdhdf.flag_persistent_rfi"


def test_damaged_dimension_labels_v3(copy_of):
    path = copy_of("sdhdf_v3.0.hdf")
    with h5py.File(path, "r+") as h5:
        for name in ("data", "frequency"):
            dset = h5[f"beam_0/band_SB0/astronomy_data/{name}"]
            del dset.attrs["DIMENSION_LABELS"]
            dset.attrs["DIMENSION_LABELS"] = np.bytes_("NOT SET")
    ds = SDHDF(path).subbands[0].astronomy_dataset
    assert ds["data"].dims == DATA_DIMS
    assert ds.sizes["frequency"] == 256


def test_damaged_upgrade(damaged_v4, tmp_path):
    from sdhdf import upgrade  # noqa: PLC0415

    out = tmp_path / "upgraded.hdf"
    upgrade(damaged_v4, out)
    upgraded = SDHDF(out)
    assert upgraded.metadata.version_info.recorded == "4.0"
    np.testing.assert_array_equal(
        upgraded.subbands[0].astronomy_dataset["data"].to_numpy(),
        SDHDF(damaged_v4).subbands[0].astronomy_dataset["data"].to_numpy(),
    )


def key_attrs_by_description(obj):
    """Mimic INSPECTA writing attributes keyed by their description."""
    for key in list(obj.attrs):
        value = obj.attrs[key]
        if (
            isinstance(value, np.ndarray)
            and value.dtype.names
            and "description" in value.dtype.names
        ):
            description = value[0]["description"].decode()
            del obj.attrs[key]
            obj.attrs[description] = np.bytes_(value[0]["value"])
    if "DIMENSION_LABELS" in obj.attrs:
        del obj.attrs["DIMENSION_LABELS"]


def test_description_keyed_attributes(copy_of):
    """Attributes keyed by description are renamed; labels are rebuilt."""
    path = copy_of("sdhdf_v4.0.hdf")
    with h5py.File(path, "r+") as h5:
        data = h5["beam_00/band_SB0/astronomy_data/data"]
        unit = data.attrs["UNIT"].copy()
        unit[0]["value"] = b"Jy"
        data.attrs["UNIT"] = unit
        for key, description, value in (
            ("TIME", "Integrations, data array dimension 0", "time"),
            ("PRODUCT_TYPE", "Product type, data array dimension 1", "polarisation"),
            ("FREQUENCY", "Frequency, data array dimension 2", "frequency"),
            ("PHASE_BIN", "Phase bins, data array dimension 3", "bin"),
        ):
            data.attrs[key] = np.array(
                [(description.encode(), b"None", value.encode())],
                dtype=[("description", "S40"), ("unit", "S4"), ("value", "S12")],
            )
        h5.visititems(lambda _, obj: key_attrs_by_description(obj))
        assert "UNIT" not in data.attrs

    from sdhdf.attributes import decode_attrs  # noqa: PLC0415

    with h5py.File(path) as h5:
        attrs = decode_attrs(h5["beam_00/band_SB0/astronomy_data/data"].attrs)
    assert attrs["UNIT"] == "Jy"
    assert attrs["DIMENSION_LABELS"] == ["time", "polarisation", "frequency", "bin"]
    ds = SDHDF(path).subbands[0].astronomy_dataset
    assert ds["data"].attrs["units"] == "Jy"
    assert ds["data"].dims == DATA_DIMS
