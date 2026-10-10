from __future__ import annotations

import shutil
from importlib import resources
from pathlib import Path

import pytest

TEST_FILES = sorted(
    p.name
    for p in resources.files("sdhdf.data.tests").iterdir()
    if p.name.endswith(".hdf")
)
TEMPLATE_FILES = sorted(
    p.name
    for p in resources.files("sdhdf.data.templates").iterdir()
    if p.name.endswith(".hdf")
)


def data_path(name: str) -> Path:
    package = (
        "sdhdf.data.templates"
        if name.startswith("sdhdf_template")
        else "sdhdf.data.tests"
    )
    with resources.as_file(resources.files(package) / name) as path:
        return Path(path)


@pytest.fixture(params=TEST_FILES + TEMPLATE_FILES)
def any_file(request) -> Path:
    """Every bundled test file and SDHDF template."""
    return data_path(request.param)


@pytest.fixture(params=TEST_FILES)
def test_file(request) -> Path:
    """Every bundled test file (real observations)."""
    return data_path(request.param)


@pytest.fixture
def copy_of(tmp_path):
    """Copy a bundled file to a temporary location for modification."""

    def _copy(name: str) -> Path:
        dst = tmp_path / name
        shutil.copy(data_path(name), dst)
        return dst

    return _copy
