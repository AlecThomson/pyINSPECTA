from __future__ import annotations

import shutil
from importlib import resources
from pathlib import Path

import pytest

TEMPLATE_DIR = Path(__file__).parent / "data" / "templates"
TEST_FILES = sorted(
    p.name
    for p in resources.files("sdhdf.data.tests").iterdir()
    if p.name.endswith(".hdf")
)


def data_path(name: str) -> Path:
    if name.startswith("sdhdf_template"):
        return TEMPLATE_DIR / name
    with resources.as_file(resources.files("sdhdf.data.tests") / name) as path:
        return Path(path)


@pytest.fixture(params=TEST_FILES + sorted(p.name for p in TEMPLATE_DIR.glob("*.hdf")))
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
