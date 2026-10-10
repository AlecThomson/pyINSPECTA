from __future__ import annotations

import importlib.metadata
import logging
import warnings

import sdhdf as m


def test_version():
    assert importlib.metadata.version("sdhdf") == m.__version__


def test_import_has_no_side_effects():
    """Importing must not install log handlers or silence warnings."""
    logger = logging.getLogger("sdhdf")
    assert all(
        isinstance(h, logging.NullHandler) or getattr(h, "_sdhdf", False)
        for h in logger.handlers
    )
    assert not any(f[0] == "ignore" and f[2] is Warning for f in warnings.filters)


def test_public_api():
    assert set(m.__all__) == {
        "SDHDF",
        "Beam",
        "MetaData",
        "SubBand",
        "__version__",
        "upgrade",
    }
