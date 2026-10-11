from __future__ import annotations

import importlib.metadata
import logging
import warnings

import sdhdf as m


def test_version():
    assert importlib.metadata.version("sdhdf") == m.__version__


def test_import_does_not_silence_warnings():
    assert not any(f[0] == "ignore" and f[2] is Warning for f in warnings.filters)


def test_logging_prints_once(capsys):
    """INFO is printed by default, but not when the root logger handles it."""
    logger = logging.getLogger("sdhdf")
    root = logging.getLogger()
    saved = root.handlers[:]
    try:
        root.handlers = []
        logger.info("default")
        assert "default" in capsys.readouterr().err
        root.addHandler(logging.NullHandler())
        logger.info("configured")
        assert "configured" not in capsys.readouterr().err
    finally:
        root.handlers = saved


def test_public_api():
    assert set(m.__all__) == {
        "SDHDF",
        "Beam",
        "MetaData",
        "SubBand",
        "__version__",
        "upgrade",
    }
