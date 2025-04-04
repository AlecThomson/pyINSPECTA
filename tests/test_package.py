from __future__ import annotations

import importlib.metadata

import sdhdf as m


def test_version():
    assert importlib.metadata.version("sdhdf") == m.__version__
