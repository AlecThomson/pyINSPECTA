from __future__ import annotations

import importlib.metadata

import pyINSPECTA as m


def test_version():
    assert importlib.metadata.version("pyINSPECTA") == m.__version__
