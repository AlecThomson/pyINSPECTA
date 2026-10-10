"""Constants used throughout the package.

The SDHDF format itself (object names across versions, column renames and so
on) is described by ``sdhdf/data/definition.toml``; see :mod:`sdhdf.definition`.
"""

from __future__ import annotations

#: SDHDF definition version written by this package.
LATEST_VERSION = "4.0"

#: Canonical dimension names, in SDHDF order, for data-like arrays.
DATA_DIMS = ("time", "polarisation", "frequency", "bin")
