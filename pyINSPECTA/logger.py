"""Logging module for pyINSPECTA"""

from __future__ import annotations

import logging

# Create logger
logger = logging.getLogger("pyINSPECTA")
logger.setLevel(logging.INFO)

# Create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# Create formatter
formatter = logging.Formatter("%(name)s - %(levelname)s: %(message)s")

# Add formatter to ch
ch.setFormatter(formatter)

# Add ch to logger
logger.addHandler(ch)
