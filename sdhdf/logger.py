"""Logging module for sdhdf

The package logs to the ``sdhdf`` logger and does not configure handlers, so
that applications control where messages go. Use :func:`set_verbosity` for
quick interactive use.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("sdhdf")
logger.addHandler(logging.NullHandler())


def set_verbosity(level: int | str = logging.INFO) -> None:
    """Print sdhdf log messages at or above ``level`` to stderr.

    Args:
        level (int | str, optional): Logging level. Defaults to INFO.
    """
    logger.setLevel(level)
    if not any(getattr(h, "_sdhdf", False) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(name)s - %(levelname)s: %(message)s"))
        handler._sdhdf = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
