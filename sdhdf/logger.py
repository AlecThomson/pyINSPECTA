"""Logging module for sdhdf

Messages at INFO and above are printed by default. If the application has
configured logging itself (e.g. with ``logging.basicConfig``), messages go
through that configuration instead, so nothing is printed twice. Use
:func:`set_verbosity` to change the level.
"""

from __future__ import annotations

import logging
import sys


class _DefaultHandler(logging.StreamHandler):
    """Print messages unless the root logger has handlers of its own."""

    def emit(self, record: logging.LogRecord) -> None:
        if not logging.getLogger().handlers:
            self.stream = sys.stderr  # stderr may be replaced (e.g. by Jupyter)
            super().emit(record)


logger = logging.getLogger("sdhdf")
logger.setLevel(logging.INFO)
_handler = _DefaultHandler()
_handler.setFormatter(logging.Formatter("%(name)s - %(levelname)s: %(message)s"))
logger.addHandler(_handler)


def set_verbosity(level: int | str = logging.INFO) -> None:
    """Set the level of sdhdf log messages, e.g. ``"DEBUG"`` or ``"WARNING"``.

    Args:
        level (int | str, optional): Logging level. Defaults to INFO.
    """
    logger.setLevel(level)
