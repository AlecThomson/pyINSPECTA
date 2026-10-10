"""SDHDF history utilities"""

from __future__ import annotations

import datetime
import json
import socket
from typing import Any

import pandas as pd


def generate_history_row(
    process: str,
    description: str = "",
    arguments: dict[str, Any] | None = None,
    log: str = "",
) -> pd.DataFrame:
    """Generate a history row.

    Args:
        process (str): Name of the process, e.g. ``"sdhdf.average"``.
        description (str, optional): What the process did.
        arguments (dict, optional): Arguments the process was called with.
        log (str, optional): Logged output from the process.

    Returns:
        pd.DataFrame: A single-row history table.
    """
    return pd.DataFrame(
        {
            "DATE": datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y-%m-%d-%H:%M:%S"
            ),
            "PROCESS": process,
            "PROCESS_DESCRIPTION": description,
            "PROCESS_ARGUMENTS": json.dumps(
                arguments or {}, default=repr, sort_keys=True
            ),
            "PROCESSING_HOST": socket.gethostname(),
            "PROCESS_LOG": log,
        },
        index=[0],
    )


def append(history: pd.DataFrame | None, *rows: pd.DataFrame) -> pd.DataFrame:
    """Append history rows to a history table."""
    frames = [f for f in (history, *rows) if f is not None and len(f)]
    if not frames:
        return generate_history_row("").iloc[:0]
    return pd.concat(frames, ignore_index=True)
