"""SDHDF history utilities"""

from __future__ import annotations

import datetime
import inspect
import socket

import pandas as pd


def generate_history_row() -> pd.DataFrame:
    """Generate a history row.
    Returns:
        pd.DataFrame: History row.
    """

    # Get the calling function from inspect
    process_name = inspect.stack()[1][3]
    # Get the calling function's docstring
    process_description = inspect.stack()[1][0].f_locals["self"].__doc__
    # Get the calling function's arguments
    process_arguments = str(inspect.stack()[1][0].f_locals["self"].__dict__)

    return pd.DataFrame(
        {
            "DATE": datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"),
            "PROC": process_name,
            "PROC_DESCR": process_description,
            "PROC_ARGS": process_arguments,
            "PROC_HOST": socket.getfqdn(),
        },
        index=[0],
    )
