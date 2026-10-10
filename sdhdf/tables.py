"""Reading of SDHDF metadata tables (compound HDF5 datasets)."""

from __future__ import annotations

from typing import Any

import h5py
import numpy as np
import pandas as pd

from sdhdf._attrs import attr_unit, clean, decode_attrs
from sdhdf.schema import COLUMN_RENAMES

__author__ = ["Danny Price", "Alec Thomson", "Lawrence Toomey"]


class SDHDFTable:
    """An SDHDF metadata table.

    Behaves like a read-only view of a :class:`pandas.DataFrame` (indexing,
    ``len``, ``in``, iteration), with column names converted to their SDHDF
    4.0 names.

    Attributes:
        name (str): Canonical (4.0) name of the table.
        table (pd.DataFrame): Table contents.
        attrs (dict): Decoded table attributes (placeholders are ``None``).
        units (dict[str, str]): Units of the columns, where recorded.
    """

    def __init__(
        self,
        table: pd.DataFrame,
        name: str = "",
        attrs: dict[str, Any] | None = None,
        units: dict[str, str] | None = None,
    ):
        self.name = name
        self.table = table
        self.attrs = attrs or {}
        self.units = units or {}

    @classmethod
    def from_dataset(cls, dataset: h5py.Dataset, name: str) -> SDHDFTable:
        """Read a table from an HDF5 compound dataset.

        Args:
            dataset (h5py.Dataset): The dataset to read.
            name (str): Canonical name of the table kind, used to rename legacy
                columns (e.g. ``"observation_parameters"``).

        Returns:
            SDHDFTable: The table.
        """
        renames = COLUMN_RENAMES.get(name, {})
        attrs = decode_attrs(dataset.attrs)
        units = _column_units(dataset, renames)
        attrs = {renames.get(k, k): v for k, v in attrs.items()}
        table = _read_compound(dataset).rename(columns=renames)
        if name in ("observation_parameters", "calibrator_observation_parameters"):
            table = _split_mjd(table)
        return cls(table=table, name=name, attrs=attrs, units=units)

    def __repr__(self):
        return self.table.__repr__()

    def __str__(self):
        return self.table.__str__()

    def _repr_html_(self):
        return self.table._repr_html_()

    def __getitem__(self, key):
        return self.table.__getitem__(key)

    def __setitem__(self, key, value):
        return self.table.__setitem__(key, value)

    def __len__(self):
        return self.table.__len__()

    def __iter__(self):
        return self.table.__iter__()

    def __contains__(self, key):
        return self.table.__contains__(key)

    @property
    def columns(self) -> pd.Index:
        return self.table.columns

    def get(self, key: str, default: Any = None, row: int = 0) -> Any:
        """Return a single value from the table, or ``default`` if missing.

        Args:
            key (str): Column name.
            default (Any, optional): Value returned if the column is missing,
                the table is empty or the value is a placeholder.
            row (int, optional): Row to read. Defaults to 0.
        """
        if key not in self.table.columns or len(self.table) <= row:
            return default
        value = self.table[key].iloc[row]
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        return value


def _read_compound(dataset: h5py.Dataset) -> pd.DataFrame:
    """Read a compound dataset into a DataFrame, decoding strings."""
    if dataset.dtype.names is None or dataset.shape == ():
        return pd.DataFrame()
    raw = dataset[()]
    columns: dict[str, Any] = {}
    for col in dataset.dtype.names:
        values = raw[col]
        if values.dtype.kind in "SUO":
            columns[col] = pd.array([clean(v) for v in values], dtype=object)
        elif values.ndim > 1:
            columns[col] = list(values)
        else:
            columns[col] = values
    return pd.DataFrame(columns)


def _column_units(dataset: h5py.Dataset, renames: dict[str, str]) -> dict[str, str]:
    """Units of each column, from 4.0 compound attrs or legacy ``<COL>_UNIT``."""
    units = {}
    for col in dataset.dtype.names or ():
        unit = None
        if col in dataset.attrs:
            try:
                unit = attr_unit(dataset.attrs[col])
            except (OSError, TypeError):
                unit = None
        if unit is None and f"{col}_UNIT" in dataset.attrs:
            unit = clean(dataset.attrs[f"{col}_UNIT"])
        if unit is not None:
            units[renames.get(col, col)] = str(unit)
    return units


def _split_mjd(table: pd.DataFrame) -> pd.DataFrame:
    """Ensure integer ``MJD`` and ``FRACTIONAL_MJD`` columns (4.0 convention).

    Files before 4.0 store a single floating-point MJD.
    """
    if "MJD" not in table.columns or "FRACTIONAL_MJD" in table.columns:
        return table
    mjd = np.asarray(table["MJD"], dtype=float)
    day = np.floor(mjd)
    table = table.copy()
    table["MJD"] = day.astype(np.int64)
    table["FRACTIONAL_MJD"] = mjd - day
    return table
