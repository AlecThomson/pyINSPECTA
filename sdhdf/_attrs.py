"""Decoding of SDHDF HDF5 attributes and cell values.

SDHDF attributes have been stored in several ways over the life of the format:

* plain (byte-)strings and numbers (SDHDF < 4.0),
* compound ``(description, unit, value)`` arrays (SDHDF 4.0 files),
* compound ``(description, unit, default)`` arrays (SDHDF 4.0 templates),
* arrays of strings (e.g. ``DIMENSION_LABELS``).

In addition, files processed by older versions of INSPECTA can contain
damaged attributes: compound attributes collapsed into scalar strings, and
placeholder values such as ``"NOT SET"``, ``"NOT FOUND"``, ``"UNKNOWN"`` or
``"FIX ME 2"``. Everything in this module treats such placeholders as missing.
"""

from __future__ import annotations

import re
from typing import Any

import h5py
import numpy as np

_PLACEHOLDER_RE = re.compile(
    r"""^\s*(
        not[\s_]*(set|found|defined|available|applicable)
        | unknown | uncertain | undefined
        | null | none | nan | n/?a
        | fix[\s_]*me.*
        | string(\s*\(.*\))?
    )\s*$""",
    re.IGNORECASE | re.VERBOSE,
)

#: Field names that hold the value in compound SDHDF attributes, in priority order.
_VALUE_FIELDS = ("value", "default")


def is_placeholder(value: Any) -> bool:
    """Return True if ``value`` is a placeholder meaning "no value".

    Args:
        value (Any): A decoded value.

    Returns:
        bool: True for ``None``, empty strings and known placeholder strings.
    """
    if value is None:
        return True
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value.strip() == "" or bool(_PLACEHOLDER_RE.match(value))
    return False


def decode_str(value: Any) -> Any:
    """Decode bytes (or numpy bytes) to ``str``, leaving other values alone."""
    if isinstance(value, bytes | np.bytes_):
        return bytes(value).decode("utf-8", errors="replace").strip("\x00").strip()
    if isinstance(value, np.str_):
        return str(value)
    return value


def clean(value: Any) -> Any:
    """Decode a value and map placeholders to ``None``."""
    value = decode_str(value)
    if is_placeholder(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _decode_compound(value: np.ndarray) -> Any:
    names: tuple[str, ...] = value.dtype.names or ()
    record = value.reshape(-1)[0] if value.ndim else value
    for field in _VALUE_FIELDS:
        if field in names:
            return clean(record[field])
    # Unknown compound layout: (description, unit, value) is positional
    if len(names) >= 3:
        return clean(record[names[2]])
    return None


def decode_attr(value: Any) -> Any:
    """Decode a raw HDF5 attribute value into a Python value.

    Args:
        value (Any): Raw attribute value as returned by h5py.

    Returns:
        Any: ``str``, number, list of values, or ``None`` if missing/placeholder.
    """
    if isinstance(value, h5py.Reference):
        return None
    if isinstance(value, np.ndarray):
        if value.dtype.names:
            return _decode_compound(value)
        if value.dtype == object or value.dtype.kind in "SU":
            items = [clean(v) for v in value.reshape(-1)]
            if value.size == 1:
                return items[0]
            return items
        if value.size == 1:
            return value.reshape(-1)[0].item()
        return value
    if isinstance(value, np.void) and value.dtype.names:
        return _decode_compound(np.asarray(value))
    return clean(value)


def attr_unit(value: Any) -> str | None:
    """Return the unit stored in a compound SDHDF attribute, if any."""
    if (
        isinstance(value, np.ndarray)
        and value.dtype.names
        and "unit" in value.dtype.names
    ):
        return clean(value.reshape(-1)[0]["unit"])
    return None


def decode_attrs(attrs: h5py.AttributeManager | dict) -> dict[str, Any]:
    """Decode all attributes of an HDF5 object, dropping damaged entries.

    Args:
        attrs (h5py.AttributeManager | dict): Attributes to decode.

    Returns:
        dict[str, Any]: Decoded attributes. Missing/placeholder values are
            ``None``. HDF5 dimension-scale bookkeeping is dropped.
    """
    out: dict[str, Any] = {}
    for key in attrs:
        if key in ("DIMENSION_LIST", "REFERENCE_LIST") or is_placeholder(key):
            continue
        try:
            raw = attrs[key]
        except (OSError, TypeError, KeyError):  # unreadable/damaged attribute
            out[key] = None
            continue
        out[key] = decode_attr(raw)
    return out


def get_str_list(attrs: dict[str, Any], key: str) -> list[str] | None:
    """Return ``attrs[key]`` as a list of strings, or None if unusable.

    Accepts a list of strings, or a single comma-separated string. Returns
    None if any entry is missing or a placeholder.
    """
    value = attrs.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        value = [v.strip() for v in value.split(",")]
    if not isinstance(value, list | tuple):
        return None
    if any(is_placeholder(v) for v in value):
        return None
    return [str(v) for v in value]
