"""Decoding of SDHDF HDF5 attributes and cell values.

Placeholders written by older INSPECTA versions (e.g. ``"NOT FOUND"``,
``"FIX ME 2"``) are treated as missing.
"""

from __future__ import annotations

import re
from typing import Any

import h5py
import numpy as np

from sdhdf.definition import definition


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
    if not isinstance(value, str):
        return False
    placeholder = r"""^\s*(
        not[\s_]*(set|found|defined|available|applicable)
        | unknown | uncertain | undefined
        | null | none | nan | n/?a
        | fix[\s_]*me.*
        | string(\s*\(.*\))?
    )\s*$"""
    return value.strip() == "" or bool(
        re.match(placeholder, value, re.IGNORECASE | re.VERBOSE)
    )


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
    for field in ("value", "default"):  # 4.0 files, 4.0 templates
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
        unit = clean(value.reshape(-1)[0]["unit"])
        return None if unit is None else str(unit)
    return None


def key_from_description(text: str) -> str | None:
    """Return the attribute key whose SDHDF 4.0 description starts ``text``.

    Some versions of INSPECTA write attributes keyed by their description.
    """
    lowered = text.strip().lower()
    for prefix, key in definition().attribute_descriptions.items():
        if lowered.startswith(prefix):
            return key
    return None


def _attr_description(key: str, raw: Any) -> str:
    """Description of an attribute: its compound field, or the key itself."""
    if (
        isinstance(raw, np.ndarray)
        and raw.dtype.names
        and "description" in raw.dtype.names
    ):
        return str(decode_str(raw.reshape(-1)[0]["description"]) or "")
    return key


def decode_attrs(attrs: h5py.AttributeManager | dict[str, Any]) -> dict[str, Any]:
    """Decode all attributes of an HDF5 object, repairing damaged entries.

    Attributes keyed by their SDHDF 4.0 description (as written by some
    versions of INSPECTA) are renamed to their proper keys, and
    ``DIMENSION_LABELS`` is reconstructed from the per-dimension attributes
    (``TIME``, ``PRODUCT_TYPE``, ``FREQUENCY``, ``PHASE_BIN``) when it is
    missing or unusable.

    Args:
        attrs (h5py.AttributeManager | dict): Attributes to decode.

    Returns:
        dict[str, Any]: Decoded attributes. Missing/placeholder values are
            ``None``. HDF5 dimension-scale bookkeeping is dropped.
    """
    out: dict[str, Any] = {}
    renamed: dict[str, Any] = {}
    dimensions: dict[int, Any] = {}
    for key in attrs:
        if key in ("DIMENSION_LIST", "REFERENCE_LIST") or is_placeholder(key):
            continue
        try:
            raw = attrs[key]
        except (OSError, TypeError, KeyError):  # unreadable/damaged attribute
            out[key] = None
            continue
        value = decode_attr(raw)
        if match := re.search(
            r"data array dimension\s*(\d+)", _attr_description(key, raw), re.IGNORECASE
        ):
            dimensions[int(match.group(1))] = value
        canonical = key_from_description(key) if " " in key else None
        if canonical is not None:
            renamed[canonical] = value
        else:
            out[key] = value
    for key, value in renamed.items():
        out.setdefault(key, value)
    labels = [dimensions.get(i) for i in range(len(dimensions))]
    if (
        get_str_list(out, "DIMENSION_LABELS") is None
        and labels
        and not any(is_placeholder(v) for v in labels)
    ):
        out["DIMENSION_LABELS"] = [str(v) for v in labels]
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
