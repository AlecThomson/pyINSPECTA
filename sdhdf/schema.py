"""Version-independent discovery of the SDHDF layout.

Objects are found by trying their candidate names, since some files record a
damaged or placeholder version number.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import h5py
import numpy as np

from sdhdf.attributes import clean, decode_attr
from sdhdf.definition import definition


def canonical_dim(label: str) -> str:
    """Return the canonical name of a dimension label."""
    label = str(label).strip().lower()
    return definition().dimension_aliases.get(label, label)


def find_child(
    group: h5py.Group | None, candidates: tuple[str, ...]
) -> h5py.Dataset | h5py.Group | None:
    """Return the first child of ``group`` whose name is in ``candidates``."""
    if group is None:
        return None
    for name in candidates:
        if name in group:
            return group[name]
    return None


def find_group(group: h5py.Group | None, name: str) -> h5py.Group | None:
    """Return the child group ``name`` of ``group``, or None."""
    obj = group.get(name) if group is not None else None
    return obj if isinstance(obj, h5py.Group) else None


def beam_labels(h5: h5py.File) -> list[str]:
    """Labels of the beam groups in a file, in numerical order."""
    labels = [
        k for k in h5 if re.match(r"^beam_\d+$", k) and isinstance(h5[k], h5py.Group)
    ]
    return sorted(labels, key=lambda s: int(s.split("_")[1]))


def band_labels(beam: h5py.Group) -> list[str]:
    """Labels of the band groups in a beam, in file order."""
    return [
        k
        for k in beam
        if re.match(r"^band_\w+$", k) and isinstance(beam[k], h5py.Group)
    ]


@dataclass(frozen=True)
class VersionInfo:
    """SDHDF definition version of a file.

    Attributes:
        recorded (str | None): Version string recorded in the file, if usable.
        inferred (str): Version inferred from the file structure.
    """

    recorded: str | None
    inferred: str

    @property
    def version(self) -> str:
        """Best estimate of the version: the recorded one if usable."""
        return self.recorded or self.inferred

    @property
    def major_minor(self) -> tuple[int, int]:
        """Version as a (major, minor) tuple."""
        parts = re.findall(r"\d+", self.version)
        major = int(parts[0]) if parts else 0
        minor = int(parts[1]) if len(parts) > 1 else 0
        return major, minor


def recorded_version(h5: h5py.File) -> str | None:
    """The definition version recorded in the primary header, if usable."""
    ph = h5.get("metadata/primary_header")
    if not isinstance(ph, h5py.Dataset):
        return None
    for col in ("HEADER_DEFINITION_VERSION", "HDR_DEFN_VERSION"):
        if ph.dtype.names and col in ph.dtype.names and ph.shape and ph.shape[0] > 0:
            value = clean(ph[0][col])
            if value is not None and re.match(r"^\d+(\.\d+)*$", str(value)):
                return str(value)
        if col in ph.attrs:
            value = decode_attr(ph.attrs[col])
            if value is not None and re.match(r"^\d+(\.\d+)*$", str(value)):
                return str(value)
    return None


def inferred_version(h5: h5py.File) -> str:
    """The definition version implied by the structure of the file."""
    beams = beam_labels(h5)
    if "configuration" in h5:
        return "4.0"
    if not beams:
        return "unknown"
    bands = band_labels(h5[beams[0]])
    if not bands:
        return "unknown"
    band = h5[beams[0]][bands[0]]
    astro = band.get("astronomy_data")
    if astro is None or "data" not in astro:
        return "unknown"
    if "observation_parameters" in band.get("metadata", {}):
        return "4.0"
    if astro["data"].ndim == 5:
        return "2.0"
    if "frequency" in astro and astro["frequency"].ndim == 2:
        return "3.0"
    return "2.9"


def detect_version(h5: h5py.File) -> VersionInfo:
    """Detect the SDHDF definition version of an open file."""
    return VersionInfo(recorded=recorded_version(h5), inferred=inferred_version(h5))


def polarisation_labels(pol_type: str | None, npol: int) -> np.ndarray:
    """Labels for the polarisation axis given the POLARISATION_TYPE."""
    pol_types = definition().polarisation_types
    if pol_type is not None:
        key = re.sub(r"[\s,]", "", str(pol_type)).upper()
        labels = pol_types.get(key)
        if labels is None and len(key) == npol:
            labels = tuple(key)  # e.g. "I", "IQUV"-like single-letter codes
        if labels is not None and len(labels) == npol:
            return np.array(labels)
    defaults = definition().default_polarisation_types
    if npol in defaults:
        return np.array(pol_types[defaults[npol]])
    return np.array([f"P{i}" for i in range(npol)])
