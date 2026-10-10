"""Version-independent description of the SDHDF layout.

Every SDHDF version stores the same information under different names.
Rather than branching on the version number, the reader *discovers* each
object by probing a list of known names (newest first), and renames table
columns to their SDHDF 4.0 names. The version number recorded in the file is
used for information only, since files written by some tools record a damaged
or placeholder version.

Canonical names throughout the package are the SDHDF 4.0 names.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import h5py
import numpy as np

from sdhdf._attrs import clean, decode_attr

#: Latest SDHDF definition version written by this package.
LATEST_VERSION = "4.0"

BEAM_RE = re.compile(r"^beam_\d+$")
BAND_RE = re.compile(r"^band_\w+$")

# ---------------------------------------------------------------------------
# Object names. Each entry maps a canonical role to candidate names (newest
# first). Paths are relative to the parent group given in the comment.
# ---------------------------------------------------------------------------

#: Relative to the file root.
ROOT_GROUPS: dict[str, tuple[str, ...]] = {
    "metadata": ("metadata",),
    "configuration": ("configuration", "config"),
}

#: Relative to ``/metadata``.
FILE_TABLES: dict[str, tuple[str, ...]] = {
    "primary_header": ("primary_header",),
    "beam_parameters": ("beam_parameters", "beam_params"),
    "history": ("history",),
    "schedule": ("schedule",),
    "software_versions": ("software_versions",),
}

#: Relative to ``/configuration`` (or ``/config``).
CONFIG_TABLES: dict[str, tuple[str, ...]] = {
    "instrument_configuration": ("instrument_configuration", "backend_config"),
    "calibrator_instrument_configuration": (
        "calibrator_instrument_configuration",
        "cal_backend_config",
    ),
    "receiver_configuration": ("receiver_configuration",),
    "telescope_configuration": ("telescope_configuration",),
}

#: Relative to ``/beam_N/metadata``.
BEAM_TABLES: dict[str, tuple[str, ...]] = {
    "band_parameters": ("band_parameters", "band_params"),
    "calibrator_band_parameters": ("calibrator_band_parameters", "cal_band_params"),
}

#: Relative to ``/beam_N/band_X/metadata``.
BAND_TABLES: dict[str, tuple[str, ...]] = {
    "observation_parameters": ("observation_parameters", "obs_params"),
    "calibrator_observation_parameters": (
        "calibrator_observation_parameters",
        "cal_obs_params",
    ),
}

#: Relative to ``/beam_N/band_X/astronomy_data``.
ASTRONOMY_DATASETS: dict[str, tuple[str, ...]] = {
    "data": ("data",),
    "frequency": ("frequency",),
    "flags": ("flags", "data_flags"),
    "weights": ("weights", "data_weights"),
}

#: Relative to ``/beam_N/band_X/calibrator_data``.
CALIBRATOR_DATASETS: dict[str, tuple[str, ...]] = {
    "on": ("calibrator_data_on", "cal_on", "cal_data_on"),
    "off": ("calibrator_data_off", "cal_off", "cal_data_off"),
    "binned": ("calibrator_data_binned", "cal_binned", "cal32_data"),
    # The 4.0 definition document describes a single ``data`` dataset
    "data": ("data",),
    "frequency": ("frequency", "cal_frequency"),
    "flags": ("flags", "cal_data_flags"),
    "weights": ("weights", "cal_data_weights"),
}

#: Canonical (4.0) path of each object, used when writing.
V4_PATHS = {
    "astronomy_data": "astronomy_data",
    "calibrator_data": "calibrator_data",
    "band_metadata": "metadata",
    "on": "calibrator_data_on",
    "off": "calibrator_data_off",
    "binned": "calibrator_data_binned",
}

# ---------------------------------------------------------------------------
# Column renames (legacy -> 4.0), per table kind.
# ---------------------------------------------------------------------------

_OBS_COLUMNS = {
    "DUMP_TIME": "INTEGRATION_TIME",
    "AEST": "LOCAL_TIME",
    "RA_STR": "RIGHT_ASCENSION",
    "DEC_STR": "DECLINATION",
    "FSTAT": "DRIVE_STATUS",
    "GL": "GALACTIC_LONGITUDE",
    "GB": "GALACTIC_LATITUDE",
    "AZ": "AZIMUTH_ANGLE",
    "ZE": "ZENITH_ANGLE",
    "EL": "ELEVATION_ANGLE",
    "PARA_ANGLE": "PARALLACTIC_ANGLE",
    "WIND_DIR": "WIND_DIRECTION",
    "WIND_SPD": "WIND_SPEED",
    "REL_HUMIDITY": "RELATIVE_HUMIDITY",
}

_BAND_COLUMNS = {
    "CENTRE_FREQ": "CENTRE_FREQUENCY",
    "LOW_FREQ": "LOW_FREQUENCY",
    "HIGH_FREQ": "HIGH_FREQUENCY",
    "N_CHANS": "NUMBER_OF_CHANNELS",
    "N_POLS": "NUMBER_OF_POLARISATIONS",
    "POL_TYPE": "POLARISATION_TYPE",
    "DUMP_TIME": "REQUESTED_INTEGRATION_TIME",
    "N_DUMPS": "NUMBER_OF_INTEGRATIONS",
    "PARTIAL_N_DUMPS": "PARTIAL_NUMBER_OF_INTEGRATIONS",
    "N_BINS": "NUMBER_OF_BINS",
}

_BEAM_COLUMNS = {
    "N_BANDS": "NUMBER_OF_BANDS",
    "RA": "RIGHT_ASCENSION",
    "DEC": "DECLINATION",
}

_PRIMARY_COLUMNS = {
    "HDR_DEFN": "HEADER_DEFINITION",
    "HDR_DEFN_VERSION": "HEADER_DEFINITION_VERSION",
    "SCHED_BLOCK_ID": "SCHEDULE_ID",
    "CAL_MODE": "CALIBRATION_MODE",
    "OBS_TYPE": "OBSERVATION_TYPE",
    "PID": "PROJECT_ID",
    "N_BEAMS": "NUMBER_OF_BEAMS",
}

_HISTORY_COLUMNS = {
    "PROC": "PROCESS",
    "PROC_DESCR": "PROCESS_DESCRIPTION",
    "PROC_ARGS": "PROCESS_ARGUMENTS",
    "PROC_HOST": "PROCESSING_HOST",
    "PROC_LOG": "PROCESS_LOG",
}

_SOFTWARE_COLUMNS = {
    "PROC": "PROCESS",
    "SOFTWARE_DESCR": "SOFTWARE_DESCRIPTION",
}

_SCHEDULE_COLUMNS = {
    "SCHED_HOST": "SCHEDULE_HOST",
    "SCHED_VERSION": "SCHEDULE_VERSION",
    "SCHED_BLOCK": "SCHEDULE_BLOCK",
}

COLUMN_RENAMES: dict[str, dict[str, str]] = {
    "observation_parameters": _OBS_COLUMNS,
    "calibrator_observation_parameters": _OBS_COLUMNS,
    "band_parameters": _BAND_COLUMNS,
    "calibrator_band_parameters": _BAND_COLUMNS,
    "beam_parameters": _BEAM_COLUMNS,
    "primary_header": _PRIMARY_COLUMNS,
    "history": _HISTORY_COLUMNS,
    "software_versions": _SOFTWARE_COLUMNS,
    "schedule": _SCHEDULE_COLUMNS,
}

#: Canonical dimension names, in SDHDF order, for data-like arrays.
DATA_DIMS = ("time", "polarisation", "frequency", "bin")

_DIM_ALIASES = {
    "polarization": "polarisation",
    "pol": "polarisation",
    "product": "polarisation",
    "none": "polarisation",  # 4.0 template labels the product axis 'None'
    "channel": "frequency",
    "freq": "frequency",
    "phase_bin": "bin",
    "integration": "time",
    "dump": "time",
}


def canonical_dim(label: str) -> str:
    """Return the canonical name of a dimension label."""
    label = str(label).strip().lower()
    return _DIM_ALIASES.get(label, label)


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
    obj = group.get(name) if group is not None else None
    return obj if isinstance(obj, h5py.Group) else None


def beam_labels(h5: h5py.File) -> list[str]:
    """Labels of the beam groups in a file, in numerical order."""
    labels = [k for k in h5 if BEAM_RE.match(k) and isinstance(h5[k], h5py.Group)]
    return sorted(labels, key=lambda s: int(s.split("_")[1]))


def band_labels(beam: h5py.Group) -> list[str]:
    """Labels of the band groups in a beam, in file order."""
    return [k for k in beam if BAND_RE.match(k) and isinstance(beam[k], h5py.Group)]


# ---------------------------------------------------------------------------
# Version detection
# ---------------------------------------------------------------------------


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


def _recorded_version(h5: h5py.File) -> str | None:
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


def _inferred_version(h5: h5py.File) -> str:
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
    return VersionInfo(recorded=_recorded_version(h5), inferred=_inferred_version(h5))


# ---------------------------------------------------------------------------
# Polarisation products
# ---------------------------------------------------------------------------

_POL_TYPES = {
    "AABBCRCI": ["AA", "BB", "CR", "CI"],
    "AABB": ["AA", "BB"],
    "AA+BB": ["AA+BB"],
    "IQUV": ["I", "Q", "U", "V"],
    "XXYY": ["XX", "YY"],
    "XXYYXYYX": ["XX", "YY", "XY", "YX"],
    "RRLL": ["RR", "LL"],
    "RRLLRLLR": ["RR", "LL", "RL", "LR"],
}


def polarisation_labels(pol_type: str | None, npol: int) -> np.ndarray:
    """Labels for the polarisation axis given the POLARISATION_TYPE."""
    if pol_type is not None:
        key = re.sub(r"[\s,]", "", str(pol_type)).upper()
        labels = _POL_TYPES.get(key)
        if labels is None and len(key) == npol:
            labels = list(key)  # e.g. "I", "IQUV"-like single-letter codes
        if labels is not None and len(labels) == npol:
            return np.array(labels)
    if npol == 4:
        return np.array(_POL_TYPES["AABBCRCI"])
    if npol == 2:
        return np.array(_POL_TYPES["AABB"])
    if npol == 1:
        return np.array(_POL_TYPES["AA+BB"])
    return np.array([f"P{i}" for i in range(npol)])
