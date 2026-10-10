"""Constants describing the SDHDF format.

Every SDHDF version stores the same information under different names. The
name tables below map each canonical (SDHDF 4.0) role to its candidate names,
newest first, so that objects can be found by probing rather than by
branching on the recorded version number.
"""

from __future__ import annotations

import numpy as np

#: Latest SDHDF definition version written by this package.
LATEST_VERSION = "4.0"

#: Canonical dimension names, in SDHDF order, for data-like arrays.
DATA_DIMS = ("time", "polarisation", "frequency", "bin")

# ---------------------------------------------------------------------------
# Object names: canonical role -> candidate names (newest first)
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

#: Canonical (4.0) name of each object, used when writing.
V4_PATHS = {
    "astronomy_data": "astronomy_data",
    "calibrator_data": "calibrator_data",
    "band_metadata": "metadata",
    "on": "calibrator_data_on",
    "off": "calibrator_data_off",
    "binned": "calibrator_data_binned",
}

# ---------------------------------------------------------------------------
# Table columns
# ---------------------------------------------------------------------------

#: Legacy -> 4.0 column names, per table.
COLUMN_RENAMES: dict[str, dict[str, str]] = {
    "observation_parameters": {
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
    },
    "band_parameters": {
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
    },
    "beam_parameters": {
        "N_BANDS": "NUMBER_OF_BANDS",
        "RA": "RIGHT_ASCENSION",
        "DEC": "DECLINATION",
    },
    "primary_header": {
        "HDR_DEFN": "HEADER_DEFINITION",
        "HDR_DEFN_VERSION": "HEADER_DEFINITION_VERSION",
        "SCHED_BLOCK_ID": "SCHEDULE_ID",
        "CAL_MODE": "CALIBRATION_MODE",
        "OBS_TYPE": "OBSERVATION_TYPE",
        "PID": "PROJECT_ID",
        "N_BEAMS": "NUMBER_OF_BEAMS",
    },
    "history": {
        "PROC": "PROCESS",
        "PROC_DESCR": "PROCESS_DESCRIPTION",
        "PROC_ARGS": "PROCESS_ARGUMENTS",
        "PROC_HOST": "PROCESSING_HOST",
        "PROC_LOG": "PROCESS_LOG",
    },
    "software_versions": {
        "PROC": "PROCESS",
        "SOFTWARE_DESCR": "SOFTWARE_DESCRIPTION",
    },
    "schedule": {
        "SCHED_HOST": "SCHEDULE_HOST",
        "SCHED_VERSION": "SCHEDULE_VERSION",
        "SCHED_BLOCK": "SCHEDULE_BLOCK",
    },
}
COLUMN_RENAMES["calibrator_observation_parameters"] = COLUMN_RENAMES[
    "observation_parameters"
]
COLUMN_RENAMES["calibrator_band_parameters"] = COLUMN_RENAMES["band_parameters"]

#: Columns of observation_parameters that readers (e.g. INSPECTA) require,
#: with the fill value used when a file does not record them.
REQUIRED_OBS_COLUMNS: dict[str, float | str] = {
    "ELAPSED_TIME": np.nan,
    "INTEGRATION_TIME": np.nan,
    "MJD": 0,
    "FRACTIONAL_MJD": np.nan,
    "UTC": "None",
    "LOCAL_TIME": "None",
    "RIGHT_ASCENSION": "None",
    "DECLINATION": "None",
    "GALACTIC_LONGITUDE": np.nan,
    "GALACTIC_LATITUDE": np.nan,
    "AZIMUTH_ANGLE": np.nan,
    "ZENITH_ANGLE": np.nan,
    "ELEVATION_ANGLE": np.nan,
    "HOUR_ANGLE": np.nan,
    "PARALLACTIC_ANGLE": np.nan,
    "WIND_DIRECTION": np.nan,
    "WIND_SPEED": np.nan,
}

#: Columns of the persistent RFI tables.
RFI_COLUMNS = [
    "type",
    "observatory label",
    "receiver label",
    "freq0 MHz",
    "freq1 MHz",
    "MJD0",
    "MJD1",
    "text string for label",
]
