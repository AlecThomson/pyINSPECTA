"""Writing of SDHDF files in the latest definition (4.0).

Whatever version a file was read from, it is written in the SDHDF 4.0
layout: 4.0 group, dataset and column names, compound
``(description, unit, value)`` attributes, 2-D frequency axes and 4-D flags
and weights (size-1 axes where they apply to all polarisations or bins).
Metadata tables that describe the data shape (band parameters, observation
parameters) are rebuilt from the data, so processing such as averaging is
reflected in them.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import dask.array as da
import h5py
import numpy as np
import pandas as pd
import xarray as xr

from sdhdf import history
from sdhdf._version import version as __version__
from sdhdf.constants import DATA_DIMS, LATEST_VERSION
from sdhdf.definition import definition
from sdhdf.logger import logger

if TYPE_CHECKING:
    from sdhdf.sdhdf import SDHDF, Beam, SubBand


def compound_attr(
    value: Any, description: str = "", unit: str | None = None
) -> np.ndarray:
    """Build an SDHDF 4.0 compound ``(description, unit, value)`` attribute."""
    fields = [
        str(description).encode(),
        str(unit if unit is not None else "None").encode(),
        str(value if value is not None else "None").encode(),
    ]
    dtype = [
        (name, f"S{max(1, len(v))}")
        for name, v in zip(("description", "unit", "value"), fields, strict=True)
    ]
    return np.array([tuple(fields)], dtype=dtype)


def _set_attrs(
    obj: h5py.HLObject, attrs: dict[str, Any], units: dict[str, str] | None = None
) -> None:
    units = units or {}
    for key, value in attrs.items():
        text = (
            ",".join(str(v) for v in value)
            if isinstance(value, list | tuple | np.ndarray)
            else value
        )
        obj.attrs[key] = compound_attr(text, description=key, unit=units.get(key))


def _set_class(obj: h5py.HLObject, sdhdf_class: str, description: str) -> None:
    obj.attrs["SDHDF_CLASS"] = compound_attr(sdhdf_class, "SDHDF class name")
    obj.attrs["SDHDF_DESCRIPTION"] = compound_attr(
        description, "SDHDF object description"
    )


def to_records(frame: pd.DataFrame) -> np.ndarray:
    """Convert a DataFrame to a numpy structured array for HDF5."""
    if frame.shape[1] == 0:
        return np.zeros(len(frame), dtype=[("EMPTY", "S1")])
    columns, dtypes = [], []
    for col in frame.columns:
        values = frame[col].to_numpy()
        if values.dtype.kind in "biuf":
            arr = values
        elif values.dtype.kind == "M":
            arr = np.array([str(v).encode() for v in values])
        elif all(
            isinstance(v, int | float | np.number | bool | np.bool_) for v in values
        ) and len(values):
            arr = np.asarray(values.tolist())
        else:
            arr = np.array(
                [
                    (
                        "None"
                        if v is None or (isinstance(v, float) and np.isnan(v))
                        else str(v)
                    ).encode()
                    for v in values
                ]
            )
            if arr.dtype.itemsize == 0 or len(arr) == 0:
                arr = arr.astype("S1")
        columns.append(arr)
        dtypes.append((str(col), arr.dtype))
    records = np.zeros(len(frame), dtype=dtypes)
    for (name, _), arr in zip(dtypes, columns, strict=True):
        records[name] = arr
    return records


def write_table(
    group: h5py.Group,
    name: str,
    frame: pd.DataFrame,
    units: dict[str, str] | None = None,
    attrs: dict[str, Any] | None = None,
) -> None:
    """Write a metadata table as an SDHDF 4.0 compound dataset."""
    dset = group.create_dataset(name, data=to_records(frame))
    _set_class(dset, "sdhdf_table", definition().table_descriptions.get(name, name))
    units = units or {}
    for col in frame.columns:
        dset.attrs[str(col)] = compound_attr(
            "None", description=str(col), unit=units.get(col)
        )
    for key, value in (attrs or {}).items():
        if key not in dset.attrs and not key.startswith("SDHDF_"):
            dset.attrs[key] = compound_attr(value, description=key)


def write_array(group: h5py.Group, name: str, arr: xr.DataArray, dtype) -> h5py.Dataset:
    """Write a DataArray to a new dataset, streaming dask arrays chunk by chunk."""
    data = arr.data
    dset = group.create_dataset(
        name, shape=arr.shape, dtype=dtype, chunks=True if arr.size else None
    )
    if isinstance(data, da.Array):
        da.store(data.astype(dtype), dset, lock=True, scheduler="threads")
    else:
        dset[...] = np.asarray(data, dtype=dtype)
    dset.attrs["DIMENSION_LABELS"] = np.array(list(arr.dims), dtype=h5py.string_dtype())
    return dset


def _to_4d(arr: xr.DataArray, rename_bin: str | None = None) -> xr.DataArray:
    """Order an array as (time, polarisation, frequency, bin), adding size-1 axes."""
    if rename_bin and rename_bin in arr.dims:
        arr = arr.rename({rename_bin: "bin"})
    missing = [d for d in DATA_DIMS if d not in arr.dims]
    if missing:
        arr = arr.expand_dims(missing)
    return arr.transpose(*DATA_DIMS)


def observation_parameters_table(ds: xr.Dataset) -> pd.DataFrame:
    """Rebuild an observation_parameters table from the time coordinates.

    Every required column (see ``definition.toml``) is present, filled with a
    placeholder where the data do not record it.
    """
    n_time = ds.sizes["time"]
    columns = {
        name: np.asarray(coord.to_numpy())
        for name, coord in ds.coords.items()
        if coord.dims == ("time",) and name != "time"
    }
    if "MJD" not in columns:
        mjd = ds["time"].to_numpy().astype(float)
        columns["MJD"] = np.floor(mjd).astype(np.int64)
        columns["FRACTIONAL_MJD"] = mjd - np.floor(mjd)
    required = definition().required_columns["observation_parameters"]
    for name, fill in required.items():
        if name not in columns:
            columns[name] = np.full(
                n_time, fill, dtype=object if isinstance(fill, str) else float
            )
    ordered = [c for c in required if c in columns]
    return pd.DataFrame(columns)[ordered + [c for c in columns if c not in ordered]]


def write_data_group(
    group: h5py.Group,
    ds: xr.Dataset,
    variables: dict[str, str],
    description: str,
) -> None:
    """Write an astronomy or calibrator data group (data, frequency, flags, weights)."""
    _set_class(group, "sdhdf_data", description)
    for var, path in variables.items():
        if var not in ds:
            continue
        arr = _to_4d(ds[var], rename_bin="cal_bin" if var == "binned" else None)
        dset = write_array(group, path, arr, np.float32)
        _set_class(dset, "sdhdf_waterfall", f"{description} ({var})")
        attrs = {
            k: v
            for k, v in ds[var].attrs.items()
            if k not in ("units", "SDHDF_CLASS", "SDHDF_DESCRIPTION")
        }
        attrs["UNIT"] = ds[var].attrs.get("units") or attrs.get("UNIT") or "counts"
        _set_attrs(dset, attrs)

    freq = (
        ds["frequency_per_time"]
        if "frequency_per_time" in ds.coords
        else ds["frequency"].expand_dims(time=1)
    )
    fdset = group.create_dataset(
        "frequency", data=np.asarray(freq.to_numpy(), dtype=np.float64)
    )
    fdset.attrs["DIMENSION_LABELS"] = np.array(
        ["time", "frequency"], dtype=h5py.string_dtype()
    )
    _set_class(fdset, "sdhdf_frequency", "Channel centre frequencies")
    _set_attrs(
        fdset,
        {
            "UNIT": ds["frequency"].attrs.get("units") or "MHz",
            "FRAME": ds["frequency"].attrs.get("FRAME") or "topocentric",
        },
    )

    for var in ("flags", "weights"):
        extra = [d for d in ds[var].dims if d not in ("time", "frequency")]
        if extra:
            logger.warning(
                f"Writing {var} with dimensions {extra}; INSPECTA can only read "
                f"{var} with one value per integration and channel."
            )
    flags = write_array(group, "flags", _to_4d(ds["flags"]), np.uint8)
    _set_class(flags, "sdhdf_flags", "Flags dataset")
    _set_attrs(flags, {"FLAG_TYPE": "0=not flagged, 1=flagged"})
    weights = write_array(group, "weights", _to_4d(ds["weights"]), np.float32)
    _set_class(weights, "sdhdf_weights", "Weights dataset")
    _set_attrs(
        weights,
        {
            "UNIT": "dimensionless",
            "WEIGHT_TYPE": ds["weights"].attrs.get("WEIGHT_TYPE")
            or "Product of the channel bandwidth (Hz) and integration time (s)",
        },
    )


def band_parameters_row(
    label: str, ds: xr.Dataset, original: dict[str, Any]
) -> dict[str, Any]:
    """This band's band_parameters row, updated to describe the data as written."""
    freq = np.asarray(ds["frequency"].to_numpy(), dtype=float)
    pols = [str(p) for p in ds["polarisation"].to_numpy()]
    row = dict(original)
    row.update(
        {
            "LABEL": label,
            "CENTRE_FREQUENCY": float(original.get("CENTRE_FREQUENCY", np.nan))
            if "CENTRE_FREQUENCY" in original
            else float(np.mean(freq)),
            "LOW_FREQUENCY": float(np.min(freq)),
            "HIGH_FREQUENCY": float(np.max(freq)),
            "NUMBER_OF_CHANNELS": int(ds.sizes["frequency"]),
            "NUMBER_OF_POLARISATIONS": len(pols),
            "POLARISATION_TYPE": original.get("POLARISATION_TYPE") or "".join(pols),
            "NUMBER_OF_INTEGRATIONS": int(ds.sizes["time"]),
            "NUMBER_OF_BINS": int(ds.sizes.get("bin", ds.sizes.get("cal_bin", 1))),
        }
    )
    row.setdefault("REQUESTED_INTEGRATION_TIME", np.nan)
    row.setdefault("PARTIAL_NUMBER_OF_INTEGRATIONS", 0)
    return row


def write_beam(h5: h5py.File, beam: Beam) -> None:
    """Write a beam group: its bands and band_parameters tables."""
    group = h5.create_group(beam.label)
    _set_class(
        group,
        "sdhdf_beam",
        "SDHDF group containing data products specific to the antenna beam",
    )
    band_rows, cal_rows = [], []
    for sb in beam.subbands:
        write_band(group, sb)
        band_rows.append(
            band_parameters_row(sb.label, sb.astronomy_dataset, sb.band_parameters)
        )
        if sb.calibrator_dataset is not None:
            cal_rows.append(
                band_parameters_row(
                    sb.label, sb.calibrator_dataset, sb.calibrator_band_parameters
                )
            )
    meta = group.create_group("metadata")
    _set_class(meta, "sdhdf_meta", "SDHDF group containing observation metadata")
    band_units = beam.band_parameters.units if beam.band_parameters is not None else {}
    write_table(meta, "band_parameters", pd.DataFrame(band_rows), units=band_units)
    if cal_rows:
        write_table(
            meta, "calibrator_band_parameters", pd.DataFrame(cal_rows), units=band_units
        )


def write_band(beam_group: h5py.Group, sb: SubBand) -> None:
    """Write a band group: astronomy data, calibrator data and observation metadata."""
    group = beam_group.create_group(sb.label)
    _set_class(
        group,
        "sdhdf_band",
        "SDHDF group containing data products specific to the frequency band",
    )
    astro = group.create_group("astronomy_data")
    write_data_group(astro, sb.astronomy_dataset, {"data": "data"}, "Astronomy data")
    meta = group.create_group("metadata")
    _set_class(meta, "sdhdf_meta", "SDHDF group containing observation metadata")
    obs = sb.astronomy_dataset
    write_table(
        meta,
        "observation_parameters",
        observation_parameters_table(obs),
        units=_coord_units(obs),
    )
    if sb.calibrator_dataset is not None:
        cal = group.create_group("calibrator_data")
        write_data_group(
            cal,
            sb.calibrator_dataset,
            {
                role: definition().calibrator_datasets[role][0]
                for role in ("on", "off", "binned")
            }
            | {"data": "data"},
            "Calibrator data",
        )
        write_table(
            meta,
            "calibrator_observation_parameters",
            observation_parameters_table(sb.calibrator_dataset),
            units=_coord_units(sb.calibrator_dataset),
        )


def _coord_units(ds: xr.Dataset) -> dict[str, str]:
    return {
        name: c.attrs["units"] for name, c in ds.coords.items() if "units" in c.attrs
    }


def write_sdhdf(sdhdf: SDHDF, filename: str | Path, overwrite: bool = False) -> None:
    """Write an :class:`~sdhdf.SDHDF` object as an SDHDF 4.0 file.

    Args:
        sdhdf (SDHDF): The object to write.
        filename (str | Path): Output file.
        overwrite (bool, optional): Overwrite an existing file. Defaults to False.

    Raises:
        FileExistsError: If the file exists and ``overwrite`` is False.
        ValueError: If asked to overwrite the file the data are read from.
    """
    filename = Path(filename)
    if filename.exists():
        if not overwrite:
            msg = f"File '{filename}' already exists"
            raise FileExistsError(msg)
        if filename.resolve() == sdhdf.filename.resolve():
            msg = "Cannot overwrite the file being read; write to a new file instead."
            raise ValueError(msg)

    tables = dict(sdhdf.metadata.tables)
    hist = history.append(
        tables["history"].table,
        history.generate_history_row(
            "sdhdf.write",
            f"Written as SDHDF {LATEST_VERSION} by sdhdf (pyINSPECTA) {__version__}",
            {"filename": str(filename), "source_version": sdhdf.metadata.version},
        ),
    )

    tmp = filename.with_name(filename.name + ".partial")
    with h5py.File(tmp, "w") as h5:
        _set_class(h5, "sdhdf_file", "SDHDF format file")
        meta = h5.create_group("metadata")
        _set_class(meta, "sdhdf_meta", "SDHDF group containing observation metadata")
        for name, table in tables.items():
            if name in ("history",) or name.endswith("configuration"):
                continue
            frame = table.table.copy()
            if name == "primary_header" and len(frame):
                frame["HEADER_DEFINITION_VERSION"] = LATEST_VERSION
            if name == "software_versions":
                frame = pd.concat(
                    [
                        frame,
                        pd.DataFrame(
                            {
                                "PROCESS": ["sdhdf.write"],
                                "SOFTWARE": ["sdhdf"],
                                "SOFTWARE_DESCRIPTION": ["pyINSPECTA"],
                                "SOFTWARE_VERSION": [__version__],
                            }
                        ),
                    ],
                    ignore_index=True,
                )
            write_table(meta, name, frame, units=table.units)
        write_table(meta, "history", hist)

        config = h5.create_group("configuration")
        _set_class(
            config,
            "sdhdf_configuration",
            "SDHDF group containing configuration parameters",
        )
        for name, table in tables.items():
            if name.endswith("configuration"):
                write_table(config, name, table.table, units=table.units)

        for beam in sdhdf.beams:
            write_beam(h5, beam)
    tmp.replace(filename)
    sdhdf.metadata.history.table = hist
    logger.info(f"Wrote '{filename}' (SDHDF {LATEST_VERSION})")
