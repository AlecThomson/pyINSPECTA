"""Core SDHDF module"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import dask
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from astropy.units import Quantity
from dask.diagnostics import ProgressBar
from matplotlib.axes import Axes

from sdhdf import averaging, flagging, history
from sdhdf.constants import LATEST_VERSION
from sdhdf.definition import definition
from sdhdf.logger import logger
from sdhdf.reader import read_band, table_row
from sdhdf.schema import (
    VersionInfo,
    band_labels,
    beam_labels,
    detect_version,
    find_child,
    find_group,
)
from sdhdf.tables import SDHDFTable

__all__ = ["SDHDF", "Beam", "MetaData", "SubBand", "upgrade"]


def _read_tables(
    group: h5py.Group | None, kinds: dict[str, tuple[str, ...]]
) -> dict[str, SDHDFTable]:
    tables = {}
    for kind, candidates in kinds.items():
        dset = find_child(group, candidates)
        if isinstance(dset, h5py.Dataset):
            tables[kind] = SDHDFTable.from_dataset(dset, kind)
    return tables


@dataclass
class MetaData:
    """File-level SDHDF metadata.

    Tables are available as attributes by their SDHDF 4.0 names, e.g.
    ``metadata.primary_header``.

    Attributes:
        version_info (VersionInfo): Recorded and inferred definition version.
        tables (dict[str, SDHDFTable]): Metadata and configuration tables.
    """

    version_info: VersionInfo
    tables: dict[str, SDHDFTable] = field(default_factory=dict)

    @classmethod
    def from_file(cls, h5: h5py.File) -> MetaData:
        """Read the file-level metadata of an open SDHDF file."""
        groups = definition().groups
        meta = find_child(h5, groups["metadata"])
        config = find_child(h5, groups["configuration"])
        tables = {
            **_read_tables(
                meta if isinstance(meta, h5py.Group) else None,
                definition().file_tables,
            ),
            **_read_tables(
                config if isinstance(config, h5py.Group) else None,
                definition().configuration_tables,
            ),
        }
        if "history" not in tables:
            tables["history"] = SDHDFTable(history.append(None), name="history")
        return cls(version_info=detect_version(h5), tables=tables)

    def __getattr__(self, name: str) -> SDHDFTable:
        tables = self.__dict__.get("tables", {})
        if name in tables:
            table: SDHDFTable = tables[name]
            return table
        msg = f"'{type(self).__name__}' has no table '{name}'"
        raise AttributeError(msg)

    def __dir__(self) -> list[str]:
        return [*super().__dir__(), *self.tables]

    @property
    def version(self) -> str:
        """SDHDF definition version of the file."""
        return self.version_info.version

    @property
    def telescope(self) -> str | None:
        return self._primary_header_value("TELESCOPE")

    @property
    def receiver(self) -> str | None:
        return self._primary_header_value("RECEIVER")

    def _primary_header_value(self, column: str) -> str | None:
        primary_header = self.tables.get("primary_header")
        value = primary_header.get(column) if primary_header is not None else None
        return None if value is None else str(value)

    def with_history(
        self, process: str, description: str, arguments: dict[str, Any]
    ) -> MetaData:
        """Copy of the metadata with a row added to the history table."""
        row = history.generate_history_row(process, description, arguments)
        table = SDHDFTable(
            history.append(self.tables["history"].table, row),
            name="history",
            attrs=self.tables["history"].attrs,
            units=self.tables["history"].units,
        )
        return replace(self, tables={**self.tables, "history": table})

    def print_tables(self, kinds: dict[str, tuple[str, ...]]) -> None:
        for kind in kinds:
            if kind in self.tables:
                print(kind)  # noqa: T201
                print(self.tables[kind].table.to_string())  # noqa: T201

    def print_obs_metadata(self) -> None:
        """Print the observation metadata tables."""
        self.print_tables(definition().file_tables)

    def print_obs_config(self) -> None:
        """Print the observation configuration tables."""
        self.print_tables(definition().configuration_tables)


@dataclass(repr=False)
class SubBand:
    """An SDHDF band of one beam.

    Attributes:
        label (str): Band label (e.g. ``"band_SB0"``).
        beam_label (str): Label of the parent beam.
        astronomy_dataset (xr.Dataset): Astronomy data, flags and weights.
        calibrator_dataset (xr.Dataset | None): Noise diode data, if recorded.
        band_parameters (dict): This band's row of ``band_parameters``.
        calibrator_band_parameters (dict): This band's row of
            ``calibrator_band_parameters``.
    """

    label: str
    beam_label: str
    astronomy_dataset: xr.Dataset
    calibrator_dataset: xr.Dataset | None = None
    band_parameters: dict[str, Any] = field(default_factory=dict)
    calibrator_band_parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(
        cls,
        h5: h5py.File,
        beam_label: str,
        label: str,
        band_parameters: dict[str, Any],
        calibrator_band_parameters: dict[str, Any],
    ) -> SubBand:
        """Read one band of an open SDHDF file, lazily."""
        astronomy, calibrator = read_band(
            h5,
            beam_label,
            label,
            band_row=band_parameters,
            calibrator_band_row=calibrator_band_parameters,
        )
        return cls(
            label=label,
            beam_label=beam_label,
            astronomy_dataset=astronomy,
            calibrator_dataset=calibrator,
            band_parameters=band_parameters,
            calibrator_band_parameters=calibrator_band_parameters,
        )

    def __repr__(self) -> str:
        sizes = dict(self.astronomy_dataset["data"].sizes)
        return f"SubBand(label={self.label!r}, beam={self.beam_label!r}, sizes={sizes})"

    def apply(
        self, func: Callable[..., xr.Dataset], calibrator: bool = True, **kwargs: Any
    ) -> SubBand:
        """Apply a Dataset function to the astronomy (and calibrator) data."""
        calibrator_dataset = self.calibrator_dataset
        if calibrator and calibrator_dataset is not None:
            calibrator_dataset = func(calibrator_dataset, **kwargs)
        return replace(
            self,
            astronomy_dataset=func(self.astronomy_dataset, **kwargs),
            calibrator_dataset=calibrator_dataset,
        )

    def plot_waterfall(
        self,
        polarisation: int | str = 0,
        flag: bool = False,
        **plot_kwargs: Any,
    ) -> Axes | None:
        """Waterfall plot of the data

        Args:
            polarisation (int | str, optional): Polarisation index or label.
                Defaults to 0.
            flag (bool, optional): Blank flagged data. Defaults to False.

        Returns:
            Axes | None: The plot axes, or None for a single integration.
        """
        ds = _select_polarisation(self.astronomy_dataset, polarisation).isel(bin=0)
        if ds.sizes["time"] == 1:
            logger.warning("Cannot create a waterfall plot with a single integration!")
            return None
        data = ds["data"].where(~ds["flags"]) if flag else ds["data"]
        data.plot(x="frequency", **plot_kwargs)
        ax = plt.gca()
        ax.set_title(f"Waterfall Plot (flagged = {flag!s})", fontsize=10)
        return ax

    def plot_spectrum(
        self,
        time: int = 0,
        polarisation: int | str = 0,
        flag: bool = False,
        **plot_kwargs: Any,
    ) -> Axes:
        """Plot a single spectrum

        Args:
            time (int, optional): Integration index. Defaults to 0.
            polarisation (int | str, optional): Polarisation index or label.
            flag (bool, optional): Blank flagged data. Defaults to False.

        Returns:
            Axes: The plot axes.
        """
        ds = _select_polarisation(
            self.astronomy_dataset.isel(time=time), polarisation
        ).isel(bin=0)
        data = ds["data"].where(~ds["flags"]) if flag else ds["data"]
        plot_kwargs.setdefault("x", "frequency")
        data.plot(**plot_kwargs)
        ax = plt.gca()
        ax.set_title(f"Spectrum (flagged = {flag!s})", fontsize=10)
        return ax

    def flag_persistent_rfi(
        self,
        rfi: pd.DataFrame,
        telescope: str | None = None,
        receiver: str | None = None,
    ) -> SubBand:
        """Flag persistent RFI (see :func:`sdhdf.flagging.flag_persistent_rfi`)."""
        return self.apply(
            flagging.flag_persistent_rfi,
            rfi=rfi,
            telescope=telescope,
            receiver=receiver,
        )

    def flag_outliers(self, sigma: float = 3, n_windows: int = 100) -> SubBand:
        """Flag outliers by windowed sigma clipping of each spectrum."""
        return self.apply(
            flagging.flag_sigma_clip, calibrator=False, sigma=sigma, n_windows=n_windows
        )

    def average(
        self,
        time: int | Quantity | None = None,
        frequency: int | Quantity | None = None,
        method: str = "mean",
    ) -> SubBand:
        """Weighted average in time and/or frequency (see :func:`sdhdf.averaging.average`).

        The calibrator data are averaged in time only, since their frequency
        resolution is independent of the astronomy data.
        """
        calibrator_dataset = self.calibrator_dataset
        if calibrator_dataset is not None and time is not None:
            try:
                calibrator_dataset = averaging.average(
                    calibrator_dataset, time=time, method=method
                )
            except ValueError as e:
                logger.warning(f"Calibrator data of {self.label} not averaged: {e}")
        return replace(
            self,
            astronomy_dataset=averaging.average(
                self.astronomy_dataset, time=time, frequency=frequency, method=method
            ),
            calibrator_dataset=calibrator_dataset,
        )


def _select_polarisation(ds: xr.Dataset, polarisation: int | str) -> xr.Dataset:
    if isinstance(polarisation, str):
        return ds.sel(polarisation=polarisation)
    return ds.isel(polarisation=polarisation)


@dataclass(repr=False)
class Beam:
    """An SDHDF beam.

    Attributes:
        label (str): The beam label.
        subbands (list[SubBand]): The bands of this beam.
        band_parameters (SDHDFTable | None): Band parameters table.
        calibrator_band_parameters (SDHDFTable | None): Calibrator band
            parameters table.
    """

    label: str
    subbands: list[SubBand]
    band_parameters: SDHDFTable | None = None
    calibrator_band_parameters: SDHDFTable | None = None

    @classmethod
    def from_file(cls, h5: h5py.File, label: str) -> Beam:
        """Read one beam of an open SDHDF file, lazily."""
        tables = _read_tables(
            find_group(h5[label], "metadata"), definition().beam_tables
        )
        band_parameters = tables.get("band_parameters")
        calibrator_band_parameters = tables.get("calibrator_band_parameters")
        subbands = [
            SubBand.from_file(
                h5,
                beam_label=label,
                label=band,
                band_parameters=table_row(band_parameters, band),
                calibrator_band_parameters=table_row(calibrator_band_parameters, band),
            )
            for band in _ordered_band_labels(h5[label], band_parameters)
        ]
        return cls(
            label=label,
            subbands=subbands,
            band_parameters=band_parameters,
            calibrator_band_parameters=calibrator_band_parameters,
        )

    def __repr__(self) -> str:
        return (
            f"Beam(label={self.label!r}, subbands={[sb.label for sb in self.subbands]})"
        )

    def __getattr__(self, name: str) -> SubBand:
        for sb in self.__dict__.get("subbands", []):
            if sb.label == name:
                subband: SubBand = sb
                return subband
        msg = f"'{type(self).__name__}' has no attribute '{name}'"
        raise AttributeError(msg)

    def __getitem__(self, key: int | str) -> SubBand:
        if isinstance(key, int):
            return self.subbands[key]
        return self.__getattr__(key)

    def plot_waterfall(
        self,
        subband: int | str,
        polarisation: int | str = 0,
        flag: bool = False,
        **plot_kwargs: Any,
    ) -> Axes | None:
        return self[subband].plot_waterfall(
            polarisation=polarisation, flag=flag, **plot_kwargs
        )

    def plot_spectrum(
        self,
        subband: int | str,
        time: int = 0,
        polarisation: int | str = 0,
        flag: bool = False,
        **plot_kwargs: Any,
    ) -> Axes:
        return self[subband].plot_spectrum(
            time=time, polarisation=polarisation, flag=flag, **plot_kwargs
        )

    def plot_wide(
        self,
        time: int = 0,
        polarisation: int | str = 0,
        flag: bool = False,
        **plot_kwargs: Any,
    ) -> Axes:
        """Plot spectra from all bands on one axis."""
        ax = plot_kwargs.pop("ax", None) or plt.subplots()[1]
        for sb in self.subbands:
            sb.plot_spectrum(
                time=time,
                polarisation=polarisation,
                flag=flag,
                ax=ax,
                label=sb.label,
                **plot_kwargs,
            )
        ax.legend()
        ax.set_title(f"Wide-band Spectrum (flagged = {flag!s})", fontsize=10)
        return ax


def _ordered_band_labels(
    beam: h5py.Group, band_parameters: SDHDFTable | None
) -> list[str]:
    """Band labels in band_parameters order, then any others found in the file."""
    in_file = band_labels(beam)
    ordered: list[str] = []
    if band_parameters is not None and "LABEL" in band_parameters.columns:
        ordered = [b for b in band_parameters["LABEL"] if b in in_file]
    return ordered + [b for b in in_file if b not in ordered]


@dataclass(repr=False)
class SDHDF:
    """An SDHDF file, of any definition version.

    Use :meth:`from_file` to read a file. Files of every version are read into
    the same SDHDF 4.0 representation. Processing methods return a new SDHDF
    and leave the original unchanged.

    Attributes:
        metadata (MetaData): File-level metadata.
        beams (list[Beam]): The beams in the file.
        filename (Path | None): The file the data were read from, if any.
    """

    metadata: MetaData
    beams: list[Beam]
    filename: Path | None = None

    @classmethod
    def from_file(
        cls, filename: str | Path, in_memory: bool = False, progress: bool = True
    ) -> SDHDF:
        """Read an SDHDF file of any version.

        Args:
            filename (str | Path): Path to the SDHDF file.
            in_memory (bool, optional): Load the data into memory. Defaults to
                False, reading data only when needed.
            progress (bool, optional): Show a progress bar when loading into
                memory. Defaults to True.

        Returns:
            SDHDF: The file contents.
        """
        filename = Path(filename)
        with h5py.File(filename, "r") as h5:
            metadata = MetaData.from_file(h5)
            beams = [Beam.from_file(h5, label) for label in beam_labels(h5)]
        logger.info(f"Read '{filename.name}' (SDHDF {metadata.version})")
        sdhdf = cls(metadata=metadata, beams=beams, filename=filename)
        return sdhdf.load(progress=progress) if in_memory else sdhdf

    def __repr__(self) -> str:
        return (
            f"SDHDF(filename={str(self.filename)!r}, version={self.metadata.version!r}, "
            f"beams={[b.label for b in self.beams]})"
        )

    def __getattr__(self, name: str) -> Beam:
        for beam in self.__dict__.get("beams", []):
            if beam.label == name:
                found: Beam = beam
                return found
        msg = f"'{type(self).__name__}' has no attribute '{name}'"
        raise AttributeError(msg)

    def __getitem__(self, key: int | str) -> Beam:
        if isinstance(key, int):
            return self.beams[key]
        return self.__getattr__(key)

    @property
    def subbands(self) -> list[SubBand]:
        """All bands of all beams."""
        return [sb for beam in self.beams for sb in beam.subbands]

    @property
    def history(self) -> pd.DataFrame:
        return self.metadata.history.table

    def map_subbands(self, func: Callable[[SubBand], SubBand]) -> SDHDF:
        """Copy with ``func`` applied to every band of every beam."""
        beams = [
            replace(beam, subbands=[func(sb) for sb in beam.subbands])
            for beam in self.beams
        ]
        return replace(self, beams=beams)

    def with_history(
        self, process: str, description: str, arguments: dict[str, Any]
    ) -> SDHDF:
        """Copy with a row added to the history table."""
        return replace(
            self, metadata=self.metadata.with_history(process, description, arguments)
        )

    def load(self, progress: bool = True) -> SDHDF:
        """Copy with all data loaded into memory.

        Args:
            progress (bool, optional): Show a progress bar. Defaults to True.

        Returns:
            SDHDF: The loaded copy.
        """
        datasets = [
            ds
            for sb in self.subbands
            for ds in (sb.astronomy_dataset, sb.calibrator_dataset)
            if ds is not None
        ]
        with ProgressBar() if progress else nullcontext():
            loaded = iter(dask.compute(*datasets))

        def load_subband(sb: SubBand) -> SubBand:
            astronomy = next(loaded)
            calibrator = next(loaded) if sb.calibrator_dataset is not None else None
            return replace(
                sb, astronomy_dataset=astronomy, calibrator_dataset=calibrator
            )

        return self.map_subbands(load_subband)

    def plot_waterfall(
        self,
        beam: int | str,
        subband: int | str,
        polarisation: int | str = 0,
        flag: bool = False,
        **plot_kwargs: Any,
    ) -> Axes | None:
        """Waterfall plot of one band."""
        return self[beam].plot_waterfall(
            subband=subband, polarisation=polarisation, flag=flag, **plot_kwargs
        )

    def plot_spectrum(
        self,
        beam: int | str,
        subband: int | str,
        time: int = 0,
        polarisation: int | str = 0,
        flag: bool = False,
        **plot_kwargs: Any,
    ) -> Axes:
        """Plot a single spectrum of one band."""
        return self[beam].plot_spectrum(
            subband=subband,
            time=time,
            polarisation=polarisation,
            flag=flag,
            **plot_kwargs,
        )

    def plot_wide(
        self,
        beam: int | str,
        time: int = 0,
        polarisation: int | str = 0,
        flag: bool = False,
        **plot_kwargs: Any,
    ) -> Axes:
        """Plot spectra from all bands of a beam."""
        return self[beam].plot_wide(
            time=time, polarisation=polarisation, flag=flag, **plot_kwargs
        )

    def print_obs_metadata(self) -> None:
        self.metadata.print_obs_metadata()

    def print_obs_config(self) -> None:
        self.metadata.print_obs_config()

    def flag_persistent_rfi(self, rfi: pd.DataFrame | None = None) -> SDHDF:
        """Flag persistent RFI in all bands.

        Args:
            rfi (pd.DataFrame, optional): RFI table. Defaults to the bundled
                table for the file's telescope.

        Returns:
            SDHDF: Flagged copy.
        """
        telescope = self.metadata.telescope
        if telescope is None:
            logger.warning("No telescope recorded in file; assuming Parkes.")
            telescope = "Parkes"
        if rfi is None:
            rfi = flagging.get_persistent_rfi(telescope=telescope)
        receiver = self.metadata.receiver
        flagged = self.map_subbands(
            lambda sb: sb.flag_persistent_rfi(
                rfi, telescope=telescope, receiver=receiver
            )
        )
        return flagged.with_history(
            "sdhdf.flag_persistent_rfi",
            "Flagged persistent RFI",
            {"telescope": telescope, "receiver": receiver, "n_entries": len(rfi)},
        )

    def auto_flag_rfi(
        self, sigma: float = 3, n_windows: int = 100, flag_persistent: bool = True
    ) -> SDHDF:
        """Automatic flagging: persistent RFI, then windowed sigma clipping.

        Args:
            sigma (float, optional): Clipping threshold. Defaults to 3.
            n_windows (int, optional): Windows per spectrum. Defaults to 100.
            flag_persistent (bool, optional): Flag persistent RFI first.

        Returns:
            SDHDF: Flagged copy.
        """
        sdhdf = self.flag_persistent_rfi() if flag_persistent else self
        flagged = sdhdf.map_subbands(
            lambda sb: sb.flag_outliers(sigma=sigma, n_windows=n_windows)
        )
        return flagged.with_history(
            "sdhdf.auto_flag_rfi",
            "Flagged outliers by windowed sigma clipping",
            {"sigma": sigma, "n_windows": n_windows},
        )

    def average(
        self,
        time: int | Quantity | None = None,
        frequency: int | Quantity | None = None,
        method: str = "mean",
    ) -> SDHDF:
        """Weighted average of all bands in time and/or frequency.

        Args:
            time (int | Quantity, optional): Number of integrations to average,
                or the integration time to average to (e.g. ``10 * u.s``).
            frequency (int | Quantity, optional): Number of channels to
                average, or the channel width to average to (e.g.
                ``0.5 * u.MHz``).
            method (str, optional): "mean" (weighted) or "median".

        Returns:
            SDHDF: Averaged copy.
        """
        averaged = self.map_subbands(
            lambda sb: sb.average(time=time, frequency=frequency, method=method)
        )
        return averaged.with_history(
            "sdhdf.average",
            "Weighted average in time and/or frequency",
            {"time": str(time), "frequency": str(frequency), "method": method},
        )

    def decimate(
        self, bins: float | int, axis: str = "frequency", use_median: bool = False
    ) -> SDHDF:
        """Average along one axis. Prefer :meth:`average`.

        Args:
            bins (float | int): If int, the number of samples to average. If
                float, the width to average to, in the units of the axis
                (seconds for time).
            axis (str, optional): "frequency" or "time". Defaults to "frequency".
            use_median (bool, optional): Use the median. Defaults to False.

        Returns:
            SDHDF: Averaged copy.
        """
        width: int | Quantity = bins
        if isinstance(bins, float):
            unit = "s"
            if axis == "frequency" and self.subbands:
                frequency = self.subbands[0].astronomy_dataset["frequency"]
                unit = frequency.attrs.get("units", "MHz")
            width = Quantity(bins, unit)
        return self.average(**{axis: width}, method="median" if use_median else "mean")

    def write(
        self, filename: str | Path, overwrite: bool = False, progress: bool = True
    ) -> None:
        """Write to a new file in the latest SDHDF format (4.0).

        Args:
            filename (str | Path): Output file.
            overwrite (bool, optional): Overwrite an existing file.
            progress (bool, optional): Show a progress bar. Defaults to True.
        """
        from sdhdf.writer import write_sdhdf  # noqa: PLC0415

        write_sdhdf(self, filename, overwrite=overwrite, progress=progress)


def upgrade(src: str | Path, dst: str | Path, overwrite: bool = False) -> None:
    """Convert an SDHDF file of any version to the latest version (4.0).

    Reading and writing any file also upgrades it; this adds a history row
    recording the conversion.

    Args:
        src (str | Path): Input file.
        dst (str | Path): Output file.
        overwrite (bool, optional): Overwrite ``dst`` if it exists.
    """
    sdhdf = SDHDF.from_file(src)
    sdhdf.with_history(
        "sdhdf.upgrade",
        f"Converted from SDHDF {sdhdf.metadata.version} to {LATEST_VERSION}",
        {"src": str(src), "dst": str(dst)},
    ).write(dst, overwrite=overwrite)
