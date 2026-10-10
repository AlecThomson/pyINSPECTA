"""Core SDHDF module"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

from sdhdf import averaging, flagging, history
from sdhdf.logger import logger
from sdhdf.reader import read_band, table_row
from sdhdf.schema import (
    BEAM_TABLES,
    CONFIG_TABLES,
    FILE_TABLES,
    ROOT_GROUPS,
    VersionInfo,
    band_labels,
    beam_labels,
    detect_version,
    find_child,
    find_group,
)
from sdhdf.tables import SDHDFTable

__all__ = ["SDHDF", "Beam", "MetaData", "SubBand", "upgrade"]


def _read_tables(group: h5py.Group | None, kinds: dict[str, tuple[str, ...]]) -> dict:
    tables = {}
    for kind, candidates in kinds.items():
        dset = find_child(group, candidates)
        if isinstance(dset, h5py.Dataset):
            tables[kind] = SDHDFTable.from_dataset(dset, kind)
    return tables


@dataclass
class MetaData:
    """File-level SDHDF metadata.

    Attributes:
        version_info (VersionInfo): Recorded and inferred definition version.
        primary_header (SDHDFTable): Primary header.
        beam_parameters (SDHDFTable): Beam parameters.
        history (SDHDFTable): Processing history.
        schedule (SDHDFTable): Observation schedule metadata.
        software_versions (SDHDFTable): Software used to create the file.
        instrument_configuration (SDHDFTable): Backend configuration.
        receiver_configuration (SDHDFTable): Receiver configuration (>= 4.0).
        telescope_configuration (SDHDFTable): Telescope configuration (>= 4.0).
    """

    version_info: VersionInfo
    tables: dict[str, SDHDFTable] = field(default_factory=dict)

    @classmethod
    def from_file(cls, h5: h5py.File) -> MetaData:
        version_info = detect_version(h5)
        meta = find_group(h5, ROOT_GROUPS["metadata"][0])
        config = find_child(h5, ROOT_GROUPS["configuration"])
        tables = {
            **_read_tables(meta, FILE_TABLES),
            **_read_tables(
                config if isinstance(config, h5py.Group) else None, CONFIG_TABLES
            ),
        }
        if "history" not in tables:
            tables["history"] = SDHDFTable(history.append(None), name="history")
        return cls(version_info=version_info, tables=tables)

    def __getattr__(self, name: str) -> SDHDFTable:
        tables = self.__dict__.get("tables", {})
        if name in tables:
            return tables[name]
        msg = f"'{type(self).__name__}' has no table '{name}'"
        raise AttributeError(msg)

    def __dir__(self):
        return [*super().__dir__(), *self.tables]

    @property
    def version(self) -> str:
        """SDHDF definition version of the file."""
        return self.version_info.version

    @property
    def telescope(self) -> str | None:
        ph = self.tables.get("primary_header")
        return ph.get("TELESCOPE") if ph is not None else None

    @property
    def receiver(self) -> str | None:
        ph = self.tables.get("primary_header")
        return ph.get("RECEIVER") if ph is not None else None

    def _print(self, kinds) -> None:
        for kind in kinds:
            if kind in self.tables:
                print(f"--- {kind} ---")  # noqa: T201
                print(self.tables[kind].table.to_string())  # noqa: T201

    def print_obs_metadata(self) -> None:
        """Print the observation metadata tables."""
        self._print(FILE_TABLES)

    def print_obs_config(self) -> None:
        """Print the observation configuration tables."""
        self._print(CONFIG_TABLES)


class SubBand:
    """An SDHDF band of one beam.

    Attributes:
        label (str): Band label (e.g. ``"band_SB0"``).
        beam_label (str): Label of the parent beam.
        astronomy_dataset (xr.Dataset): Astronomy data, flags and weights.
        calibrator_dataset (xr.Dataset | None): Noise-diode data, if recorded.
        band_parameters (dict): This band's row of ``band_parameters``.
        calibrator_band_parameters (dict): This band's row of
            ``calibrator_band_parameters``.
    """

    def __init__(
        self,
        h5: h5py.File,
        beam_label: str,
        label: str,
        band_parameters: dict[str, Any],
        calibrator_band_parameters: dict[str, Any],
        in_memory: bool = False,
    ):
        self.label = label
        self.beam_label = beam_label
        self.band_parameters = band_parameters
        self.calibrator_band_parameters = calibrator_band_parameters
        band = read_band(
            h5,
            beam_label,
            label,
            band_row=band_parameters,
            calibrator_band_row=calibrator_band_parameters,
            lazy=not in_memory,
        )
        self.astronomy_dataset: xr.Dataset = band.astronomy
        self.calibrator_dataset: xr.Dataset | None = band.calibrator
        if in_memory:
            logger.info(f"Loading {beam_label}/{label} into memory...")
            self.astronomy_dataset = self.astronomy_dataset.load()
            if self.calibrator_dataset is not None:
                self.calibrator_dataset = self.calibrator_dataset.load()

    def __repr__(self) -> str:
        sizes = dict(self.astronomy_dataset["data"].sizes)
        return f"SubBand(label={self.label!r}, beam={self.beam_label!r}, sizes={sizes})"

    def _apply(self, func, calibrator: bool = True, **kwargs) -> None:
        self.astronomy_dataset = func(self.astronomy_dataset, **kwargs)
        if calibrator and self.calibrator_dataset is not None:
            self.calibrator_dataset = func(self.calibrator_dataset, **kwargs)

    def plot_waterfall(
        self,
        polarisation: int | str = 0,
        flag: bool = False,
        **plot_kwargs,
    ):
        """Waterfall plot of the data

        Args:
            polarisation (int | str, optional): Polarisation index or label.
                Defaults to 0.
            flag (bool, optional): Blank flagged data. Defaults to False.
        """
        ds = _select_pol(self.astronomy_dataset, polarisation).isel(bin=0)
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
        **plot_kwargs,
    ):
        """Plot a single spectrum

        Args:
            time (int, optional): Integration index. Defaults to 0.
            polarisation (int | str, optional): Polarisation index or label.
            flag (bool, optional): Blank flagged data. Defaults to False.
        """
        ds = _select_pol(self.astronomy_dataset.isel(time=time), polarisation).isel(
            bin=0
        )
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
    ) -> None:
        """Flag persistent RFI (see :func:`sdhdf.flagging.flag_persistent_rfi`)."""
        self._apply(
            flagging.flag_persistent_rfi,
            rfi=rfi,
            telescope=telescope,
            receiver=receiver,
        )

    def autoflag(self, sigma: float = 3, n_windows: int = 100) -> None:
        """Flag outliers by windowed sigma clipping of each spectrum."""
        self._apply(
            flagging.flag_sigma_clip, calibrator=False, sigma=sigma, n_windows=n_windows
        )

    def average(
        self,
        time: int | float | None = None,
        frequency: int | float | None = None,
        method: str = "mean",
    ) -> None:
        """Weighted average in time and/or frequency (see :func:`sdhdf.averaging.average`).

        The calibrator data are averaged in time only, since their frequency
        resolution is independent of the astronomy data.
        """
        self.astronomy_dataset = averaging.average(
            self.astronomy_dataset, time=time, frequency=frequency, method=method
        )
        if self.calibrator_dataset is not None and time is not None:
            n_time = self.calibrator_dataset.sizes["time"]
            if isinstance(time, float) or n_time >= int(time):
                self.calibrator_dataset = averaging.average(
                    self.calibrator_dataset, time=time, method=method
                )


def _select_pol(ds: xr.Dataset, polarisation: int | str) -> xr.Dataset:
    if isinstance(polarisation, str):
        return ds.sel(polarisation=polarisation)
    return ds.isel(polarisation=polarisation)


class Beam:
    """An SDHDF beam.

    Attributes:
        label (str): The beam label.
        subbands (list[SubBand]): The bands of this beam.
        band_parameters (SDHDFTable | None): Band parameters table.
        calibrator_band_parameters (SDHDFTable | None): Calibrator band
            parameters table.
    """

    def __init__(self, h5: h5py.File, label: str, in_memory: bool = False):
        self.label = label
        tables = _read_tables(find_group(h5[label], "metadata"), BEAM_TABLES)
        self.band_parameters = tables.get("band_parameters")
        self.calibrator_band_parameters = tables.get("calibrator_band_parameters")
        self.subbands = [
            SubBand(
                h5,
                beam_label=label,
                label=band,
                band_parameters=table_row(self.band_parameters, band),
                calibrator_band_parameters=table_row(
                    self.calibrator_band_parameters, band
                ),
                in_memory=in_memory,
            )
            for band in _ordered_bands(h5[label], self.band_parameters)
        ]

    def __repr__(self) -> str:
        return (
            f"Beam(label={self.label!r}, subbands={[sb.label for sb in self.subbands]})"
        )

    def __getattr__(self, name: str) -> SubBand:
        for sb in self.__dict__.get("subbands", []):
            if sb.label == name:
                return sb
        msg = f"'{type(self).__name__}' has no attribute '{name}'"
        raise AttributeError(msg)

    def __getitem__(self, key: int | str) -> SubBand:
        if isinstance(key, int):
            return self.subbands[key]
        return getattr(self, key)

    def plot_waterfall(
        self,
        subband: int | str,
        polarisation: int | str = 0,
        flag: bool = False,
        **plot_kwargs,
    ):
        return self[subband].plot_waterfall(
            polarisation=polarisation, flag=flag, **plot_kwargs
        )

    def plot_spectrum(
        self,
        subband: int | str,
        time: int = 0,
        polarisation: int | str = 0,
        flag: bool = False,
        **plot_kwargs,
    ):
        return self[subband].plot_spectrum(
            time=time, polarisation=polarisation, flag=flag, **plot_kwargs
        )

    def plot_wide(
        self,
        time: int = 0,
        polarisation: int | str = 0,
        flag: bool = False,
        **plot_kwargs,
    ):
        """Plot spectra from all bands on one axis."""
        _, ax = plt.subplots()
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


def _ordered_bands(beam: h5py.Group, band_parameters: SDHDFTable | None) -> list[str]:
    """Band labels in band_parameters order, then any others found in the file."""
    in_file = band_labels(beam)
    ordered = []
    if band_parameters is not None and "LABEL" in band_parameters.columns:
        ordered = [b for b in band_parameters["LABEL"] if b in in_file]
    return ordered + [b for b in in_file if b not in ordered]


class SDHDF:
    """An SDHDF file, of any definition version.

    Files are read into a common, SDHDF 4.0-like representation, so the same
    code works for all versions. Data are read lazily unless ``in_memory`` is
    set.

    Args:
        filename (str | Path): Path to the SDHDF file.
        in_memory (bool, optional): Load data into memory. Defaults to False.

    Attributes:
        metadata (MetaData): File-level metadata.
        beams (list[Beam]): The beams in the file.
    """

    def __init__(self, filename: str | Path, in_memory: bool = False):
        self.filename = Path(filename)
        self.in_memory = in_memory
        with h5py.File(self.filename, "r") as h5:
            self.metadata = MetaData.from_file(h5)
            logger.info(
                f"Read '{self.filename.name}' (SDHDF {self.metadata.version_info.version})"
            )
            self.beams = [
                Beam(h5, label, in_memory=in_memory) for label in beam_labels(h5)
            ]

    def __repr__(self) -> str:
        return (
            f"SDHDF(filename={str(self.filename)!r}, version={self.metadata.version!r}, "
            f"beams={[b.label for b in self.beams]})"
        )

    def __getattr__(self, name: str) -> Beam:
        for beam in self.__dict__.get("beams", []):
            if beam.label == name:
                return beam
        msg = f"'{type(self).__name__}' has no attribute '{name}'"
        raise AttributeError(msg)

    def __getitem__(self, key: int | str) -> Beam:
        if isinstance(key, int):
            return self.beams[key]
        return getattr(self, key)

    @property
    def subbands(self) -> list[SubBand]:
        """All bands of all beams."""
        return [sb for beam in self.beams for sb in beam.subbands]

    @property
    def history(self) -> pd.DataFrame:
        return self.metadata.history.table

    def _add_history(self, process: str, description: str, arguments: dict) -> None:
        row = history.generate_history_row(process, description, arguments)
        self.metadata.history.table = history.append(self.history, row)

    # --- Plotting ---------------------------------------------------------

    def plot_waterfall(
        self,
        beam: int | str,
        subband: int | str,
        polarisation: int | str = 0,
        flag: bool = False,
        **plot_kwargs,
    ):
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
        **plot_kwargs,
    ):
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
        **plot_kwargs,
    ):
        """Plot spectra from all bands of a beam."""
        return self[beam].plot_wide(
            time=time, polarisation=polarisation, flag=flag, **plot_kwargs
        )

    def print_obs_metadata(self) -> None:
        self.metadata.print_obs_metadata()

    def print_obs_config(self) -> None:
        self.metadata.print_obs_config()

    # --- Processing -------------------------------------------------------

    def flag_persistent_rfi(self, rfi: pd.DataFrame | None = None) -> None:
        """Flag persistent RFI in all bands.

        Args:
            rfi (pd.DataFrame, optional): RFI table. Defaults to the bundled
                table for the file's telescope.
        """
        telescope = self.metadata.telescope
        if telescope is None:
            logger.warning("No telescope recorded in file; assuming Parkes.")
            telescope = "Parkes"
        if rfi is None:
            rfi = flagging.get_persistent_rfi(telescope=telescope)
        for sb in self.subbands:
            sb.flag_persistent_rfi(
                rfi, telescope=telescope, receiver=self.metadata.receiver
            )
        self._add_history(
            "sdhdf.flag_persistent_rfi",
            "Flagged persistent RFI",
            {
                "telescope": telescope,
                "receiver": self.metadata.receiver,
                "n_entries": len(rfi),
            },
        )

    def auto_flag_rfi(
        self, sigma: float = 3, n_windows: int = 100, flag_persistent: bool = True
    ) -> None:
        """Automatic flagging: persistent RFI, then windowed sigma clipping.

        Args:
            sigma (float, optional): Clipping threshold. Defaults to 3.
            n_windows (int, optional): Windows per spectrum. Defaults to 100.
            flag_persistent (bool, optional): Flag persistent RFI first.
        """
        if flag_persistent:
            self.flag_persistent_rfi()
        for sb in self.subbands:
            sb.autoflag(sigma=sigma, n_windows=n_windows)
        self._add_history(
            "sdhdf.auto_flag_rfi",
            "Flagged outliers by windowed sigma clipping",
            {"sigma": sigma, "n_windows": n_windows},
        )

    def average(
        self,
        time: int | float | None = None,
        frequency: int | float | None = None,
        method: str = "mean",
    ) -> None:
        """Weighted average of all bands in time and/or frequency.

        Args:
            time (int | float, optional): If int, the number of integrations to
                average. If float, the desired integration time in seconds.
            frequency (int | float, optional): If int, the number of channels to
                average. If float, the desired channel width (MHz).
            method (str, optional): "mean" (weighted) or "median".
        """
        for sb in self.subbands:
            sb.average(time=time, frequency=frequency, method=method)
        self._add_history(
            "sdhdf.average",
            "Weighted average in time and/or frequency",
            {"time": time, "frequency": frequency, "method": method},
        )

    def decimate(
        self, bins: float | int, axis: str = "frequency", use_median: bool = False
    ) -> None:
        """Average along one axis. Deprecated: use :meth:`average`."""
        self.average(**{axis: bins}, method="median" if use_median else "mean")

    # --- Output -----------------------------------------------------------

    def write(self, filename: str | Path, overwrite: bool = False) -> None:
        """Write to a new file in the latest SDHDF format (4.0).

        Args:
            filename (str | Path): Output file.
            overwrite (bool, optional): Overwrite an existing file.
        """
        from sdhdf.writer import write_sdhdf  # noqa: PLC0415

        write_sdhdf(self, filename, overwrite=overwrite)


def upgrade(src: str | Path, dst: str | Path, overwrite: bool = False) -> None:
    """Convert an SDHDF file of any version to the latest version (4.0).

    Args:
        src (str | Path): Input file.
        dst (str | Path): Output file.
        overwrite (bool, optional): Overwrite ``dst`` if it exists.
    """
    sdhdf = SDHDF(src)
    sdhdf._add_history(
        "sdhdf.upgrade",
        f"Converted from SDHDF {sdhdf.metadata.version} to 4.0",
        {"src": str(src), "dst": str(dst)},
    )
    sdhdf.write(dst, overwrite=overwrite)
