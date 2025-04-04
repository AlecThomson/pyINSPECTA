"""Core SDHDF module"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import cast

import dask.array as da
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from astropy.table import Table
from dask.distributed import Client
from tqdm.auto import tqdm
from xarray import DataArray, Dataset, Variable

from pyINSPECTA import flagging, history
from pyINSPECTA.logger import logger
from pyINSPECTA.tables import SDHDFTable


def _get_sdhdf_version(filename: Path) -> tuple[float, Path]:
    """Get the SDHDF version of a file and return the path
       to the definition template

    Args:
        filename (Path): Path to the SDHDF file

    Returns:
        str: SDHDF version
        str: Path to definition json template
    """
    with h5py.File(filename, "r") as f:
        primary_header = SDHDFTable(f["metadata/primary_header"], 0.0)
        version = None
        if "HDR_DEFN_VERSION" in primary_header:
            hdr_key = "HDR_DEFN_VERSION"
            if "1.9" in str(primary_header[hdr_key]):
                version = float("1.9")
            else:
                version = float(primary_header[hdr_key])
        else:
            hdr_key = "HEADER_DEFINITION_VERSION"
            # compound attributes for SDHDF >=4.0
            version = float(primary_header.attrs[hdr_key][0][2].decode())
        if version is None:
            msg = f"SDHDF version not found in file '{filename}'"
            raise ValueError(msg)
        if version <= 2.0:
            version = 2.0
        elif (version > 2.0) and (version <= 2.1):
            version = 2.1
        elif (version >= 2.2) and (version <= 2.9):
            version = 2.9

        try:
            with resources.path("pyINSPECTA", "definitions") as definition_dir:
                definition_file = definition_dir / f"sdhdf_def_v{version}.json"
        except ValueError as e:
            msg = f"SDHDF definition template {definition_file} not found."
            raise ValueError(msg) from e

    return version, definition_file


@dataclass
class MetaData:
    """An SDHDF metadata object

    Args:
        filename (Path): Path to the SDHDF file

    Attributes:
        beam_params (SDHDFTable): The beam parameters
        history (SDHDFTable): File history
        primary_header (SDHDFTable): Primary header
        backend_config (SDHDFTable): Backend configuration
        cal_backend_config (SDHDFTable): Calibration backend configuration
        software (SDHDFTable): Software versions used in creation of the file
        schedule (SDHDFTable): Observation schedule metadata (if available)

    Methods:
        print_obs_metadata: Quickly list the observation metadata
        print_obs_config: Quickly list the observation configuration
        write: Write metadata to file [NOT YET IMPLEMENTED]

    """

    filename: Path

    def __post_init__(self):
        version, definition_file = _get_sdhdf_version(self.filename)
        logger.info(f"SDHDF version: {version}")
        logger.debug(f"Loading SDHDF definition template: {definition_file}")

        # load the definition
        with definition_file.open("r") as f:
            definition = dict(json.load(f))

        self.definition = definition
        self.version = version

        with h5py.File(self.filename, "r") as h5file:
            h5file.visititems(
                lambda name, obj: logger.debug(f"Name: {name} Object: {obj}")
            )
            for top_group in h5file:
                is_beam = "beam" in top_group

                base_path = f"/{top_group}" if is_beam else ""
                def_values = (
                    definition.get(top_group) if not is_beam else definition.get("beam")
                )

                if def_values is None:
                    logger.warning(
                        f"No definition found for group '{top_group}'. Ignoring..."
                    )
                    continue
                self._get_metadata(def_values, base_path, h5file)

    def _get_metadata(
        self, def_values: dict, base_path: str, h5file: h5py.File
    ) -> None:
        """Set the metadata attributes from the definition

        Args:
            def_values (dict): Definition values
            base_path (str): Base path in the h5file
            h5file (h5py.File): h5py file object
        """
        for key, val in def_values.items():
            self._set_attributes(key, val, base_path, h5file)

    def _set_attributes(
        self, key: str, val: str | dict, base_path: str, h5file: h5py.File
    ) -> None:
        """Recursively set the metadata attributes

        Args:
            key (str): The key in the definition
            val (str | dict): The value in the definition
            base_path (str): Base path in the h5file
            h5file (h5py.File): h5py file object
        """
        if key == "attributes":
            return None
        if isinstance(val, dict):
            if key == "band":
                band_paths = [k for k in h5file[base_path] if "band" in k]
                for band_path in band_paths:
                    self._get_metadata(
                        val, base_path=f"{base_path}/{band_path}", h5file=h5file
                    )
                return None
            self._get_metadata(val, base_path=base_path, h5file=h5file)
            return None
        path = f"{base_path}/{val}"
        logger.debug(f"Path: {path}")
        if path not in h5file:
            logger.warning(f"Path '{path}' not found in file")
            return None
        attr = SDHDFTable(h5file[path], self.version)

        return setattr(self, key, attr)

    def print_obs_metadata(self, format: str = "grid") -> None:
        """Print observation metadata to the terminal"""
        _ = format
        for key in self.definition["metadata"]:
            if key in self.__dict__:
                self.__dict__[key]
            else:
                logger.warning(f"No metadata found for key '{key}'. Ignoring...")

    def print_obs_config(self, format: str = "grid") -> None:
        """Print the observation configuration to the terminal"""
        _ = format
        for key in self.definition["config"]:
            if key in self.__dict__:
                self.__dict__[key]
            else:
                logger.warning(f"No metadata found for key '{key}'. Ignoring...")

    def write(self, filename: str | Path, overwrite: bool = False) -> pd.DataFrame:
        """Write the metadata to a file

        Args:
            filename (str | Path): Path to the output file
        """
        if isinstance(filename, str):
            filename = Path(filename)

        if filename.exists() and not overwrite:
            msg = f"File '{filename}' already exists"
            raise FileExistsError(msg)
        for name in ("metadata", "config"):
            for key, val in tqdm(self.definition[name].items(), desc=f"Writing {name}"):
                meta_df = self.__dict__[key]
                meta_df.to_hdf(filename, key=f"{val}", mode="a", data_columns=True)

        return history.generate_history_row()


@dataclass
class SubBand:
    """An SDHDF sub-band data object

    Args:
        label (str): Sub-band label
        filename (Path): Path to the SDHDF file
        definition (dict): SDHDF definition
        beam_label (str): Beam label
        in_memory (bool, optional): Load the data into memory. Defaults to False.
        client (Client, optional): Dask client. Defaults to None.

    Attributes:
        data (DataArray): The sub-band data as an xarray DataArray
        flag (DataArray): The sub-band flag as an xarray DataArray
        meta (DataFrame): The sub-band metadata as a pandas DataFrame

    Methods:
        plot_waterfall: Plot the sub-band data as a waterfall plot
        plot_spectrum: Plot a single spectrum from the sub-band data

    """

    label: str
    filename: Path
    definition: dict
    beam_label: str
    in_memory: bool = False
    client: Client | None = None

    def __post_init__(self):
        # Get the astronomy data
        self.astronomy_dataset = self._get_data()
        # Now get the calibrator data
        self.calibrator_dataset = self._get_cal()
        # TODO: Get the calibrator data

    def _get_cal(self):
        return

    @staticmethod
    def _get_data_dimensions(
        dim_labels: str,
        data_shape: tuple[int],
        meta: SDHDFTable,
    ) -> list[str] | np.ndarray:
        if isinstance(dim_labels, np.ndarray):
            return dim_labels
        if isinstance(dim_labels, bytes):
            dim_labels = dim_labels.decode()
        if dim_labels != "NOT SET":
            return dim_labels.split(",")

        logger.warning("No dimension labels found in file! Using default labels.")
        usual_dims = ["time", "polarization", "channel", "bin"]
        dims = []
        for i, shape in enumerate(data_shape):
            if shape == len(meta):
                dims.append("time")
            else:
                dims.append(usual_dims[i])
        return dims

    @staticmethod
    def _get_freq_dimensions(
        dims: list[str],
        freq_dim_labels: str,
        data_shape: tuple[int],
        freq_shape: tuple[int],
    ) -> list[str] | np.ndarray:
        if isinstance(freq_dim_labels, np.ndarray):
            return freq_dim_labels
        if isinstance(freq_dim_labels, bytes):
            freq_dim_labels = freq_dim_labels.decode()
        if freq_dim_labels != "NOT SET":
            return freq_dim_labels.split(",")

        logger.warning(
            "No frequency dimension labels found in file! Attempting to match dimensions to data shape..."
        )
        # Attempt to match the dimensions to the data shape
        freq_dims = []
        for i_shape in freq_shape:
            for i in range(len(data_shape)):
                if i_shape == data_shape[i]:
                    freq_dims.append(dims[i])
                    break

        if len(freq_dims) == len(freq_shape):
            return freq_dims

        logger.warning(
            "Could not automatically match frequency dimensions to data dimensions! Using default labels."
        )
        return [f"freq_{i}" for i in range(len(freq_shape))]

    def _get_data(self):
        """Get the astronomy sub-band data"""
        astro_def = self.definition["beam"]["band"]["astronomy"]
        meta_def = self.definition["beam"]["band"]["metadata"]
        version = self.definition["version"]
        sb_path = f"{self.beam_label}/{self.label}"

        with h5py.File(self.filename, "r") as h5:
            data_path = f"{sb_path}/{astro_def['data']}"
            freq_path = f"{sb_path}/{astro_def['frequency']}"
            meta_path = f"{sb_path}/{meta_def['obs_params']}"

            data = h5[data_path]
            freqs = h5[freq_path]
            meta = SDHDFTable(h5[meta_path], version)
            self.metadata = meta

            # Get the flags (if they exist)
            if (
                "flags" in astro_def
                and astro_def["flags"]
                and f"{sb_path}/{astro_def['flags']}" in h5
            ):
                flag_path = f"{sb_path}/{astro_def['flags']}"
                flags = h5[flag_path][:]
                # Ensure flag has same shape as data
                flag_reshape = flags[:].copy()
                for i, s in enumerate(data.shape):
                    if i > len(flag_reshape.shape) - 1:
                        flag_reshape = np.expand_dims(flag_reshape, axis=-1)
                    elif flag_reshape.shape[i] == s:
                        continue
                    else:
                        flag_reshape = np.expand_dims(flag_reshape, axis=i)
                flags = flag_reshape
            else:
                logger.warning(
                    f"No flags found for sub-band '{self.label}' in file '{self.filename}'!"
                )
                logger.warning(f"Band '{self.label}' flags will be set to all zeros.")
                flags = np.zeros_like(data)

            # Load into memory if requested
            if self.in_memory:
                logger.info(f"Loading {self.label} into memory...")
                data = np.array(data)
                freqs = np.array(freqs)
                flags = np.array(flags)
            else:
                data = da.from_array(data[:], chunks="auto")
                freqs = da.from_array(freqs[:], chunks="auto")
                flags = da.from_array(flags[:], chunks="auto")

            # Process into xarray
            coords = {col: ("time", meta[col].to_numpy()) for col in meta.table.columns}

            dim_labels = h5[data_path].attrs["DIMENSION_LABELS"]
            if isinstance(dim_labels, bytes):
                dim_labels = dim_labels.decode()
            elif isinstance(dim_labels, np.ndarray):
                dim_labels = dim_labels.astype(str)

            if "polarisation" in dim_labels:
                dim_labels[1] = "polarization"

            dims = self._get_data_dimensions(
                dim_labels=dim_labels,
                data_shape=data.shape,
                meta=meta,
            )
            if "DIMENSION_LABELS" in h5[freq_path].attrs:
                freq_dim_labels = h5[freq_path].attrs["DIMENSION_LABELS"]
            else:
                freq_dim_labels = "NOT SET"
            if isinstance(freq_dim_labels, bytes):
                freq_dim_labels = freq_dim_labels.decode()
            elif isinstance(freq_dim_labels, np.ndarray):
                freq_dim_labels = freq_dim_labels.astype(str)

            freq_dims = self._get_freq_dimensions(
                dims=dims,
                freq_dim_labels=freq_dim_labels,
                data_shape=data.shape,
                freq_shape=freqs.shape,
            )
            if "UNIT" in h5[freq_path].attrs:
                freq_unit = h5[freq_path].attrs["UNIT"]
                if isinstance(h5[freq_path].attrs["UNIT"], bytes):
                    if float(version) >= 4.0:
                        freq_unit = freq_unit[0][2].decode()
                    else:
                        freq_unit = freq_unit.decode()
            else:
                freq_unit = "NOT SET"
            coords["frequency"] = Variable(
                dims=freq_dims,
                data=freqs,
                attrs={"units": freq_unit},
            ).squeeze()

            attrs = dict(h5[data_path].attrs)
            for key, val in attrs.items():
                if float(version) >= 4.0:
                    if isinstance(val, bytes):
                        attrs[key] = val[0][2].decode()
                    elif "DIMENSION" in key:
                        attrs[key] = val
                    else:
                        attrs[key] = val[0][2].decode()
                elif isinstance(val, bytes):
                    attrs[key] = val.decode()
                else:
                    attrs[key] = val

            data_xr = DataArray(
                data,
                dims=dims,
                coords=coords,
                name=f"{self.label}_data",
                attrs=attrs,
            )
            # Check if data has beam dimension
            if "beam" in data_xr.dims:
                data_xr = data_xr.isel(beam=0)
            data_xr.attrs["units"] = data_xr.UNIT
            self.attrs = attrs

            flag_xr = DataArray(
                flags,
                dims=dims,
                coords=coords,
                name=f"{self.label}_flag",
            ).squeeze()
            # Same as above
            if "beam" in flag_xr.dims:
                flag_xr = flag_xr.isel(beam=0)

            return Dataset(
                {
                    "data": data_xr,
                    "flag": flag_xr,
                    "metadata": xr.DataArray(meta.table, dims=["time", "meta"]),
                },
            )

    def plot_waterfall(
        self,
        polarization: int = 0,
        flag: bool = False,
        **plot_kwargs,
    ):
        """Waterfall plot of the data

        Args:
            polarization (int, optional): Polarization to select. Defaults to 0.
            bin (int, optional): Bin to select. Defaults to 0.
            flag (bool, optional): Blank flagged data. Defaults to False.
        """
        sub_data = self.astronomy_dataset.isel(
            polarization=polarization,
        )
        if sub_data.dims["time"] == 1:
            logger.warning("Cannot create a waterfall plot with a single integration!")
            ax = None
        else:
            if flag:
                sub_data = sub_data.where(sub_data.flag == 0)
            sub_data.data.plot(x="frequency", **plot_kwargs)
            ax = plt.gca()
            ax.set_title(f"Waterfall Plot (flagged = {flag!s})", fontsize=10)

        return ax

    def plot_spectrum(
        self,
        time: int,
        polarization: int = 0,
        flag: bool = False,
        **plot_kwargs,
    ):
        sub_data = self.astronomy_dataset.isel(
            time=time,
            polarization=polarization,
        )
        if flag:
            sub_data = sub_data.where(sub_data.flag == 0)
        sub_data.data.plot(**plot_kwargs)
        ax = plt.gca()
        ax.set_title(f"Spectrum (flagged = {flag!s})", fontsize=10)

        return ax

    def autoflag(self, sigma=3, n_windows=100):
        """Automatic flagging using rolling sigma clipping"""
        data_xr_flg = self.astronomy_dataset.data.where(
            ~self.astronomy_dataset.flag.astype(bool)
        )
        # Set chunks for parallel processing
        chunks = dict.fromkeys(data_xr_flg.dims, 1)
        chunks["frequency"] = len(self.astronomy_dataset.data.frequency)
        data_xr_flg = data_xr_flg.chunk(chunks)
        mask = (
            xr.DataArray(
                data_xr_flg.data.map_blocks(
                    flagging.box_filter,
                    sigma=sigma,
                    n_windows=n_windows,
                    dtype=bool,
                ),
                dims=data_xr_flg.dims,
                coords=data_xr_flg.coords,
            )
            .sum(dim="polarization")
            .astype(bool)
        )
        self.astronomy_dataset["flag"] = mask.astype(int).compute()
        return history.generate_history_row()

    def decimate(
        self, bins: float | int, axis: str = "frequency", use_median: bool = False
    ) -> pd.DataFrame:
        """Average the data along the an axis

        Args:
            bins (float | int): If int, the number of channels to bin in an average.
                If float, the desired width of a channel after averaging.
            axis (str, optional): The axis to decimate along. Defaults to "frequency".
            use_median (bool, optional): Use the median instead of the mean. Defaults to False.

        Returns:
            pd.DataFrame: The history row

        Raises:
            NotImplementedError: Decimation along the time axis is not yet implemented

        """

        if axis == "time":
            # TODO: Figure out how to decimate along the time axis - including the metadata / coords
            msg = "Decimation along the time axis is not yet implemented"
            raise NotImplementedError(msg)
        dataset = self.astronomy_dataset
        if isinstance(bins, float):
            # Convert to integer number of bins
            try:
                unit = dataset.data[axis].units
            except AttributeError:
                unit = "units"
            logger.info(f"Asked for a bin width of {bins} {unit}")
            logger.info(
                f"Dimension {axis} has range {dataset[axis].min()} to {dataset[axis].max()}: {dataset[axis].max() - dataset[axis].min()} {unit}"
            )
            bins = int((dataset[axis].max() - dataset[axis].min()) / bins)

        logger.info(f"Using {bins} channels per bin")

        # Apply CASA-style decimation
        flagged = dataset.where(dataset.flag == 0)
        unflagged = dataset

        if use_median:
            unflagged_dec = (
                unflagged.coarsen(**{axis: bins}, boundary="trim")
                .construct(**{axis: ("decimated", "original")})
                .median(dim="original", skipna=True)
                .rename({"decimated": axis})
            )

            flagged_dec = (
                flagged.coarsen(**{axis: bins}, boundary="trim")
                .construct(**{axis: ("decimated", "original")})
                .median(dim="original", skipna=True)
                .rename({"decimated": axis})
            )
            axis_dec = unflagged[axis].coarsen(**{axis: bins}, boundary="trim").median()

        else:
            unflagged_dec = (
                unflagged.coarsen(**{axis: bins}, boundary="trim")
                .construct(**{axis: ("decimated", "original")})
                .mean(dim="original", skipna=True)
                .rename({"decimated": axis})
            )

            flagged_dec = (
                flagged.coarsen(**{axis: bins}, boundary="trim")
                .construct(**{axis: ("decimated", "original")})
                .mean(dim="original", skipna=True)
                .rename({"decimated": axis})
            )
            axis_dec = unflagged[axis].coarsen(**{axis: bins}, boundary="trim").mean()

        unflagged_dec[axis] = axis_dec
        flagged_dec[axis] = axis_dec
        new_flag = flagged_dec.flag.fillna(1)
        new_data = flagged_dec.data
        new_data = new_data.fillna(unflagged_dec.data)
        dataset_dec = xr.Dataset(
            {
                "data": new_data,
                "flag": new_flag,
                "metadata": dataset.metadata,
            },
            attrs=dataset.attrs,
        )
        self.astronomy_dataset = dataset_dec
        return history.generate_history_row()

    def _write_astronomy_dataset(self, filename: Path) -> pd.DataFrame:
        astro_def = self.definition["beam"]["band"]["astronomy"]
        sb_path = f"{self.beam_label}/{self.label}"
        with h5py.File(filename, "w") as f:
            f[f"{sb_path}/{astro_def['data']}"] = self.astronomy_dataset.data.to_numpy()
            f[f"{sb_path}/{astro_def['frequency']}"] = (
                self.astronomy_dataset.frequency.to_numpy()
            )
            if "flags" in self.definition["beam"]["band"]["astronomy"]:
                f[f"{sb_path}/{astro_def['flags']}"] = (
                    self.astronomy_dataset.flag.to_numpy()
                )
            else:
                logger.warning("No flags in definition")
                logger.info("Saving flags to /astronomy_data/flags")
                f[f"{sb_path}/astronomy_data/flags"] = (
                    self.astronomy_dataset.flag.to_numpy()
                )

        self.astronomy_dataset.metadata.to_dataframe().to_hdf(
            filename,
            f"{sb_path}/metadata",
            # f"{sb_path}/{astro_def['metadata']}",
            mode="a",
        )
        return history.generate_history_row()

    def _write_cal_dataset(self, filename: Path):
        # TODO: Write the cal dataset
        _ = filename
        return history.generate_history_row()

    def write(
        self, filename: str | Path, overwrite: bool = False
    ) -> list[pd.DataFrame]:
        """Write the dataset to a file

        Args:
            filename (str | Path): The filename to write to
            overwrite (bool, optional): Overwrite the file if it exists. Defaults to False.

        Raises:
            FileExistsError: The file exists and overwrite is False

        """
        if isinstance(filename, str):
            filename = Path(filename)
        if filename.exists() and not overwrite:
            msg = f"{filename} already exists"
            raise FileExistsError(msg)

        astro_hist = self._write_astronomy_dataset(filename)
        cal_hist = self._write_cal_dataset(filename)

        return [astro_hist, cal_hist, history.generate_history_row()]


@dataclass
class Beam:
    """An SDHDF beam data object

    Args:
        label (str): The beam label
        filename (Path): The SDHDF file
        definition (dict): The SDHDF definition
        in_memory (bool, optional): Load data into memory. Defaults to False.
        client (Client, optional): Dask client. Defaults to None.

    Attributes:
        subbands (list[SubBand]): A list of subbands

    Methods:
        plot_waterfall: Plot a waterfall plot of the data
        plot_spectrum: Plot a spectrum of the data
        plot_wide: Plot spectra from all subbands

    """

    label: str
    filename: Path
    definition: dict
    in_memory: bool = False
    client: Client | None = None

    def __post_init__(self):
        meta_def = self.definition["beam"]["metadata"]
        with h5py.File(self.filename, "r") as f:
            sb_avail = Table.read(f, path=self.label + f"/{meta_def['band_params']}")
            self.subbands = [
                SubBand(
                    label=sb,
                    filename=self.filename,
                    definition=self.definition,
                    beam_label=self.label,
                    in_memory=self.in_memory,
                    client=self.client,
                )
                for sb in sb_avail["LABEL"]
            ]

            for sb in self.subbands:
                self.__dict__[sb.label] = sb

    def plot_waterfall(
        self,
        subband: int | str,
        polarization: int = 0,
        flag: bool = False,
        **plot_kwargs,
    ):
        if isinstance(subband, int):
            sub_band = self.subbands[subband]
        elif isinstance(subband, str):
            sub_band = cast(SubBand, self.__dict__[subband])

        return sub_band.plot_waterfall(
            polarization=polarization,
            flag=flag,
            **plot_kwargs,
        )

    def plot_spectrum(
        self,
        subband: int | str,
        time: int = 0,
        polarization: int = 0,
        flag: bool = False,
        **plot_kwargs,
    ):
        if isinstance(subband, int):
            sub_band = self.subbands[subband]
        elif isinstance(subband, str):
            sub_band = cast(SubBand, self.__dict__[subband])

        return sub_band.plot_spectrum(
            time=time,
            polarization=polarization,
            flag=flag,
            **plot_kwargs,
        )

    def plot_wide(
        self,
        time: int = 0,
        polarization: int = 0,
        flag: bool = False,
        **plot_kwargs,
    ):
        fig, ax = plt.subplots()
        for _i, sb in enumerate(self.subbands):
            sb.plot_spectrum(
                time=time,
                polarization=polarization,
                flag=flag,
                ax=ax,
                label=sb.label,
                **plot_kwargs,
            )
        ax.legend()
        ax.set_title(f"Wide-band Spectrum (flagged = {flag!s})", fontsize=10)

        return ax

    def autoflag(self, sigma=3, n_windows=100) -> list[pd.DataFrame]:
        """Automatic flagging using rolling sigma clipping"""
        hists = []
        for sb in tqdm(self.subbands, desc="Flagging subbands"):
            hist = sb.autoflag(
                sigma=sigma,
                n_windows=n_windows,
            )
            hists.append(hist)
        return hists

    def decimate(
        self, bins: float | int, axis: str = "frequency", use_median: bool = False
    ) -> list[pd.DataFrame]:
        """Decimate the data

        Args:
            bins (float | int): If int, the number of channels to bin in an average.
                If float, the desired width of a channel after averaging.
            axis (str, optional): The axis to decimate along. Defaults to "frequency".
            use_median (bool, optional): Use the median instead of the mean. Defaults to False.

        Returns:
            list[pd.DataFrame]: list of history rows
        """
        hists = []
        for sb in tqdm(self.subbands, desc="Decimating subbands"):
            hist = sb.decimate(
                bins=bins,
                axis=axis,
                use_median=use_median,
            )
            hists.append(hist)
        return hists

    def write(
        self, filename: str | Path, overwrite: bool = False
    ) -> list[pd.DataFrame]:
        """Write the data to a new file

        Args:
            filename (str | Path): The filename to write to
            overwrite (bool, optional): Overwrite the file if it exists. Defaults to False.

        Returns:
            list[pd.DataFrame]: list of history rows
        """
        hists = []
        for sb in tqdm(self.subbands, "Writing subbands"):
            hists.extend(sb.write(filename, overwrite=overwrite))

        return [*hists, history.generate_history_row()]


@dataclass
class SDHDF:
    """An SDHDF data object

    Args:
        filename (Path): Path to the SDHDF file
        in_memory (bool, optional): Load data into memory. Defaults to False.
        parallel (bool, optional): Use dask for parallel processing. Defaults to False.

    Attributes:
        metadata (MetaData): Observation metadata
        beams (list[Beam]): list of beams

    Methods:
        plot_waterfall: Waterfall plot of the data
        plot_spectrum: Spectrum plot of the data
        plot_wide: Plot spectra from all subbands
        print_obs_metadata: list the observation metadata in the file
        write: Write the data to a new file

    """

    filename: Path
    in_memory: bool = False
    parallel: bool = False

    def __post_init__(self):
        self.client = Client() if self.parallel else None
        if self.parallel:
            logger.info(f"Dask dashboard at: {self.client.dashboard_link}")
        self.metadata = MetaData(self.filename)
        self.definition = self.metadata.definition
        with h5py.File(self.filename, "r") as f:
            keys = list(f.keys())
            self.beams = [
                Beam(
                    label=key,
                    filename=self.filename,
                    in_memory=self.in_memory,
                    definition=self.definition,
                    client=self.client,
                )
                for key in keys
                if "beam_" in key
            ]
            for beam in self.beams:
                self.__dict__[beam.label] = beam

    def plot_waterfall(
        self,
        beam: int | str,
        subband: int | str,
        polarization: int = 0,
        flag: bool = False,
        **plot_kwargs,
    ):
        """Waterfall plot of the data

        Args:
            beam (int | str): Beam to select.
            subband (int | str): Subband to select.
            polarization (int, optional): Polarization to select. Defaults to 0.
            flag (bool, optional): Blank flagged data. Defaults to False.
        """
        if isinstance(beam, int):
            beam_object = self.beams[beam]
        elif isinstance(beam, str):
            beam_object = cast(Beam, self.__dict__[beam])
        return beam_object.plot_waterfall(
            subband=subband,
            polarization=polarization,
            flag=flag,
            **plot_kwargs,
        )

    def plot_spectrum(
        self,
        beam: int | str,
        subband: int | str,
        time: int = 0,
        polarization: int = 0,
        flag: bool = False,
        **plot_kwargs,
    ):
        if isinstance(beam, int):
            beam_object = self.beams[beam]
        elif isinstance(beam, str):
            beam_object = cast(Beam, self.__dict__[beam])

        return beam_object.plot_spectrum(
            subband=subband,
            time=time,
            polarization=polarization,
            flag=flag,
            **plot_kwargs,
        )

    def plot_wide(
        self,
        beam: int | str,
        time: int = 0,
        polarization: int = 0,
        flag: bool = False,
        **plot_kwargs,
    ):
        if isinstance(beam, int):
            beam_object = self.beams[beam]
        elif isinstance(beam, str):
            beam_object = cast(Beam, self.__dict__[beam])

        return beam_object.plot_wide(
            time=time,
            polarization=polarization,
            flag=flag,
            **plot_kwargs,
        )

    def print_obs_metadata(self, format: str = "fancy_outline"):
        self.metadata.print_obs_metadata(format=format)

    def print_obs_config(self, format: str = "fancy_outline"):
        self.metadata.print_obs_config(format=format)

    def flag_persistent_rfi(self) -> pd.DataFrame:
        """Flag persistent RFI in all subbands."""

        if "TELESCOPE" in self.metadata.primary_header:
            telescope = self.metadata.primary_header["TELESCOPE"][0]
            logging.info("Using persistent RFI lookup table for Murriyang")
        else:
            logger.warning(
                "No telescope information found in file! Guessing `Parkes`..."
            )
            telescope = "Parkes"
        rfi = flagging.get_persistent_rfi(telescope=telescope)
        for _i, x in tqdm(
            rfi.iterrows(), desc="Flagging persistent RFI", total=len(rfi)
        ):
            for beam in self.beams:
                for sb in beam.subbands:
                    freqs = sb.astronomy_dataset.frequency
                    low_freq, high_freq = x["freq0 MHz"], x["freq1 MHz"]
                    flags = (freqs > low_freq) & (freqs < high_freq)
                    sb.astronomy_dataset["flag"] = sb.astronomy_dataset.flag.where(
                        flags, 1, 0
                    )
        return history.generate_history_row()

    def auto_flag_rfi(
        self,
        sigma=3,
        n_windows=100,
        flag_persistent: bool = True,
    ):
        """Automatic flagging using rolling sigma clipping"""
        if flag_persistent:
            self.flag_persistent_rfi()
        hists = []
        for beam in tqdm(self.beams, desc="Flagging beams"):
            hists.extend(beam.autoflag(sigma=sigma, n_windows=n_windows))

        self.metadata.history.table = pd.concat([self.metadata.history.table, *hists])

    def decimate(self, bins: float | int, axis: str = "frequency", use_median=False):
        """Decimate the data in all subbands.

        Args:
            bins (float | int): If int, the number of channels to bin in an average.
                If float, the desired width of a channel after averaging.
            axis (str, optional): Axis to decimate along. Defaults to 'frequency'.
            use_median (bool, optional): Use median instead of mean. Defaults to False.

        """
        hists = []
        for beam in tqdm(self.beams, desc="Decimating beams"):
            hists.extend(beam.decimate(bins=bins, axis=axis, use_median=use_median))

        self.metadata.history = pd.concat([self.metadata.history, *hists])

    def write(self, filename: str | Path, overwrite: bool = False):
        """Write the SDHDF object to a file.

        Args:
            filename (Path): Filename to write to.
        """
        hists = []
        for beam in tqdm(self.beams, desc="Writing beams"):
            hists.extend(beam.write(filename, overwrite=overwrite))

        self.metadata.history = pd.concat(
            [self.metadata.history, *hists, history.generate_history_row()]
        )
        self.metadata.write(filename, overwrite=overwrite)
