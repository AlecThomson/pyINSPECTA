# pyINSPECTA

[![Actions Status][actions-badge]][actions-link]
[![Codecov Status][codecov-badge]][codecov-link]
[![Documentation Status][rtd-badge]][rtd-link]

[![PyPI version][pypi-version]][pypi-link]

<!-- [![Conda-Forge][conda-badge]][conda-link] -->

[![PyPI platforms][pypi-platforms]][pypi-link]

<!-- [![GitHub Discussion][github-discussions-badge]][github-discussions-link] -->

<!-- prettier-ignore-start -->
[codecov-link]:             https://codecov.io/gh/AlecThomson/pyINSPECTA
[codecov-badge]:            https://codecov.io/gh/AlecThomson/pyINSPECTA/graph/badge.svg?token=7EARBRN20D
[actions-badge]:            https://github.com/AlecThomson/pyINSPECTA/actions/workflows/ci.yml/badge.svg
[actions-link]:             https://github.com/AlecThomson/pyINSPECTA/actions
[pypi-link]:                https://pypi.org/project/sdhdf/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/sdhdf
[pypi-version]:             https://img.shields.io/pypi/v/sdhdf
[rtd-badge]:                https://readthedocs.org/projects/pyinspecta/badge/?version=latest
[rtd-link]:                 https://pyinspecta.readthedocs.io/en/latest/?badge=latest


<!-- prettier-ignore-end -->

Python tooling for reading, interactive with, and writing SDHDF files.

<!-- SPHINX-START -->
## Installation

Installing can be done with `pip` (we recommend using `uv`):

Stable version from PyPI:

```bash
pip install sdhdf
```

Latest version from git:

```bash
pip install git+https://github.com/AlecThomson/pyINSPECTA.git
```

## Usage

```python
import astropy.units as u
from sdhdf import SDHDF

my_sdhdf = SDHDF.from_file("myfile.hdf")
band = my_sdhdf.beams[0].subbands[0].astronomy_dataset  # an xarray.Dataset

processed = my_sdhdf.auto_flag_rfi().average(time=4, frequency=0.5 * u.MHz)
processed.write("processed.hdf")
```

Processing methods return a new `SDHDF` and leave the original alone. Averaging
takes either a number of samples (`frequency=8`) or a width with units
(`frequency=0.5 * u.MHz`, `time=10 * u.s`).

Files of every SDHDF definition version (1.9 to 4.0) are read the same way, using
the SDHDF 4.0 names. Files with attributes damaged by older processing software
(e.g. placeholder values such as `NOT FOUND`, or missing dimension labels) are
also supported. Files are always written as SDHDF 4.0, so reading and writing a
file upgrades it. To do this in one go:

```python
from sdhdf import upgrade

upgrade("old.hdf", "new.hdf")
```

Data are read lazily with `dask`, which uses all of your cores by default, so
there's no need to set up a dask `Client`. Writing always uses local threads.

Full documentation on [Read the Docs](https://pyinspecta.readthedocs.io/en/latest/).

## Contributing

Contributions are welcome! Please be sure to install the developer tools and pre-commit hooks:

```bash
git https://github.com/AlecThomson/pyINSPECTA.git
cd pyINSPECTA
pip install -e .[dev]
pre-commit install
```
