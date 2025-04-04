# pyINSPECTA

[![Actions Status][actions-badge]][actions-link]
[![Codecov Status][codecov-badge]][codecov-link]
[![Documentation Status][rtd-badge]][rtd-link]

[![PyPI version][pypi-version]][pypi-link]

<!-- [![Conda-Forge][conda-badge]][conda-link] -->

[![PyPI platforms][pypi-platforms]][pypi-link]

<!-- [![GitHub Discussion][github-discussions-badge]][github-discussions-link] -->

<!-- SPHINX-START -->

<!-- prettier-ignore-start -->
[codecov-link]:             https://codecov.io/gh/AlecThomson/pyINSPECTA
[codecov-badge]:            https://codecov.io/gh/AlecThomson/pyINSPECTA/graph/badge.svg?token=7EARBRN20D
[actions-badge]:            https://github.com/AlecThomson/pyINSPECTA/workflows/CI/badge.svg
[actions-link]:             https://github.com/AlecThomson/pyINSPECTA/actions
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/pyINSPECTA
[conda-link]:               https://github.com/conda-forge/pyINSPECTA-feedstock
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/AlecThomson/pyINSPECTA/discussions
[pypi-link]:                https://pypi.org/project/pyINSPECTA/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/pyINSPECTA
[pypi-version]:             https://img.shields.io/pypi/v/pyINSPECTA
[rtd-badge]:                https://readthedocs.org/projects/pyINSPECTA/badge/?version=latest
[rtd-link]:                 https://pyinspecta.readthedocs.io/en/latest/?badge=latest

<!-- prettier-ignore-end -->

Python classes for reading, interactive with, and writing SDHDF files.

## Installation

Installing can be done with `pip` (we recommend using `uv`):

Stable version from PyPI:

```bash
pip install pyINSPECTA
```

Latest version from git:

```bash
git clone https://bitbucket.csiro.au/scm/cpda/sdhdf_tools.git
cd sdhdf_tools/python
pip install -e .
```

## Contributing

Contributions are welcome! Please be sure to install the developer tools and pre-commit hooks:

```bash
git clone https://bitbucket.csiro.au/scm/cpda/sdhdf_tools.git
cd sdhdf_tools/python
pip install -e .[dev]
pre-commit install
```
