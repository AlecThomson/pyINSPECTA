from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest
from conftest import data_path

from sdhdf import SDHDF

mpl.use("Agg")


@pytest.fixture
def sdhdf() -> SDHDF:
    return SDHDF.from_file(data_path("sdhdf_v4.0.hdf"))


def test_plots_draw_on_given_axes(sdhdf: SDHDF) -> None:
    for plot in (sdhdf.plot_spectrum, sdhdf.plot_waterfall):
        _, ax = plt.subplots()
        assert plot(beam=0, subband=0, ax=ax) is ax
    _, ax = plt.subplots()
    assert sdhdf.plot_wide(beam=0, ax=ax) is ax
    plt.close("all")


def test_waterfall_needs_more_than_one_integration() -> None:
    sdhdf = SDHDF.from_file(data_path("sdhdf_v1.9.3.hdf"))
    assert sdhdf.plot_waterfall(beam=0, subband=0) is None
