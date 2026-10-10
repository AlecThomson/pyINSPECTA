"""Interoperability with INSPECTA (sdhdf_tools).

Skipped unless an INSPECTA build is available: set ``SDHDF_INSPECTA_BIN`` to
the directory containing the ``sdhdf_*`` programs and ``SDHDF_RUNTIME`` to
INSPECTA's runtime directory.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from conftest import TEST_FILES, data_path

from sdhdf import SDHDF, upgrade

BIN = os.environ.get("SDHDF_INSPECTA_BIN", "")
pytestmark = pytest.mark.skipif(
    not BIN or not os.environ.get("SDHDF_RUNTIME"),
    reason="INSPECTA not available (set SDHDF_INSPECTA_BIN and SDHDF_RUNTIME)",
)


def _run(prog: str, *args, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(Path(BIN) / prog), *map(str, args)],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def _quickdump(path: Path) -> np.ndarray:
    """INSPECTA's view of a file: freq, AA, BB, CR, CI, weight, MJD per row."""
    result = _run("sdhdf_quickdump", "-mjd", path.name, cwd=path.parent)
    rows = [
        line.split()[5:]
        for line in result.stdout.splitlines()
        if line.startswith(f"{path.name} ")
    ]
    assert rows, result.stdout[-500:] + result.stderr[-500:]
    return np.array(rows, dtype=float)


def _ours(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    for sb in SDHDF(path).subbands:
        ds = sb.astronomy_dataset.isel(bin=0)
        data = ds["data"].to_numpy()
        freq = ds["frequency"].to_numpy()
        for t, mjd in enumerate(ds["time"].to_numpy()):
            rows.extend(
                [freq[c], *data[t, :, c], mjd] for c in range(ds.sizes["frequency"])
            )
    return np.array(rows)


@pytest.mark.parametrize("name", TEST_FILES)
def test_inspecta_reads_upgraded_files(name, tmp_path):
    """INSPECTA reads files upgraded to 4.0 exactly as it reads the originals."""
    src = tmp_path / name
    shutil.copy(data_path(name), src)
    up = tmp_path / f"up_{name}"
    upgrade(src, up)
    original, upgraded, ours = _quickdump(src), _quickdump(up), _ours(src)
    # Data, frequencies and times are unchanged. Weights are not compared with
    # the original: without stored weights INSPECTA computes its own default
    # from the band edges, which differs slightly from the channel spacing.
    columns = [0, 1, 2, 3, 4, 6]
    np.testing.assert_allclose(
        upgraded[:, columns], original[:, columns], rtol=1e-6, atol=1e-6
    )
    weights = np.concatenate(
        [
            sb.astronomy_dataset["weights"].to_numpy().ravel()
            for sb in SDHDF(src).subbands
        ]
    )
    np.testing.assert_allclose(upgraded[:, 5], weights, rtol=1e-6)
    np.testing.assert_allclose(original[:, :5], ours[:, :5], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(original[:, 6], ours[:, 5], rtol=0, atol=2e-6)
    describe = _run("sdhdf_describe", up.name, cwd=tmp_path)
    assert describe.returncode == 0
    assert "4.0" in describe.stdout


@pytest.mark.parametrize(
    ("args", "kwargs"), [(("-fav", 4), {"frequency": 4}), (("-tav", 2), {"time": 2})]
)
def test_weighted_averaging_matches_inspecta(args, kwargs, tmp_path):
    """Averaging with flags and weights agrees with sdhdf_modify, and we read its output."""
    sdhdf = SDHDF(data_path("sdhdf_v3.0.hdf"), in_memory=True)
    sb = sdhdf.subbands[0]
    ds = sb.astronomy_dataset
    rng = np.random.default_rng(42)
    flags = rng.uniform(size=ds["flags"].shape) < 0.3
    weights = rng.uniform(0.2, 3.0, ds["weights"].shape).astype(np.float32)
    sb.astronomy_dataset = ds.assign(
        flags=(ds["flags"].dims, flags), weights=(ds["weights"].dims, weights)
    )
    path = tmp_path / "weighted.hdf"
    sdhdf.write(path)

    result = _run("sdhdf_modify", *args, "-e", "mod", path.name, cwd=tmp_path)
    theirs = SDHDF(tmp_path / "weighted.mod.hdf").subbands[0].astronomy_dataset
    assert theirs["data"].dims == ("time", "polarisation", "frequency", "bin"), (
        result.stdout[-300:]
    )

    mine = SDHDF(path)
    mine.average(**kwargs)
    ours = mine.subbands[0].astronomy_dataset
    np.testing.assert_array_equal(theirs["flags"].to_numpy(), ours["flags"].to_numpy())
    np.testing.assert_allclose(
        theirs["weights"].to_numpy(), ours["weights"].to_numpy(), rtol=1e-6
    )
    good = ~ours["flags"].to_numpy()[:, None, :, None]
    good = np.broadcast_to(good, ours["data"].shape)
    np.testing.assert_allclose(
        theirs["data"].to_numpy()[good], ours["data"].to_numpy()[good], rtol=1e-5
    )
