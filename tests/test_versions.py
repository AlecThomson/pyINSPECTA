from __future__ import annotations

from importlib import resources

from sdhdf import SDHDF


def test_sdhdf_versions():
    versions = {
        "1.9.3",
        "2.0",
        "2.1",
        "2.2",
        "3.0",
        "4.0",
    }

    for version in versions:
        with resources.as_file(resources.files("sdhdf.data.tests")) as test_data:
            fname = f"sdhdf_v{version}.hdf"
            file_path = test_data / fname

            my_sdhdf = SDHDF(file_path)

            expected_version = str(version)
            if float(version[:3]) < 2.0:
                expected_version = str(2.0)
            elif float(version[:3]) == 2.2:
                expected_version = str(2.9)
            assert str(my_sdhdf.metadata.version) == expected_version, (
                f"Version mismatch for {fname}"
            )
