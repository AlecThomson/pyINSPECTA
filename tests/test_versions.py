from __future__ import annotations

from importlib import resources

from sdhdf import SDHDF


def version_str_to_float(version_str: str) -> float:
    """Convert version string to float.

    e.g.
        "2.1.0" -> 2.1
        "2.0.1" -> 2.0
        "2.0" -> 2.0
        "2" -> 2.0

    Args:
        version_str (str): Version string to convert.

    Returns:
        float: Converted version as a float.
    """
    return float(".".join(version_str.split(".")[0:2]))


def test_sdhdf_versions():
    versions = {
        "1.9.3",
        "2.0",
        "2.1",
        "2.2",
        "3.0",
        "4.0",
    }

    for version_str in versions:
        with resources.as_file(resources.files("sdhdf.data.tests")) as test_data:
            fname = f"sdhdf_v{version_str}.hdf"
            file_path = test_data / fname

            my_sdhdf = SDHDF(file_path)
            my_version_float = my_sdhdf.metadata.version

            version_float = version_str_to_float(version_str)

            if version_float <= 2.0:
                version_float = 2.0
            elif version_float <= 2.1:
                version_float = 2.1
            elif version_float <= 2.9:
                version_float = 2.9

            assert my_version_float == version_float, (
                f"Version mismatch: {my_version_float} != {version_float}"
            )
