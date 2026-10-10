"""Command-line tools"""

from __future__ import annotations

import argparse
from pathlib import Path

from sdhdf.constants import LATEST_VERSION
from sdhdf.logger import set_verbosity
from sdhdf.sdhdf import upgrade


def upgrade_main(argv: list[str] | None = None) -> None:
    """Convert SDHDF files of any version to the latest version."""
    parser = argparse.ArgumentParser(
        description=f"Convert SDHDF files of any version to SDHDF {LATEST_VERSION}."
    )
    parser.add_argument("input", type=Path, help="Input SDHDF file")
    parser.add_argument("output", type=Path, help="Output SDHDF file")
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite the output file"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Print progress")
    args = parser.parse_args(argv)
    if args.verbose:
        set_verbosity("INFO")
    upgrade(args.input, args.output, overwrite=args.overwrite)


if __name__ == "__main__":
    upgrade_main()
