"""The SDHDF format description, loaded from ``sdhdf/data/definition.toml``."""

from __future__ import annotations

import functools
import sys
from dataclasses import dataclass
from importlib import resources

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

Candidates = dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class Definition:
    """Names and conventions of the SDHDF format across versions.

    Each ``Candidates`` mapping gives, for a canonical (SDHDF 4.0) role, the
    names it has had, newest first. See ``definition.toml`` for details.
    """

    groups: Candidates
    file_tables: Candidates
    configuration_tables: Candidates
    beam_tables: Candidates
    band_tables: Candidates
    astronomy_datasets: Candidates
    calibrator_datasets: Candidates
    column_renames: dict[str, dict[str, str]]
    required_columns: dict[str, dict[str, float | str]]
    table_descriptions: dict[str, str]
    attribute_descriptions: dict[str, str]
    dimension_aliases: dict[str, str]
    polarisation_types: dict[str, tuple[str, ...]]
    default_polarisation_types: dict[int, str]
    telescope_aliases: dict[str, str]

    def renames_for(self, table: str) -> dict[str, str]:
        """Legacy -> 4.0 column renames for a table kind."""
        if table in self.column_renames:
            return self.column_renames[table]
        return self.column_renames.get(table.removeprefix("calibrator_"), {})


def _candidates(table: dict[str, list[str]]) -> Candidates:
    return {role: tuple(names) for role, names in table.items()}


@functools.cache
def definition() -> Definition:
    """The SDHDF format description (read once)."""
    text = (resources.files("sdhdf.data") / "definition.toml").read_text(
        encoding="utf-8"
    )
    data = tomllib.loads(text)
    tables = data["tables"]
    datasets = data["datasets"]
    return Definition(
        groups=_candidates(data["groups"]),
        file_tables=_candidates(tables["file"]),
        configuration_tables=_candidates(tables["configuration"]),
        beam_tables=_candidates(tables["beam"]),
        band_tables=_candidates(tables["band"]),
        astronomy_datasets=_candidates(datasets["astronomy"]),
        calibrator_datasets=_candidates(datasets["calibrator"]),
        column_renames=data["column_renames"],
        required_columns=data["required_columns"],
        table_descriptions=data["table_descriptions"],
        attribute_descriptions=data["attribute_descriptions"],
        dimension_aliases=data["dimension_aliases"],
        polarisation_types={k: tuple(v) for k, v in data["polarisation_types"].items()},
        default_polarisation_types={
            int(k): v for k, v in data["default_polarisation_types"].items()
        },
        telescope_aliases=data["telescope_aliases"],
    )
