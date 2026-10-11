# pyINSPECTA style guide

- KISS: try the simpler way before anything complex.
- No module-level constants unless they are shared across modules (those go in `sdhdf/constants.py`). Keep variables local to where they are used. Format details (names across SDHDF versions, renames) live in `sdhdf/data/definition.toml`.
- Private (`_`) functions are for small helpers only.
- The data objects (`SDHDF`, `Beam`, `SubBand`, `MetaData`) are dataclasses with methods. Read files with `from_file` classmethods. Processing methods return new objects and leave the original unchanged. Algorithms are plain functions on xarray objects. No `ThingDoer.do_thing()` classes.
- Typing: `mypy --strict` for `sdhdf/`; tests are typed but not strict.
- Function names describe what the function does.
- Units:
  - SDHDF does not fix units. Values read from a file keep the file's unit in a `units` attribute and are never converted on read.
  - Put units in names only for internal variables whose unit the code sets (e.g. `channel_width_hz`).
  - Public physical arguments take `int | Quantity`: an int is a number of samples, a Quantity is a width or duration converted against the file's units. A plain float raises `TypeError`.
- Writing uses local threads only (open h5py files can't be sent to other processes).
- Comments are minimal and say why, not how. No em/en dashes, no "this, not that" phrasing, no ASCII-art or banner comments.
- Public functions have short Google-style docstrings (summary, Args, Returns). Small helpers get one line or none.
- Docs and notebook prose are plain and short.
- Tests use pytest. Run `pre-commit run --all-files` before pushing.
