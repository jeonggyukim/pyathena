# pyathena.microphysics

**Frozen.** This package is in maintenance-only mode while a
greenfield rewrite lands under `pyathena.chemistry`. New features
go in the new package; only bug fixes are accepted here, and each
bug fix is mirrored to the new package as well.

Status of the migration is tracked in
`pyathena/chemistry/CHANGES.md`. The full design and roadmap is in
`~/Dropbox/Projects/tigris-notes/docs-claude/pyathena/chemistry-rewrite-plan.md`.

A module is removed from this package when:

1. the corresponding `pyathena.chemistry` module exists and passes
   its own unit tests,
2. the parity test under `tests/chemistry/parity/` has been green
   for 30 days of CI on `master`, and
3. a cross-repo grep finds no remaining imports of the old symbol.

Until those three conditions are met, the module here stays put and
is the reference for the new implementation.

## What lives in this package

Rate coefficients (`ci_rate.py`, `rec_rate.py`, `ct_rate.py`,
`photx.py`), the cooling monolith (`cool.py`), the photochemistry
driver (`photchem.py`), abundance and ionisation-equilibrium
helpers, and the legacy TIGRESS cooling function (`coolftn.py`,
`cool_gnat12.py`).

## What has already moved out

(Updated as relocations land.)

- *Phase 0.5* (planned): `coolants/` → `pyathena.chemistry.coolants/`
  and `chianti_v11/` → `pyathena.chemistry.tables.chianti_v11/`.
  Backward-compat re-exports with `DeprecationWarning` stay in this
  package for one deprecation window.
