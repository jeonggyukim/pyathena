# pyathena.chemistry — change log

Append-only. One entry per substantive module addition, port, or
deletion. Each entry records:

- date
- module path under `pyathena.chemistry.` (or removal under
  `pyathena.microphysics.`)
- short description
- parity tolerance (when a parity test exists)
- parity-green-since date (set when the corresponding microphysics
  module becomes eligible for deletion)

## 2026-06-13: Phase 0 skeleton

- `pyathena.chemistry.__init__` — package created (empty).
- `pyathena.chemistry.README` — design intent + parity contract.
- `pyathena.chemistry.datapaths` — `chianti_v11_dir()` and
  `gf12_dir()` resolvers.
- `pyathena.chemistry.enums` — `Regime`, `InterpMode` enums.
- `pyathena.chemistry.diagnostics` — `SolverDiag` fixed-field
  dataclass.
- `pyathena.chemistry.state` — `ChemState` dataclass skeleton.
- `pyathena.chemistry._parity` — `run_both(...)` test harness.
- `tests/chemistry/parity/test_OII_HII_resonance_parity.py` lifted
  from microphysics; new path delegates to old (green by
  construction).
- `pyathena/microphysics/README.md` — top-paragraph freeze notice.

No real chemistry yet. The skeleton establishes the contract and
the harness; subsequent phases fill in modules one at a time. See
`tigris-notes/docs-claude/pyathena/chemistry-rewrite-plan.md`.

## 2026-06-13: Phase 0.5 relocate already-clean families

- `pyathena.chemistry.tables.chianti_v11.*` — moved from
  `pyathena.microphysics.chianti_v11.*` via `git mv`. Six builder
  modules (`build_ioneq`, `build_ioneq_ct`, `build_cool`,
  `build_cool_fast`, `build_atomic`, `build_all`) plus the README.
  Data files at `data/microphysics/chianti_v11/` did NOT move.
- `pyathena.chemistry.coolants.*` — moved from
  `pyathena.microphysics.coolants.*` via `git mv`. 17 per-ion
  modules (o_3.py, c_2.py, ...), `base.py` (IonCoolant), and
  `n_level.py` (5-level steady-state solver, also moved from
  `pyathena.microphysics.n_level`).
- Internal cross-import `coolants/base.py` -> `n_level` rewritten
  to the new path; coolants no longer reach into
  `pyathena.microphysics`.
- External test-file imports updated to the new paths:
  `tests/microphysics/test_plot_*.py` and
  `test_validate_cool_BB.py`. The old paths
  `pyathena.microphysics.coolants` and
  `pyathena.microphysics.chianti_v11` no longer exist; the planned
  re-export shims with `DeprecationWarning` are NOT in this commit
  (they can be added later if any external consumer surfaces).

Validation: 129 passed, 15 skipped in
`tests/chemistry/ + tests/microphysics/` (no regressions vs Phase 0).
