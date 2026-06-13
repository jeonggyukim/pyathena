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
