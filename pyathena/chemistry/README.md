# pyathena.chemistry

A typed, policy-based rewrite of `pyathena.microphysics` that mirrors
the tigris-ncr C++ Phase D / E template structure. Replaces the old
package over a deprecation window. See
`~/Dropbox/Projects/tigris-notes/docs-claude/pyathena/chemistry-rewrite-plan.md`
for full design and roadmap.

## Intent

Two goals:

1. **Clean architecture.** Six abstract base classes — `NetworkBase`,
   `SolverBase`, `CoolingBase`, `HeatingBase`, `OpacityBase`,
   `RadiationBase` — plus `ThermoPolicy` for `mu(x)` and energy /
   temperature conversion. The driver is one constructor call;
   swapping chemistry networks, ODE solvers, cooling channels, or
   opacity models is one keyword change.

2. **Reference implementation for tigris-ncr C++.** Every policy in
   this package corresponds 1:1 to a Phase E C++ template parameter or
   helper function. The C++ port reads this Python as the answer-key
   when implementing the SIMD / vectorised production version.

## Parity contract

`pyathena.microphysics` is **frozen** but kept alive for backward
compatibility. Each leaf module in `chemistry/` is paired with a
parity test in `tests/chemistry/parity/` that runs both old and new
on the same input and asserts agreement at a documented tolerance.

A microphysics module is deleted when:

- the corresponding chemistry module exists and passes its own unit
  tests,
- the parity test has been green for 30 days of CI on master,
- a cross-repo grep finds no remaining imports of the old symbol.

No calendar deadlines. The grep + parity gate is the trigger.

## Layout

```
pyathena/chemistry/
  __init__.py
  README.md          # this file
  CHANGES.md         # append-only per-module status
  datapaths.py       # resolve data files (CHIANTI v11, GF12, ...)
  enums.py           # Regime, InterpMode — single source for C++ codegen
  diagnostics.py     # SolverDiag fixed-field dataclass
  state.py           # ChemState dataclass + factories
  _parity.py         # test-only: import both packages and diff
  # Phase 0.5 — moved from microphysics:
  coolants/          # per-ion 5-level coolant modules
  tables/            # CHIANTI builders writing into data/microphysics/chianti_v11/
  # Subsequent phases (see rewrite plan):
  rates/             # Verner+96, Badnell RR/DR, Voronov CI, Kingdon-Ferland CT
  networks/          # NetworkBase + concrete implementations
  solvers/           # SolverBase + concrete implementations
  cooling/           # per-channel CoolingBase implementations
  heating/           # per-channel HeatingBase implementations
  opacity/           # OpacityBase + per-band gas + dust
  radiation/         # 1D radial, plane-parallel propagators
  thermo/            # ThermoPolicy + concrete (NCR mu formula)
  driver.py          # ChemistryDriver composing six policy slots
```

## Porting checklist (per Phase)

When porting a microphysics module to `chemistry`:

1. Create the new module under `chemistry/` matching the ABC of its
   policy slot.
2. Lift the existing unit tests to `tests/chemistry/` and adapt to the
   new API.
3. Add a parity test under `tests/chemistry/parity/` that runs both
   old and new on the same input. Document tolerance in the test
   docstring.
4. Add a `CHANGES.md` entry describing the port date, parity
   tolerance, and deletion criterion.
5. Once parity is green for 30 days on master, the leaf module can
   be deleted from `microphysics/` (open a separate PR for the
   deletion).
