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

## 2026-06-13: Phase 1 leaf-rate ports

Verbatim ports of four leaf rate modules from
`pyathena.microphysics` into `pyathena.chemistry.rates`. The only
non-cosmetic change per module is the data-file path lookup, which
now resolves through `pyathena.chemistry.datapaths._DATA_ROOT`
instead of a `__file__`-relative `pathlib` walk.

- `pyathena.chemistry.rates.photx` — Verner+1996 photoionization
  cross sections (`PhotX` class, `get_sigma_pi_H2` helper).
  Parity test: `tests/chemistry/parity/test_photx_parity.py`
  (33 tests, rtol=1e-12, atol=0).
- `pyathena.chemistry.rates.ci_rate` — collisional ionization
  rates (`CollIonRate`, `CollIonRateCHIANTI`). Parity test:
  `tests/chemistry/parity/test_ci_rate_parity.py` (18 tests,
  rtol=1e-12, atol=0). `CollIonRateCHIANTI` is ported verbatim
  but not exercised by the parity harness (optional ChiantiPy
  dependency).
- `pyathena.chemistry.rates.rec_rate` — radiative + dielectronic
  recombination rates (`RecRate`, `RecRateCHIANTI`). Parity
  test: `tests/chemistry/parity/test_rec_rate_parity.py`
  (85 collected, 83 passed, 2 skipped for N=0 fully-stripped
  ions where the dispatcher raises by design, rtol=1e-12,
  atol=0). `RecRateCHIANTI` ported verbatim but not exercised
  by the parity harness.
- `pyathena.chemistry.rates.ct_rate` — charge-transfer rates
  (`ChargeTransferRate`). Parity test:
  `tests/chemistry/parity/test_ct_rate_parity.py` (34 tests,
  rtol=1e-12, atol=0). UGA He / H2 data files exist on disk
  but are not yet read by the current loader; nothing to port.

The four ported classes plus `get_sigma_pi_H2` are re-exported from
`pyathena.chemistry.rates.__init__` so callers can do
`from pyathena.chemistry.rates import PhotX, RecRate, CollIonRate,
ChargeTransferRate`.

The corresponding `pyathena.microphysics.{photx,ci_rate,rec_rate,
ct_rate}` modules are NOT deleted in this commit. They remain in
place under the 30-day-green-CI rule and will be removed in a
follow-up once both code paths have stayed parity-green across a
full CI cycle.

## 2026-06-13: Phase 2 policy scaffolding

First chemistry policies, the species inventory they share, and the
ChemState factory that ties them together. No parity vs the legacy
microphysics solver yet -- that arrives in Phase 3 alongside the
explicit subcycling driver.

- `pyathena.chemistry.species` -- `Ion` (frozen 5-field dataclass:
  element, Z, N, charge, name), `SpeciesSet` (ordered, frozen
  collection with per-species vectors `charges`, `n_per_particle`,
  `mass_per_particle`, `is_electron`). Two factories:
  `SpeciesSet.minimal_HI_HII_H2()` (NCR3 layout: HI, HII, H2,
  electron) and `SpeciesSet.ncr3_plus_helium()` (Phase 6 prep).
  Exposes `h2_index` / `electron_index` for ThermoPolicy lookups.
- `pyathena.chemistry.config` -- `ChemistryConfig` dataclass, one
  field per `GetOrAdd*` key in `[photchem_ncr]`. Defaults match the
  tigris-ncr C++ side byte-for-byte where determinable; unknown keys
  fall through to `extra`. Factories:
  `ChemistryConfig.from_athinput(path)`,
  `ChemistryConfig.from_dict(d)`, `ChemistryConfig()`. Test count: 22
  (`tests/chemistry/test_species.py` + `test_config.py`).
- `pyathena.chemistry.networks.base.NetworkBase` -- abstract
  network policy. Methods: `evaluate_CD(state, out_C, out_D)` (writes
  the semi-implicit rate split `dx_i/dt = C_i - D_i x_i` into
  caller-owned buffers; never allocates), `closure(state)` (species
  floor + algebraic conservation), `electron_fraction(state)`. Class
  attributes: `species` (SpeciesSet), `walk_order`, capability flags
  `kSupportsStrips` / `kNeedsJacobian`.
- `pyathena.chemistry.networks.ncr3.NCRNetwork3` -- 3-species
  H I / H II / H2 NCR network. Rates ported from
  `pyathena.microphysics.cool` (Janev kcoll_H, Ferland alpha_rr_H,
  Weingartner alpha_gr_H, kgr_H2, Glover&MacLow xi_coll_H2). Hot/cold
  transition mirrors the C++ `TEMP_HOT0 = 2e4`, `TEMP_HOT1 = 3.5e4`
  with a sigmoid blend on the grain channel. Branch-free over the
  ncell axis. Test count: 12
  (`tests/chemistry/test_ncr3_network.py`).
- `pyathena.chemistry.thermo.base.ThermoPolicy` -- abstract thermal
  policy. Four methods (`mu`, `T_to_e`, `e_to_T`, `pressure`) all
  write into caller-owned `(ncell,)` buffers. `gamma` is a class
  attribute. K_B and m_H cached as module floats at import time.
- `pyathena.chemistry.thermo.ncr.NCRThermo` -- concrete policy
  matching PhotochemistryNCR::GetTemperature in
  `tigris-ncr/src/photchem/photchem.hpp:295-298`:
  `mu = mu_hyd / (1 + A_He - x_H2 + x_e)` with `mu_hyd = 1 + 4 A_He`,
  `A_He = 0.0955`, `gamma = 5/3`. Test count: 10
  (`tests/chemistry/test_thermo.py`).
- `pyathena.chemistry.state.ChemState.from_grid(r, nH, T, species,
  *, Z_g=1.0, Z_d=1.0, A_He=0.0955)` -- factory that allocates a
  strip-shaped state and validates it. `policy_versions` seeded with
  `{'network': '__none__', 'thermo': '__none__'}`; the driver fills
  in concrete tags later. The grid coordinate `r` and the helium
  abundance `A_He` are attached as non-schema attributes for
  downstream plotting / thermo use.
- `pyathena.chemistry.state.ChemState.ne` -- now computes
  `nH * (q+ . x)` where `q+ = max(charges, 0)` (positive-ion charge
  sum). Returns zero on neutral states and `nH` on fully ionised
  states, exactly the algebraic limits used by the Phase 3 driver.
- `tests/chemistry/test_state_factories.py` -- 12 tests covering the
  factory shapes, the `ne` property, and the `validate()` mismatch
  paths.
- `tests/chemistry/test_forward_euler_smoke.py` -- end-to-end smoke
  driver that initialises a 10-cell isobaric state at T = 8000 K,
  nH = 1 cm^-3 with x = (1, 0, 0, 0); runs 100 small forward-Euler
  steps under a CR ionization rate of `xi_CR = 2e-16 s^-1`; asserts
  `x_HI + x_HII + 2 x_H2 = 1` to 1e-10, all x >= 0, x_HII > 0 in at
  least one cell, mu / pressure finite and positive. Not a parity
  test; the algebraic shape `dx/dt = C - D x` is what is being
  exercised.

Phase 6 will replace `NCRNetwork3` with `NCRNetwork3PlusIons16` for
the 10-ion metal sweep (C I, C II, N I, N II, O I, O II, O III, S I,
S II, S III); the SpeciesSet schema already supports it via
`ncr3_plus_helium` (a future `ncr3_plus_ions16` factory will add the
remaining ion rows).

Validation: 354 passed, 17 skipped in `tests/chemistry/ +
tests/microphysics/` (up from 341 / 17 in Phase 1; +13 new tests, no
regressions).
