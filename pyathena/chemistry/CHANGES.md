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

## 2026-06-13: Pre-Phase-3 refactor

Three parallel tracks that together prepare the chemistry stack for
the Phase 3 explicit-subcycling driver, the Phase 4 cooling/heating
ABCs, and the Phase 6 multi-ion sweep. No public API was removed and
no parity test changed; the work is additive scaffolding.

Plan update (`chemistry-rewrite-plan.md`): ghost species are now a
first-class part of the SpeciesSet schema. The NCRNetwork3 ghost list
has six entries (electron, CI, CII, CO, OI, OII) — replacing the
earlier single-ghost (electron) sketch — to match the
`CoolingOther` algebraic closure in
`tigris-ncr/src/photchem/ncr_rates.hpp:1505-1540`. The GOW17 18-species
clarification (every species an ODE variable, so the ghost list is
empty) was folded in. A follow-up note flags that the electron row is
advected by the hydro step on the C++ side but not yet on the Python
side; the driver layer will own the closure when it lands.

- `pyathena.chemistry.species.SpeciesSet` — evolved / ghost split
  added. The constructor takes optional `evolved_names_in` /
  `ghost_names_in` tuples; both must partition `names` exactly (no
  overlap, no missing entries). Three name-tuple + index-array views
  are materialised: `evolved_names` / `evolved_idx` / `n_evolved` and
  `ghost_names` / `ghost_idx` / `n_ghost`. Default when no partition
  is provided: every species is evolved, none are ghost (back-compat
  with the Phase 2 factories). New factory:
  `SpeciesSet.ncr3_with_ghosts()` returns the 9-species NCR layout
  (HI, HII, H2 evolved; electron, CI, CII, CO, OI, OII ghost).
- `pyathena.chemistry.networks.base.NetworkBase` — `fill_ghosts(state)`
  contract: pure algebra, idempotent, mutates only `state.x[ghost_idx]`,
  zero allocation. The base class provides a no-op default for
  networks with no ghost species (e.g., a future fully-tracked GOW17).
  Class-level `evolved` / `ghost` declarations sit alongside `species`
  / `walk_order` for solver consumption. A no-op
  `allocate_scratch(state)` hook is also provided; concrete networks
  override it to register multi-buffer scratch via
  `state.alloc_scratch`.
- `pyathena.chemistry.networks.ncr3.NCRNetwork3.fill_ghosts` —
  implements the `Chem_flag == 0` branch of `CoolingOther`:
  `x_OII = x_HII * xOstd * Z_g`, `x_CII = xCstd * Z_g` (saturated),
  `x_CO = 0` (floored at `x_floor`), `x_OI = max(xOstd * Z_g - x_OII
  - x_CO, x_floor)`, `x_CI = max(xCstd * Z_g - x_CII - x_CO, x_floor)`,
  `x_e = max(x_HII + x_CII + x_OII, x_floor)`. The constants
  `x_C_std = 1.6e-4` and `x_O_std = 3.2e-4` mirror `NCRRates::kxCstd`
  / `kxOstd` on the C++ side. TODO marker in the docstring points at
  the chemistry-rewrite-plan §9 follow-up: the saturated `x_CII`
  should be replaced with the full `GetxCII(chi_FUV, chi_CI, xi_CR,
  Z_d)` prescription once the radiation-field state inputs land in
  ChemState, and CO needs the GOW17 chain in Phase 9.
  `evaluate_CD` / `closure` / `electron_fraction` now read indices
  from `state.species.idx` (not `self.species.idx`) so the network
  operates uniformly across the 9-species layout and any reduced
  legacy set that omits the metal ghosts.
- `pyathena.chemistry.state.ChemState` — schema additions for the
  Phase 3 driver. `chi_bands: Tuple[str, ...]` names the rows of
  `chi`; defaults to the 3-band NCR convention `('FUV', 'LW', 'EUV')`
  for `nfreq=3` and a positional `('chi_0', ...)` layout otherwise.
  Callers read radiation rows via `state.chi_for(band_name)` instead
  of positional indices. `scratch: Dict[str, np.ndarray]` replaces the
  earlier named scratch fields (`C` / `D` / `Lambda` / `Gamma`
  / `metal_CT` / `regime_tag` / `nsub_est` are gone); networks
  register their buffers via the `allocate_scratch(state)` hook and
  the hot path uses `state.get_scratch(name)`. `from_grid` accepts
  policy kwargs `network=`, `solver=`, `thermo=`, `cooling=`,
  `opacity=`, `radiation=` and stamps `'ClassQualname@__version__'`
  strings into `policy_versions`; unsupplied roles keep the
  `'__none__'` sentinel for back-compat. The `ne` property delegates
  to the explicit electron row when present and falls back to the
  positive-charge sum otherwise.
- `pyathena.chemistry.state.assert_no_alloc(allow=0)` — context
  manager that wraps numpy's array constructors (`empty` / `zeros` /
  `ones` / `full` / `array` / `copy` / `*_like`) and asserts no more
  than `allow` allocations happen inside the block. Process-wide
  (not thread-safe) — used by Phase 3 solver tests to confirm the
  hot path does not allocate.
- `pyathena.chemistry.config.ChemistryConfig` — `solver_type: str` was
  replaced by `solver: SolverSpec`. `solver_type` remains as a
  read-only `@property` aliasing `self.solver.name`; the legacy string
  `'explicit'` is mapped to the registry name `'explicit_subcycling'`
  via `_LEGACY_SOLVER_NAMES`. `network_params` is a free-form mapping
  that the future `NCRNetwork3PlusIons16` reads; flat NCR3 keys
  (`xCstd`, `xOstd`, `temp_mu_floor`, `x_floor`) are auto-mirrored
  into it for back-compat. A new `SOLVER_REGISTRY` dict + class
  decorator `@register_solver('name')` populates the registry at
  solver-module import time; the registry is empty at config-import
  time (Phase 3 fills it).
- `pyathena.chemistry.rates.{photx,rec_rate,ci_rate,ct_rate}` —
  rate-class audit follow-ups. All four classes accept an
  `interp_mode: InterpMode = InterpMode.kExact` constructor argument
  (the table modes raise `NotImplementedError`). Per-call
  `np.where(c1 & c2 [& c3])` table scans were replaced with O(1)
  dict lookups on `(Z, N)` (or `(Z, N, M)` for the Badnell DR / RR
  tables); `RecRate.get_dr_rate`'s inner DR sum is now one broadcast
  `np.exp(-Ed[:, None] / T[None, :]).sum(0)`; `ChargeTransferRate`
  drops `copy.deepcopy(T)` in favour of `np.asarray(T, dtype=float)`.
  New strip-vectorised accessors `PhotX.get_sigma_table`,
  `RecRate.get_rec_rate_table`, `CollIonRate.get_ci_rate_table` return
  `(n_species, ncell)` arrays in the
  `species_set.evolved_names + species_set.ghost_names` traversal
  order. Charge-transfer strip table deferred to Phase 6 (CT carries
  ion-pair structure awkward to vectorise outside the multi-ion
  sweep). `InterpMode` and `Regime` are now re-exported from
  `pyathena.chemistry.__init__`.

Merge-point fix landed in this entry: the strip-table accessors
originally looked for `species_set.evolved` / `species_set.ghost` (no
suffix), which do not exist on the SpeciesSet exposed by track 1 — the
canonical attribute names are `evolved_names` / `ghost_names`. The
three accessors now share a `_ordered_ions(species_set)` helper that
reads the name-tuples and resolves each name back to its `Ion` instance
through `species_set.idx`, falling back to `species_set.ions` for sets
that declare no partition. Three new tests in
`tests/chemistry/test_rates_strip_tables.py` exercise the
`ncr3_with_ghosts` path so the fallback path cannot silently re-cover
this regression.

The Phase 3 explicit-subcycling driver and the Phase 4
cooling / heating ABCs will be the first consumers of every change in
this entry: the driver registers itself under
`'explicit_subcycling'` via `@register_solver`, sizes its scratch
through `NetworkBase.allocate_scratch`, reads radiation rows through
`ChemState.chi_for`, and validates allocation-free hot paths via
`assert_no_alloc`. Phase 6 multi-ion work reuses the
`evolved_names` / `ghost_names` partition to size the strip rate
tables produced by `get_*_table`.

Validation: 406 passed, 17 skipped in `tests/chemistry/ +
tests/microphysics/` (up from 354 / 17 in Phase 2; +52 new tests,
no regressions).
