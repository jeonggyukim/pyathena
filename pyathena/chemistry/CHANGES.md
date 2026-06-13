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

## 2026-06-13: CIE-lumped pool ghost design + Phase 6 stub

Plan integration pass folding the CIE-lumped pool architecture into
`chemistry-rewrite-plan.md` and reserving the corresponding code path
under `pyathena.chemistry.networks.ncr3_plus_ions16` so the Phase 3
explicit-subcycling driver and the Phase 4 cooling tables land
against a stable downstream consumer.

Plan changes (`chemistry-rewrite-plan.md`):

- §4a rewritten into four subsections. §4a.1 names the three KINDS
  of ghost (charge-sum, prescription, CIE-lumped pool) so each
  concrete network's ghost composition can be read off as a
  combination of the three. §4a.2 introduces the
  `element_groups: Tuple[Tuple[str, int, Tuple[str, ...]], ...]`
  declaration the driver consumes to size per-element CIE-pool
  tables and emit one CIE-lumped pool ghost per heavy element. §4a.3
  states the closure formulation: per substep, per element, if
  `sum(x_q for q in evolved_ion_names) > (1 - x_high_frac(T)) *
  x_std * Z_g` the evolved rows are renormalised by
  `x_low_avail / sum_evolved` before `fill_ghosts` writes the pool
  row, with a worked example for O at T = 1e5 K. §4a.4 is the
  per-network catalogue: NCRNetwork3 keeps `element_groups = ()`
  (its C / O tracking is too coarse for the pool concept to add
  anything); NCRNetwork3PlusIons16 declares four pools (C, N, O, S)
  and 19 species total (13 evolved + 6 ghost); NCRNetwork3PlusHe and
  GOW17Network keep their existing characterisations and are
  annotated to make the kind composition explicit.
- §5 Phase 4 adds `build_cie_high_pool.py` as a prerequisite for
  the Phase 6 pool ghosts. The builder reads CHIANTI v11 CIE
  ionisation fractions on the standard log-T grid and writes
  `data/chemistry/cie_high_pool_{element}.nc` containing
  `x_high_frac(T)`, `q_high_mean(T)` (population-weighted mean
  charge), and `Lambda_high(T)` for each element in
  NCRNetwork3PlusIons16.element_groups. The CHIANTI invariant
  `x_high_frac + sum_q ioneq_q = 1` is checked at every grid point;
  asymptotic limits validate Lambda_high.
- §5 Phase 6 documents the hot-regime fast path. The solver
  pre-scans every strip cell for `x_low_avail(T) < eps` across all
  elements in `element_groups` and, on cells flagged hot, skips the
  evolved-ion implicit-Euler updates entirely. The evolved rows
  freeze, `fill_ghosts` writes the pool rows directly from the
  table, and cooling reads `Lambda = sum_elem Lambda_high(T) *
  x_std * Z_g`. The regression suite covers cells on both sides of
  the threshold (default eps = 1e-3) so the fast path and the slow
  path agree at the boundary.
- §9 (C++ porting mapping) rewrote the `x_e` advection row: the
  current advected scalar becomes obviously wrong at Phase D because
  `sum_q q * x_q` must equal `x_e` exactly at every chemistry-step
  entry — the HII-region path only masks this today because `x_HII`
  is the only positive ion in scope. The required follow-up is to
  drop the advected scalar at Phase D entry and compute `x_e` from
  the positive-charge sum inside `fill_ghosts(state)` (kind 1 of
  §4a.1). The Python reference computes it this way from Phase 2
  onward.

Code changes:

- `pyathena.chemistry.networks.base.NetworkBase` —
  `element_groups: ClassVar[Tuple[Tuple[str, int, Tuple[str, ...]], ...]] = ()`
  added alongside the existing `evolved` / `ghost` declarations. The
  driver consumes it to size CIE-pool tables and drive the
  per-element closure renormalisation; concrete networks that need
  no pool (e.g., NCRNetwork3) keep the default empty tuple.
- `pyathena.chemistry.networks.ncr3_plus_ions16.NCRNetwork3PlusIons16` —
  Phase 6 planning stub. Declares `evolved` (13 rows), `ghost`
  (6 rows: e, CO, x_high_C, x_high_N, x_high_O, x_high_S),
  `element_groups` (the four-tuple in §4a.2 of the plan), and
  `walk_order` (ground-state-first per element). All five method
  bodies — `__init__`, `evaluate_CD`, `closure`,
  `electron_fraction`, `fill_ghosts`, `allocate_scratch` — raise
  `NotImplementedError('Phase 6')`. The module docstring captures
  the layout, the closure formulation (mirrors plan §4a.3), and the
  hot-regime fast path (mirrors plan §5 Phase 6), so an architect
  reading the stub gets the same picture as one reading the plan.
  No test imports the class today; the stub is reachable only
  through the explicit module path.

Validation: 419 passed, 4 skipped in `tests/chemistry/ +
tests/microphysics/` (no regressions). The pass / skip totals shift
vs the previous entry because Phase 0.5 relocated several tests out
of microphysics into chemistry; net delta is 0.

## 2026-06-13: Phase 3 explicit subcycling solver + driver

Strip-first synchronous explicit subcycling solver and the
`ChemistryDriver` that owns its lifecycle. The Python solver mirrors
the planned C++ Phase C `SynchronousSemiImplicitSweep` semantics (one
`dt_sub` shared across the strip, one `nsub` count) rather than the
current per-cell adaptive `ExplicitSubcyclingSolver::SolveCell`
semantics; this lets the same algorithm port directly to the C++
NCRStrip path later and keeps the substep loop allocation-free under
`assert_no_alloc`.

- `pyathena.chemistry.solvers.explicit_subcycling.ExplicitSubcyclingSolver`
  -- registered under `'explicit_subcycling'` via
  `@register_solver`. Constructor takes a `ChemistryConfig`, a
  `NetworkBase` (NCRNetwork3 in practice), an `NCRThermo`, and an
  optional cooling policy (`None` for Phase 3). All hot-path scratch
  is named under the `solver:` prefix and registered via
  `allocate_scratch(state)`. The substep loop runs the (C, D) split,
  estimates the strip-MIN substep length, semi-implicit Euler T step
  with rejection-and-halve up to 3 retries, recompute (C, D) at the
  post-T state, implicit-Euler chemistry update on the evolved rows,
  closure. The temperature update kernel and the implicit-Euler
  chemistry kernel live in `_substep_kernels.py` so they unit-test
  without instantiating the solver class.
- `pyathena.chemistry.solvers._substep_kernels` -- two pure functions
  (`implicit_euler_update`, `semi_implicit_T_update`) that take
  caller-owned `out=` / `tmp=` buffers. Allocation-free by
  construction.
- `pyathena.chemistry.solvers._stubs` -- `CoolingStub`,
  `OpacityStub`, `RadiationStub` placeholder policies the driver
  consumes until Phase 4+ ship the real ones. `RadiationStub` copies
  fixed FUV / photo-rate scalars onto `state.chi_FUV` etc. so
  `NCRNetwork3.evaluate_CD` reads non-zero rates without reaching for
  a radiation transport policy.
- `pyathena.chemistry.driver.ChemistryDriver` -- owns the
  network / solver / thermo / cooling / opacity / radiation policy
  instances. `setup(state)` calls `network.allocate_scratch` then
  `solver.allocate_scratch`. `step(dt, state)` runs
  `radiation.update`, `opacity.update`, `cooling.update`, then
  `solver.step(dt, state)` in that order. The solver name in
  `config.solver` is resolved through `SOLVER_REGISTRY` when the
  caller omits `solver=...`.
- `pyathena.chemistry.solvers.__init__` -- re-exports
  `ExplicitSubcyclingSolver` and triggers the registry side effect
  on import.
- `tests/chemistry/test_solvers_explicit_subcycling.py` -- 8 tests
  covering registry lookup, scratch idempotency, allocation-free
  hot path (under `assert_no_alloc(allow=0)` with a stub network so
  the rate-coefficient `np.where` / `np.exp` allocations inside
  `NCRNetwork3.evaluate_CD` do not poison the contract), H mass
  closure after `step(dt)`, single-substep implicit-Euler update vs
  a hand-rolled `x_new = (x + C dt) / (1 + D dt)` reference at
  rtol=1e-8, and the rejection-path retry counter triggered by a
  swinging-cooling stub.
- `tests/chemistry/test_driver_smoke.py` -- 3 tests covering the
  end-to-end driver on a 1024-cell HII-region strip, the
  diagnostics snapshot helper, and the registry-driven solver
  resolution path.
- `tests/chemistry/parity/test_ncr_cooling_lambda_parity.py` -- 8
  tests parameterised over the NCR cooling catalog (coolCII, coolOI,
  coolOII, coolHIion, coolHI, coolHISmith21, coolH2G17) on a
  (50, 10) (T, nH) grid plus the summed-Lambda test, all at
  rtol=1e-10. Phase 3 delegates the new path to
  `pyathena.microphysics.cool` so the test passes by construction;
  Phase 4 rebinds the new-path module reference to the
  `pyathena.chemistry.coolants` package and the tolerance band kicks
  in for real.

The Phase 3 solver is not bit-stable with the merged C++
`ExplicitSubcyclingSolver::SolveCell` because the C++ side runs a
per-cell `while (t_done < dt)` loop with per-cell `dt_sub`, whereas
the Python solver runs a strip-wide loop with `dt_sub = min_strip(...)`.
The Phase C `SynchronousSemiImplicitSweep` solver in tigris-ncr will
use the strip-MIN convention too; that is the C++ target this Python
reference will eventually compare against.

Validation: 438 passed, 4 skipped in `tests/chemistry/ +
tests/microphysics/` (up from 419 / 4 in the previous entry; +19 new
tests, no regressions).

## 2026-06-13: Solver substep updates T/mu, not T (operator-splitting fix)

Phase 3 follow-up. The semi-implicit temperature kernel in
`pyathena.chemistry.solvers._substep_kernels` was updating the
temperature `T` directly. The conserved variable across the substep
under standard operator splitting is `T/mu`, not `T`: the cooling
sub-step holds `mu` fixed while it updates the thermal energy, then
the chemistry sub-step holds `T` fixed while it changes the species
composition (and therefore `mu`). Storing `T` and recomputing pressure
through the post-chemistry `mu` silently adds or removes internal
energy proportional to `Delta mu / mu * T` at every substep boundary;
the gas heats whenever species recombine and cools whenever they
ionise, with no corresponding entry in the cooling channel ledger.

The tigris-ncr C++ solver
(`tigris-ncr/src/photchem/ncr_solver.hpp::UpdateTemperature`, lines
466 - 495) and the mini-RAMSES neq cooling driver
(`mini-ramses/cooling/neq_cooling_module.f90::cool_step`, lines
500 - 575, `T2 = T/mu`) both update `T/mu` for this reason. Aligning
the Python reference removes the discrepancy and is a prerequisite
for the Phase 4 cooling-channel parity tests.

Changes:

- `pyathena.chemistry.solvers._substep_kernels` -- the temperature
  kernel renamed `semi_implicit_T_update` -> `semi_implicit_temp_mu_update`
  with the same closed-form arithmetic but new arg names and
  docstring. Inputs: `temp_mu` (= T/mu), `net_cool`,
  `d_net_cool_d_temp_mu`, `inv_heat_cap_per_temp_mu`, `dt`, and the
  output / scratch buffers. The cooling-channel derivative now lives
  in temp_mu-space (`mu * d(net_cool)/dT`) because that is what the
  semi-implicit denominator needs.
- `pyathena.chemistry.solvers.explicit_subcycling.ExplicitSubcyclingSolver`
  -- substep loop refactored to mirror the C++ flow. Scratch slots
  renamed: `solver:T_old` / `solver:T_new` ->
  `solver:temp_mu_old` / `solver:temp_mu_new`, and a new
  `solver:mu_at_entry` slot snapshots mu at substep start.
  `solver:inv_heat_capacity` (which had a mu factor) is replaced by
  `solver:inv_heat_cap_per_temp_mu` (= `(gamma - 1) / (n_H * mu_hyd *
  k_B)`, mu-independent). `solver:d_net_cool_dT` is renamed
  `solver:d_net_cool_d_temp_mu`. After the chemistry sub-step the
  solver rescales `state.T = mu_new * temp_mu_new` so the
  substep-invariant `T/mu` survives a change of species composition.
  The dt_sub estimator's cooling timescale formula now reads
  `inv_heat_cap_per_temp_mu * |net_cool| / temp_mu` (algebraically
  identical to the old expression up to a mu factor that cancels with
  the new heat-capacity formula).
- Cooling-policy contract update: Phase 4 cooling policies write
  `solver:net_cool` (unchanged) and `solver:d_net_cool_d_temp_mu`
  (was `solver:d_net_cool_dT`). The temp_mu-space derivative is
  `mu * d(net_cool)/dT`; policies that internally know
  `d(net_cool)/dT` should scale by `mu` before storing.
- Tests: `tests/chemistry/test_solvers_explicit_subcycling.py`
  scratch-slot assertions, and the `_SwingingCooling` stub writing
  the renamed derivative slot, are both updated.

Numerical behaviour: with the current Phase 3 cooling stub
(`net_cool = 0`) the dT update is zero so `state.T` end-of-substep
matches the previous solver bit-for-bit. The behavioural change
shows up only once a real cooling policy lands (Phase 4) and `mu`
drifts during chemistry -- where the new path stays consistent with
the C++ and RAMSES references and the old path would have leaked
internal energy at the substep boundary.

Validation: 438 passed, 4 skipped (matches the previous baseline; no
regressions introduced by the rewrite).
