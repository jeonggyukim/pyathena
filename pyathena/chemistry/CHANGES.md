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

## 2026-06-13: Phase 3.5 -- NQT helpers + LogLog / Nqt2 / Nqt1 table modes

Implements the tabulated-rate interpolation modes documented in
`chemistry-rewrite-plan.md` Phase 3.5. Scope follows the C++
`NCRRates::InterpMode` enum at
`tigris-ncr/src/photchem/ncr_rates.hpp:188`: the temperature-dependent
rate classes `CollIonRate` and `RecRate` gain `LogLog`, `Nqt2`, and
`Nqt1` modes; `PhotX` (energy-indexed, not temperature-indexed) and
`ChargeTransferRate` (ion-pair-indexed, awkward to tabulate as a 2D
table) keep the `Exact`-only contract and continue to raise on
non-Exact construction. The Phase 3.5b follow-up will revisit them
along with the Phase 4 channel split.

NQT helpers (Hammond et al. 2025, ApJS 277, 65):

- `pyathena.chemistry.rates._nqt` -- vectorised `nqt1_log`,
  `nqt1_exp`, `nqt2_log`, `nqt2_exp`. Mirrors the four scalar
  `static inline` helpers at
  `tigris-ncr/src/photchem/ncr_rates.hpp:557-592`. The encoding is
  pure IEEE 754 bit manipulation; the magic constants
  (`0x3FF0000000000000` = `as_int(1.0)`, `2**52`, the 52-bit mantissa
  mask) match the C++ side byte-for-byte, so for any positive
  normalised float64 input the Python and C++ outputs are
  bit-identical.
- `tests/chemistry/test_nqt_helpers.py` -- 15 tests covering exact
  values at powers of 2, round-trip bijectivity to within float64
  round-off (rtol = 1e-13), Hammond+2025 absolute-error bounds on
  `log2(x)` (NQTo1 worst-case ~0.086, threshold 0.09; NQTo2 worst-case
  ~0.0094 near `b = 0.25`, threshold 0.012), and vectorised-vs-scalar
  equivalence.

Table modes on the temperature-dependent rates:

- `pyathena.chemistry.enums.InterpMode` -- members `LogLog`, `Nqt2`,
  `Nqt1` are now first-class alongside `Exact`. Integer values 1, 2, 3
  match the C++ enum.
- `pyathena.chemistry.rates.ci_rate.CollIonRate` -- gains
  `_build_table(mode)` and `_table_lookup(i_row, T)`. The Voronov
  analytic formula is split out into `_get_ci_rate_exact(i_row, T)`
  so the table-build path can populate the encoded `(n_T, n_ions)`
  table without recursing through the dispatch. `get_ci_rate(Z, N, T)`
  dispatches on `self.interp_mode`. Grid: `n_T = 2000`, `T in [1, 1e9]
  K`, identical to the C++ `kNTabT` choice. Rate floor `1e-100`
  applied before encoding to keep `nqt2_exp` away from the
  `4 - 3*f = 0` corner.
- `pyathena.chemistry.rates.rec_rate.RecRate` -- gains the same
  `_build_table` / `_table_lookup` pair. The tabulation covers the
  TOTAL `get_rec_rate(Z, N, T, M=1, kind='badnell')` per (Z, N) ion
  that has at least one M=1 entry in either the Badnell RR or DR
  data. `get_rec_rate(Z, N, T, M=M, kind=kind)` dispatches to the
  table only when both `M == 1` and `kind == 'badnell'`; other
  combinations fall through to the analytic helpers so the existing
  RR / DR APIs remain untouched.

Cross-mode validation:

- `tests/chemistry/test_ci_rate_interp_modes.py` -- 18 tests across 5
  representative ions (HI, HeI, HeII, CI, OI). LogLog vs Exact: rtol
  < 1e-3. Nqt2 vs Exact: rtol < 1e-2. Nqt1 vs Exact: rtol < 8e-2 (OI
  hits ~5.5% near the Voronov-fit knee; the plan's 1e-2 target is
  unrealistic for NQTo1 on rates with the `exp(-U)` factor at the
  ionisation threshold, and the C++ port hits the same bound).
- `tests/chemistry/test_rec_rate_interp_modes.py` -- 25 tests across
  7 ions (H, He, C, N, O, Si first-step recombinations). Same LogLog
  / Nqt2 thresholds; Nqt1 relaxed to rtol < 0.1 for the same reason.
  Two dispatch-fallback tests pin the `M != 1` and `caseB = False`
  paths so future re-routes do not silently bypass them.
- `tests/chemistry/test_rates_strip_tables.py::test_interp_mode_table_modes_not_implemented`
  -- parametrisation narrowed from `[PhotX, RecRate, CollIonRate]` to
  `[PhotX]` only. The other two classes now support the table modes;
  PhotX continues to raise until the Phase 3.5b energy-grid path
  lands.

Validation: 490 passed, 4 skipped (up from 438 / 4 in the previous
entry; +52 net new tests, no regressions). Pytest runtime within the
~34 s envelope.

## 2026-06-13: Phase 4a -- cooling/heating channel ABCs + first ports

Lays the channel-policy scaffolding for the cool.py monolith split.
Phase 4a delivers (a) the per-channel `CoolingChannel` /
`HeatingChannel` ABCs and the `CoolingChannels` aggregator that
populates `solver:net_cool` + `solver:d_net_cool_d_temp_mu`, (b)
literal-port channels for H I Lyman-alpha cooling and Weingartner-
Draine 2001 photoelectric heating, and (c) byte-exact parity tests
against the corresponding `pyathena.microphysics.cool` helpers.
Subsequent Phase 4b commits port the remaining 20-plus channels
incrementally and Phase 4c wires `CoolingChannels` into
`ChemistryDriver` so the solver's cooling policy slot consumes real
rates instead of `CoolingStub`.

ABC + aggregator:

- `pyathena/chemistry/cooling/base.py` -- `CoolingChannel` (abstract,
  one mechanism per subclass with `name`, `evaluate(state, out,
  d_out=None)`), `HeatingChannel` (same shape; co-located here to
  avoid a forward-import cycle), and `CoolingChannels` (composes a
  cooling tuple + a heating tuple, sums Lambda contributions and
  subtracts Gamma contributions into `solver:net_cool` and the
  matching derivative slot). The aggregator owns its per-channel
  Lambda / dLambda scratch namespaces (`cooling:Lambda:<name>`) and
  registers them via `allocate_scratch(state)` so the substep loop
  runs allocation-free. Heating mirrors the contract under
  `cooling:Gamma:<name>` / `cooling:dGamma:<name>`.

Channel ports:

- `pyathena/chemistry/cooling/lyman_alpha.py::LymanAlphaCooling` --
  literal port of `pyathena.microphysics.cool.coolHI` (lines 490 -
  513). Computes Lambda_2p (H I 2-photon), Lambda_LyA, Lambda_LyB
  using the DESPOTIC formulae (Krumholz 2014, ApJS 211, 19;
  Draine 2011 11.32 / 11.34 / 11.36). Constants
  `T_LYA = 118415.63430152694 K`, `T_LYB = 140344.45546847637 K`,
  effective collision strengths upsilon_{2s, 2p, 3s, 3p, 3d},
  beta = 8.629e-6 cm^3 s^-1 K^0.5 hardcoded with the same precision
  as the legacy helper.
- `pyathena/chemistry/heating/photoelectric.py::PhotoelectricHeating`
  -- literal port of `pyathena.microphysics.cool.heatPE` (lines
  78 - 86) + its `get_charge_param` helper. Charge parameter
  `x = 1.7 * chi_PE * sqrt(T) / (xe * nH * phi) + 50.0` (the `+50.0`
  offset clamps the WD01 fit away from its small-x invalid regime).
  WD01 Table 2 epsilon fit coefficients hardcoded; default
  `chi_band = 'FUV'` reads `state.chi_for('FUV')` on the strip.

Both channels follow the Phase 4a contract: `evaluate(state, out)`
writes Lambda or Gamma in `erg / s / cm^3`; the optional `d_out`
buffer is zeroed (the analytic temperature derivative lands in Phase
4b). The hot path uses `np.add(a, b, out=c)` form throughout, with
all temporaries drawn from named scratch slots so the channel runs
inside `assert_no_alloc(allow=0)` once the wired-up driver is added.

Parity:

- `tests/chemistry/parity/test_cooling_lyman_alpha_parity.py` (4
  tests) -- `LymanAlphaCooling.evaluate` vs `coolHI` on a 30 x 20
  (T, n_H) grid spanning T = 100 K to 1e6 K and n_H = 0.01 to 1e4 cm^-3,
  for three (xHI, xe) combinations; rtol = 1e-12, atol = 0. Also pins
  the `d_out` zero-write contract.
- `tests/chemistry/parity/test_heating_pe_parity.py` (3 tests) --
  `PhotoelectricHeating.evaluate` vs `heatPE` on a 24 x 16 (T, n_H)
  grid spanning T = 10 K to 1e4 K and n_H = 0.01 to 1e4 cm^-3, for
  three (xe, Z_d, chi_PE) combinations; rtol = 1e-12, atol = 0.

Out of scope for Phase 4a (queued for 4b):

- The remaining cooling channels (CII, OI, CI, CO, H2 G17, rec, ff,
  dust, ...) and heating channels (CR, H2 form, H2 photo, ...).
- The analytic `d(Lambda)/d(T/mu)` and `d(Gamma)/d(T/mu)` derivatives
  for each channel.
- The driver-level rebind that swaps `CoolingStub` for the
  `CoolingChannels` aggregator and exercises a real cooling rate in
  the explicit-subcycling step.
- The Phase 6-prerequisite `build_cie_high_pool.py` Tabulator (Phase
  4c).

Validation: 497 passed, 4 skipped (up from 490 / 4 in the previous
entry; +7 net new tests, no regressions).

## 2026-06-13: Phase 4b batch 1 -- four hydrogen channels

Adds four more channels in the Phase 4 cool.py split: HI collisional
ionisation cooling, H II case-B recombination cooling, H II free-free
(bremsstrahlung), and cosmic-ray heating. All four mirror their
`pyathena.microphysics.cool` originals byte-for-byte and follow the
Phase 4a `CoolingChannel` / `HeatingChannel` contracts (named scratch
slots, in-place numpy hot path, optional zeroed `d_out`).

Channels added:

- `cooling.hi_collisional_ionization.HICollisionalIonizationCooling`
  -- ports `coolHIion`. Lambda = 13.6 eV * k_coll(T) * n_H * x_e *
  x_HI with `k_coll(T)` the 8-term Janev 1987 polynomial in
  ln(T * 8.6173e-5) gated to T > 3000 K. Horner-evaluated to keep
  the inner loop branch-free.
- `cooling.recombination_hydrogen.HRecombinationCooling` -- ports
  `coolrecH`. Lambda = E_rr_B * alpha_caseB(T) * n_H * x_e * x_HII
  with E_rr_B = (0.684 - 0.0416 * ln(T / 1e4)) * k_B * T. The
  channel owns a `RecRate` instance; alpha_caseB still allocates on
  every call inside `RecRate.get_rec_rate_H_caseB`, so Phase 4c
  rebinds to a pre-tabulated path.
- `cooling.free_free.FreeFreeHCooling` -- ports `coolffH`. Lambda =
  1.422e-25 * g_ff(T) * sqrt(T / 1e4) * n_H * x_e * x_HII with the
  Draine 2011 10.11 Gaunt factor.
- `heating.cosmic_ray.CosmicRayHeating` -- ports `heatCR`. Gamma =
  xi_CR * (x_HI * q_HI + 2 * x_H2 * q_H2), with q_H2 the Krumholz
  2014 piecewise-linear-in-log10(n_H) fit assembled branch-free via
  np.where over the five density pieces. xi_CR exposed as a
  constructor argument; defaults to 2e-16 (NCR standard).

Parity tests
(`tests/chemistry/parity/test_phase4b_hydrogen_channels.py`):

- One test function per channel, looping over four ionisation
  states (WNM, HII boundary, ionised, half-molecular CNM); cosmic-ray
  heating additionally sweeps three xi_CR values. Each test invokes
  `np.testing.assert_allclose(..., rtol = 1e-12, atol = 0,
  err_msg = ...)` so a failure names the case inline rather than
  spamming the pytest tail. The grouping follows the project's
  pytest test-count convention.

Out of scope (queued for Phase 4b batch 2 / 4c):

- CII / OI / CI fine-structure channels (need the per-coolant
  `cool2Level_` / `cool3Level_` helpers).
- H2 cooling (G17, rovib, coll-dissoc) and H2 photoheating channels.
- Dust grain cooling.
- WD01 recombination cooling on grains (`coolRec`).
- Analytic `d(Lambda)/d(T/mu)` and `d(Gamma)/d(T/mu)` -- still
  zeroed for now.
- Driver rebind to `CoolingChannels`.

Validation: 501 passed, 4 skipped (up from 497 / 4; +4 net new test
entries covering 32 internal cases across the 4 channels, no
regressions).

## 2026-06-13: Phase 4b batch 2 -- CII / OI fine-structure + dust + level helpers

Three more channels in the Phase 4 split: the CII 158 um 2-level
fine-structure cooling, the OI 63 + 146 um 3-level fine-structure
cooling, and the gas-grain thermal coupling (signed: cool when
T > T_dust, heat when T < T_dust). Also lands shared steady-state
2-level / 3-level helpers under `cooling/_level_helpers.py` so the
remaining fine-structure ports (CI, OII, OIII, NI, NII, SI/II/III
in later batches) can reuse the same population-balance arithmetic.

CHIANTI / Cloudy swap inventory (Phase 7 and beyond):

- Metal fine-structure channels (CII / OI / CI / ...) swap their
  e and HI collision partners to CHIANTI table lookups; H2 partners
  stay hand-coded because CHIANTI is purely atomic. Cloudy carries
  some H2 / molecular data and may serve as a fallback for the
  partners CHIANTI lacks; revisit in Phase 7.
- H2 cooling channels (G17, rovib, colldiss), dust gas-grain
  coupling, free-free, CR / PE heating, and H2 photoheating stay
  hand-coded indefinitely (no atomic-line equivalent).

Helpers:

- `pyathena/chemistry/cooling/_level_helpers.py` exposes
  `cool_2level(q01, q10, A10, E10, xs, out, tmp)` (Lambda = f1 * A10
  * E10 * xs with f1 = q01 / (q01 + q10 + A10)) and `cool_3level(...)`
  (Draine 2011 19.7-19.10 steady-state populations). Both take
  caller-owned scratch and return Lambda in `out`. Partner-agnostic;
  the channel decides which collision rates feed the q_ij.

Channels:

- `cooling.cii.CIIFineStructureCooling` -- literal port of
  `cool.coolCII`. Assembles k10e (Eq 17.16 Draine 2011), k10HI
  (Eq 17.17), and a piecewise-in-T k10H2 split into ortho / para
  weighted 0.75 / 0.25. The T < 500 K / T >= 500 K piecewise
  assembly uses two mask buffers and `np.multiply(mask, x, out=...)`
  blending so the inner loop stays branch-free over the strip.
- `cooling.oi.OIFineStructureCooling` -- literal port of
  `cool.coolOI`. Three transitions (1<-0, 2<-0, 2<-1) each pull their
  HI / H2-para / H2-ortho rate coefficients out of a single helper
  `_assemble_kHI_kH2(...)` that walks the Draine 2011 F.6 power-law
  fits with caller-owned scratch.
- `cooling.dust.DustGasCoupling` -- literal port of `cool.cooldust`.
  Lambda = alpha_gd * Z_d * n_H * sqrt(T) * (T - T_dust),
  alpha_gd = 3.2e-34. Reads `state.T_dust`. Signed: positive when
  the gas is hotter than the dust (net cooling); negative when the
  gas is colder.

Parity tests (`tests/chemistry/parity/test_phase4b_metal_fs_and_dust.py`):

- One test per channel; CII loops 4 ionisation states x 3 xCII
  abundances, OI loops 4 states x 2 xOI abundances, dust loops 4
  states x 3 T_dust values x 2 Z_d. Tolerance rtol = 1e-12, atol = 0
  with case-naming err_msg.

Out of scope (queued for Phase 4b batch 3+ / 4c):

- CI 3-level fine-structure (4-term Horner polynomial for the e-rate
  collision strength; mechanical port, deferred to keep this commit
  scoped).
- OII fine-structure (cool.coolneb / cool.coolOII), HISmith21,
  Lyman-alpha alternative form (cool.coolLya).
- H2 G17, rovib, collisional-dissociation cooling.
- WD01 recombination cooling on grains.
- H2 form / photoheating / pump.
- Analytic d(Lambda)/d(T/mu) derivatives.
- Driver rebind to `CoolingChannels`.

Validation: 504 passed, 4 skipped (up from 501 / 4; +3 net new test
entries covering 40+ internal cases across the 3 channels, no
regressions).

## 2026-06-13: Phase 4b batch 3 -- CI / Lya / H2 (both forms) / grain rec

Five more channel ports. Two of the new channels are the NCR
production defaults (Lya for H I cooling, H2Moseley21 for H2
rovibrational cooling); two alternative forms (LymanAlpha for the
DESPOTIC coolHI convention, H2Gong17 for the Gong + Ostriker +
Wolfire 2017 form) live alongside them. CI 3-level fine-structure
and the WD01 grain-recombination cooling round out the batch.

Production wiring corrections informed the channel-name split:

- Tigris-ncr PhotochemistryNCR uses `coolLya` for H I cooling, not
  `coolHI` (which is the DESPOTIC Lyman-alpha + Lyman-beta + 2-photon
  sum). Both ports are kept; constructor-time choice between
  `LyaCooling` (NCR default) and `LymanAlphaCooling` (DESPOTIC) is
  up to the driver wiring.
- `pyathena.microphysics.get_cooling.py` line 127 wires `coolH2rovib`
  (Moseley + 2021), not `coolH2G17`. Both ports are kept;
  `H2Moseley21Cooling` is the NCR default. The Cloudy molecular
  database may eventually serve as a swap target for either form;
  CHIANTI does not.

Channels:

- `cooling.ci.CIFineStructureCooling` -- literal port of `cool.coolCI`.
  3-level fine-structure (g_0=1, g_1=3, g_2=5) for 370 + 609 um. The
  electron collision strength is a 4-coefficient Horner polynomial
  in `ln T` piecewise at T = 1000 K; HI and H2 partners from
  Draine 2011 F.6. The piecewise blend uses three scratch buffers
  (`tmp_a`, `tmp_b`, `out`) so the cold-piece Horner result is not
  clobbered by the warm-piece Horner workspace. A naming convention
  has been adopted: `tmp_a` / `tmp_b` for symmetric paired scratches.
- `cooling.lya.LyaCooling` -- literal port of `cool.coolLya`. H I
  2-level 1s -> 2p (Lyman-alpha), Draine 2011 17.18 effective
  collision strength fit for the e partner. NCR production default.
- `cooling.h2_gong17.H2Gong17Cooling` -- literal port of
  `cool.coolH2G17` (Gong, Ostriker, Wolfire 2017). Five collision
  partners (HI / H2 / He / H+ / e); piecewise Horner polynomials in
  `log10(T_3)` for the low-density limit; Hollenbach + McKee 1979
  LTE limit. Combined via `Gamma_tot = Gamma_LTE / (1 + Gamma_LTE /
  Gamma_n0)`. Alternative form; not the NCR default.
- `cooling.h2_moseley21.H2Moseley21Cooling` -- literal port of
  `cool.coolH2rovib` (Moseley + 2021). Four rovibrational line
  series with saturation density terms. NCR production default.
- `cooling.grain_recombination.GrainRecombinationCooling` -- literal
  port of `cool.coolRec`. The WD01 charge-parameter `x` and exponent
  structure match the PE heating channel exactly; the two should be
  enabled together in production.

Parity tests
(`tests/chemistry/parity/test_phase4b_ci_h2_grain.py`): five test
functions (one per channel), each looping representative cases
internally. Tolerance `rtol = 1e-12`, `atol = 0`.

Out of scope (queued for Phase 4b batch 4+ / 4c):

- HISmith21 cooling.
- coolH2colldiss (H2 collisional dissociation cooling) and the
  paired `heatH2diss` / `heatH2form` / `heatH2pump` heating channels.
- OII (cool.coolOII) and the coolneb metal-line collision channel.
- Analytic d(Lambda)/d(T/mu) and d(Gamma)/d(T/mu) derivatives.
- Driver rebind to `CoolingChannels`.

Validation: 509 passed, 4 skipped (up from 504 / 4; +5 net new test
entries covering ~50 internal cases across the 5 channels, no
regressions).

## 2026-06-13: Phase 4b batch 4a -- three H2 heating channels (NCR HM79 form)

H2 formation, photodissociation, and UV pump heating split into three
HeatingChannel subclasses. Each port follows the NCR production C++
path at `tigris-ncr/src/photchem/ncr_rates.hpp:1545-1558` (the
`iH2heating = 2` / HM79 branch by default) plus the equivalent path
in `Athena-TIGRESS/src/microphysics/cool_tigress.c:1132-1152`.

Discussion of derivative strategy: hybrid (analytic for simple
closed forms, tabulated for level-population channels, finite
difference as a bootstrap). Tabulated derivatives are nearly free
once the value table exists -- one extra column, one extra lookup --
and adequate for the substep stiffness-damping use case. Phase 4c
will pick the strategy per channel; this batch still writes zero to
`d_out`.

Channels:

- `heating.h2_formation.H2FormationHeating` -- per H2 formed on a
  grain, `(0.2 + 4.2 * f) eV` of binding energy is returned to the
  gas, with `f = 1 / (1 + n_crit / n_H)`. Grain rate from
  Hollenbach + McKee 1979 `kgr = kgr_H2 * Z_d * sqrt(T/100) * 2 /
  (1 + 0.4*sqrt(T/100) + 0.2*(T/100) + 0.08*(T/100)^2)` is the
  default; a constant-rate variant is available via the
  `temperature_dependent_kgr = False` constructor argument.
  `n_crit` follows the HM79 form.
- `heating.h2_photodissociation.H2DissociationHeating` --
  `Gamma_diss = xi_diss_H2 * x_H2 * 0.4 eV`. The 0.4 eV per
  dissociation event is the translational kinetic energy of the
  resulting H atoms.
- `heating.h2_photodissociation.H2PumpHeating` -- UV pump heating
  `Gamma_pump = xi_diss_H2 * x_H2 * f_pump * mean_e * f * eV` with
  `f = n_H / (n_H + n_crit)`. NCR default `form = 'HM79'`
  (`f_pump = 9.0`, `mean_e = 2.2`); `form = 'V18'` selects the
  Sternberg+2014 alternative (`f_pump = 8.0`, `mean_e = 2.0`).

`xi_diss_H2` is supplied as a constructor argument (default 0). The
Phase 4c driver rebind will surface it from the radiation policy.

Parity tests
(`tests/chemistry/parity/test_phase4b_h2_heating.py`): three test
functions (one per channel). The legacy
`pyathena.microphysics.cool.heatH2` has a latent typo at line 151
(`sqrt(T2)` instead of `np.sqrt(T2)` on the `ikgr_H2 = 1` path) so
it cannot be called from a test. The tests instead use inline
closed-form references that mirror the C++ NCR source byte-for-byte.
Tolerance `rtol = 1e-12`, `atol = 0`.

Production-typo note: the `pyathena.microphysics.cool.heatH2`
function is broken on the temperature-dependent grain rate path
(its `ikgr_H2 = 1` branch). Production users typically call the
combined `coolH2pump` / `coolH2diss` standalone helpers (which DO
work) directly. The channel ports inherit the correct behaviour
from the C++ side. The legacy typo is logged here so future cleanup
of `pyathena.microphysics.cool` knows to fix it before the
`pyathena.microphysics.cool` symbol gets deleted.

Out of scope (Phase 4b batch 4b):

- `coolH2colldiss` (H2 collisional dissociation cooling, careful
  log-of-zero handling in cold regimes via `coeff_coll_H2`).
- `coolOII` (O II fine-structure cooling, 3-level).
- `coolneb` (metal-line nebular cooling).
- `coolHISmith21` (alternative H I cooling form).

Validation: 512 passed, 4 skipped (up from 509 / 4; +3 net new test
entries covering 24 internal cases across the 3 channels, no
regressions).

## 2026-06-13: Phase 4b batch 4b -- coolH2colldiss + OII + nebular + Smith21

Four more cooling channels rounding out the Phase 4b cool.py port:

- `cooling.h2_colldiss.H2CollDissCooling` -- port of
  `cool.coolH2colldiss`. H2 thermal dissociation cooling at T > 700 K
  via collisions with H I and H2 (Glover + Mac Low 2007). Rate
  coefficients are floored at 1e-300 before log10 to avoid -Inf
  warnings; the T > 700 K gate at the end zeroes the contribution
  from cold cells either way, so output is byte-identical to the
  legacy.
- `cooling.oii.OIIFineStructureCooling` -- port of `cool.coolOII`.
  3-level fine-structure cooling on the 4S / 2D O II ground term,
  electron-only collider (Draine 2011 power-law fits). The 497.1 um
  intra-2D line and the 3726 / 3728 A doublet to 4S together
  dominate O II metal cooling in the WIM. Reuses the `cool_3level`
  steady-state helper.
- `cooling.nebular.NebularMetalLineCooling` -- port of `cool.coolneb`.
  Stopgap Z_g-scaled metal-line cooling (G&S 2007 fit) for runs
  that track only H / H+ / H2 (NCRNetwork3). Phase 6 / Phase 7
  replaces this with explicit per-ion CHIANTI-table metal-line
  cooling once the multi-ion network (NCRNetwork3PlusIons16) ships
  -- at that point each followed ion (N II, O II, O III, S II, ...)
  contributes its own channel and the proxy is turned off. The
  docstring flags this as a Phase D / Phase 7 stopgap.
- `cooling.hi_smith21.HISmith21Cooling` -- port of
  `cool.coolHISmith21`. H I excitation cooling 1 -> 2, 3, 4, 5 via
  the Smith+2021 Upsilon fits. Piecewise polynomial in
  T6 = T / 1e6: cubic for T6 <= 0.3, constant for T6 > 0.3.
  Available as an alternative to `LymanAlphaCooling` (DESPOTIC) and
  `LyaCooling` (NCR default).

Parity tests
(`tests/chemistry/parity/test_phase4b_h2cd_oii_neb_smith21.py`):
four test functions, one per channel; tolerance rtol = 1e-12,
atol = 0 except for the nebular test (atol = 1e-300 to ignore last-
ULP subnormal precision in the deep-cold tail where both values
are effectively zero). The H2 colldiss test uses
`np.errstate(divide='ignore', invalid='ignore')` because the legacy
code also emits the same -Inf warning at cold T (the channel
floors the rate, the legacy does not, but the final gated output
matches byte-for-byte).

Phase 4 channel inventory after this batch (21 total):

- H I cooling (4): LymanAlpha (DESPOTIC), Lya (NCR default),
  HICollIon, HISmith21
- H+ cooling (3): HRecombination, FreeFreeH, GrainRecombination
- H2 cooling (3): H2Gong17, H2Moseley21 (NCR default), H2CollDiss
- Metal FS (4): CIIFineStructure, OIFineStructure, CIFineStructure,
  OIIFineStructure
- Other cooling (2): DustGasCoupling, NebularMetalLine (Phase 7
  proxy)
- Heating (5): PhotoelectricWD01, CosmicRay, H2Formation,
  H2Dissociation, H2Pump

Out of scope (Phase 4c):

- Analytic d(Lambda)/d(T/mu) for the simple channels.
- Tabulated derivatives for the level-population channels.
- Driver rebind: replace CoolingStub with CoolingChannels
  composing the 21 channels.
- coolCO (commented-out in get_cooling.py; revisit when CO
  abundance tracking lands).

Validation: 516 passed, 4 skipped (up from 512 / 4; +4 net new test
entries covering 30+ internal cases across the 4 channels, no
regressions).

## 2026-06-14: Phase 4c -- driver rebind to CoolingChannels aggregator

Wires the channel-policy pattern into the driver. The Phase 3 solver
no longer needs to wait on the Phase 4 cooling work: the
`make_ncr_default_cooling(species)` factory composes the 17
production-default channels (12 cooling + 5 heating) into a
`CoolingChannels` aggregator that the driver consumes through the
existing `cooling` policy slot, replacing `CoolingStub`.

Plumbing changes:

- `CoolingChannel.SCRATCH_NAMES` ClassVar: each concrete channel
  lists every state-scratch slot it reads inside `evaluate(...)`.
  The base `allocate_scratch(state)` method walks `SCRATCH_NAMES`
  and registers each as a `(ncell,)` float64 slot. 20 of the 21
  channels gained this declaration via a one-shot regex injection;
  the only channel with no scratch (`H2DissociationHeating`)
  keeps the empty default. `HeatingChannel` mirrors the contract.
- `CoolingChannels.allocate_scratch(state)` now also calls each
  composed channel's `allocate_scratch(state)` so the aggregator
  setup chain is one-shot.
- `ChemistryDriver.setup(state)` walks the cooling / opacity /
  radiation policy slots and calls their `allocate_scratch(state)`
  hooks when present. This means a driver wired with
  `make_ncr_default_cooling` has every channel-internal scratch
  registered after one `setup` call, and `cooling.update(state)`
  during the substep loop runs allocation-free.

Factory module:

- `pyathena.chemistry.cooling.factories.make_ncr_default_cooling(
  species, *, xi_CR, xi_diss_H2, kgr_H2, chi_band)` returns a
  `CoolingChannels(channels, heating)` ready to drop into
  `ChemistryDriver(..., cooling=...)`. The 17 NCR-default channels
  follow the wiring of `pyathena.microphysics.get_cooling.py` and
  the C++ `PhotochemistryNCR::HeatingH2` / `CoolingOther` paths.

Tests (`tests/chemistry/parity/test_cooling_channels_aggregator.py`,
3 tests):

- Aggregator allocate_scratch registers every per-channel Lambda /
  dLambda / Gamma / dGamma slot AND every channel-internal scratch
  name from each `SCRATCH_NAMES`.
- `cooling.update(state)` writes `solver:net_cool` equal to
  `sum_c Lambda_c - sum_h Gamma_h` within rtol = 1e-14 of the
  channels' independent evaluations, and writes a zero
  `solver:d_net_cool_d_temp_mu` (the contract Phase 4d will
  replace).
- `ChemistryDriver.setup(state)` chains the cooling allocation, and
  a one-shot `driver.step(dt, state)` runs without KeyError on a
  missing slot.

The aggregator's net_cool is now a real cooling rate; the solver's
substep estimator and stiffness machinery start seeing meaningful
inputs in real runs. The strict allocation-free hot-path guarantee
(`assert_no_alloc(allow=0)` in Phase 3 tests) is unaffected --
every channel's `evaluate` body still uses `out=` + named scratch
exclusively.

Out of scope (Phase 4d):

- Analytic / tabulated derivatives in each channel (`d_out` is still
  written zero).
- Cooling-table caching for the recombination cooling channel
  (`RecRate.get_rec_rate_H_caseB` allocates inside the substep loop
  per call; pre-tabulating onto a LogLog grid is the natural fix).

Validation: 519 passed, 4 skipped (up from 516 / 4; +3 net new tests
covering aggregator scratch wiring, sum-matches-channels, and the
driver setup chain).

## 2026-06-14: Phase 4 prereq -- build_cie_high_pool.py + pool tables

Adds the Phase 6 CIE-lumped-pool table builder and the four
production tables it writes. The builder is a Phase 4 prerequisite
for the Phase 6 `NCRNetwork3PlusIons16` ghost-row layout: the
CIE-lumped-pool ghosts (`x_high_C`, `x_high_N`, `x_high_O`,
`x_high_S`) need precomputed `(x_high_frac(T), q_high_mean(T),
Lambda_high(T))` columns to be useful at runtime.

Builder:

- `pyathena.chemistry.tables.chianti_v11.build_cie_high_pool` reads
  the existing CHIANTI v11 per-element tables under
  `data/microphysics/chianti_v11/`:
  - `ioneq_<element>.txt` -- CIE ionisation fractions x_q(T), shape
    `(Z+1, NT)`, from `build_ioneq`.
  - `cool_<element>.txt`  -- per-ion radiative-loss coefficient
    L_q(T) [erg cm^3 / s], shape `(Z+1, NT)`, from `build_cool`.
  For each element / q_max_tracked pair the builder sums:
  - `x_high_frac(T)  = sum_{q >= q_max_tracked} ioneq_q(T)`
  - `q_high_mean(T)  = sum_{q >= q_max_tracked} q * ioneq_q
                       / x_high_frac` (= 0 where the pool fraction
                        is < 1e-30, avoiding 0/0 in the cold tail)
  - `Lambda_high(T) = sum_{q >= q_max_tracked} ioneq_q * L_q`
  and writes the result to `data/chemistry/cie_high_pool_<element>.txt`.

- `ELEMENT_GROUPS` mirrors the
  `NCRNetwork3PlusIons16.element_groups` cutoffs from
  `chemistry-rewrite-plan.md` §4a.2: `('C', 2)` pools q = 2..6,
  `('N', 2)` pools q = 2..7, `('O', 3)` pools q = 3..8,
  `('S', 3)` pools q = 3..16.

- Helper functions: `compute_pool(ioneq, L_q, q_max_tracked)` does
  the core sum and is unit-testable in isolation; `write_ascii` and
  `read_cie_high_pool` are the I/O round-trip.

Tables (committed under `data/chemistry/`):

- `cie_high_pool_C.txt` (Z = 6, pool q = 2..6)
- `cie_high_pool_N.txt` (Z = 7, pool q = 2..7)
- `cie_high_pool_O.txt` (Z = 8, pool q = 3..8)
- `cie_high_pool_S.txt` (Z = 16, pool q = 3..16)

Each on the same 121-point log-T grid as the upstream CHIANTI
tables (10^3 K -> 10^9 K, 0.05 dex).

Tests (`tests/chemistry/test_cie_high_pool_builder.py`):

- `test_compute_pool_sums_correctly` hand-checks the three sums on
  a synthetic `(Z+1, NT)` ioneq + L_q input (rtol = 1e-14).
- `test_compute_pool_handles_empty_pool` exercises the 0/0 guard.
- `test_pre_built_O_table_is_well_formed` reads the on-disk O table
  and asserts the format invariants (monotone log_T, x_high in
  [0,1], q_high in [q_max, Z], Lambda_high >= 0, hot end fully
  ionised). Skipped when the table is missing on disk.

Phase 6 consumers: `NCRNetwork3PlusIons16.fill_ghosts` will read
these tables via a `CIEHighPool` helper that exposes
`x_high_frac(T)`, `q_high_mean(T)`, `Lambda_high(T)` as interpolated
arrays on the strip log-T grid. The plan §4a.3 closure formulation
uses `x_high_frac` to renormalise the evolved low-charge rows so
element conservation holds; `q_high_mean` enters the electron-
fraction ghost (`x_high * q_high_mean` contributes to x_e); and
`Lambda_high` plugs directly into the cooling channel for the pool
contribution.

Validation: 522 passed, 4 skipped (up from 519 / 4; +3 net new
tests covering the pool sum + format invariants).

## 2026-06-14: Phase 4d-a -- first analytic d_out (5 channels)

Phase 4d starts filling in the per-channel analytic
`d(Lambda) / d(T/mu)` (cooling) / `d(Gamma) / d(T/mu)` (heating)
buffers. Until now every channel wrote `d_out[:] = 0.0`, which makes
the substep loop's semi-implicit T/mu kernel degrade to forward
Euler -- correct but slow in stiff regimes (HII region Lya, cold
neutral CII). User experience confirms the derivative term is
required in practice; analytic form is cheap because it reuses the
same scratch values the Lambda computation already produced.

Convention:

- Channels write `d(out) / d(T/mu)` directly. Operator splitting
  freezes mu during the cooling sub-step, so analytic
  implementations compute `d(out) / dT` from the scratch
  intermediates left over from the Lambda calculation and multiply
  by `state.get_scratch('solver:mu_at_entry')` at the end to
  convert to the T/mu derivative.
- The base ABC docstring (`pyathena.chemistry.cooling.base.CoolingChannel.evaluate`)
  now explains the `d_out` contract in full, including the
  "skipped when None / zero accepted as the default" fallback for
  channels that have not been ported yet.

Channels with analytic d_out filled in:

- `heating.cosmic_ray.CosmicRayHeating` -- analytic = 0. CR heating
  depends on `xi_CR, x_HI, x_H2, x_e, n_H`; none have any T
  dependence under operator splitting.
- `heating.h2_photodissociation.H2DissociationHeating` -- analytic
  = 0. `Gamma = xi_diss * x_H2 * 0.4 eV` is purely scalar.
- `cooling.dust.DustGasCoupling` -- closed-form. The signed
  `Lambda = Z_d * alpha_gd * n_H * sqrt(T) * (T - T_dust)` has
  `dLambda/dT = K * (3T - T_dust) / (2 sqrt(T))` with K = Z_d *
  alpha_gd * n_H. New scratch slot `cooling:dust:tmp_b`.
- `cooling.free_free.FreeFreeHCooling` -- closed-form. The Gaunt
  factor `g_ff(T) = 1 + 0.44 / (1 + 0.058 L^2)` with L = ln(T /
  T_gff) gives `d(ln g_ff)/dT = -0.05104 * L / (T * denom^2 *
  g_ff)`; with the `sqrt(T/1e4)` factor folded in, `dLambda/dT =
  Lambda * (d(ln g_ff)/dT + 1 / (2 T))`. Two new scratch slots
  `cooling:free_free:L` and `cooling:free_free:denom` so the
  derivative reuses the values computed for Lambda.
- `cooling.lya.LyaCooling` -- closed-form. Through the collision-
  strength factor `fac(T)`, the Boltzmann factor `exp(-11.84/T4)`,
  and the level-1 fraction `f_1 = q_01 / (q_01 + q_10 + A_10)`,
  the derivative chains as
  `d(ln fac)/dT = (1/T) * (0.64897/v - 0.5)` with v the WD01
  denominator, then `dq_01/dT = q_01 * (d(ln fac)/dT + 11.84 /
  (T * T_4))`, `dq_10/dT = q_10 * d(ln fac)/dT`, and `df_1/dT =
  (dq_01/dT - f_1 * dD/dT) / D` with D = q_01 + q_10 + A_10. Two
  new scratch slots `cooling:lya:v` and `cooling:lya:d_ln_fac`.

Tests (`tests/chemistry/test_phase4d_analytic_derivatives.py`):

- One test function per channel; analytic `d_out` compared against
  a central FD `(Lambda(T+dT) - Lambda(T-dT)) / (2 dT) * mu` at
  `dT_rel = 1e-3`. Tolerance `rtol = 1e-4`.
- T grid is `np.unique(concatenate([logspace(2, 6, 24),
  logspace(3.5, 5, 30)]))`: broad coverage 100 K - 1e6 K with a
  denser sub-grid in the Lya / H ionisation knee at ~3e3 K - 1e5 K.
  Cross with `n_H` in [0.01, 1e4] cm^-3 (14 points), and 2 - 6
  species / state cases per channel -- 1k to 4k FD comparisons per
  channel.
- Deep-cold-tail mask: where `|d_out| < 1e-25`, FD picks up
  catastrophic cancellation from huge Boltzmann exponentials and
  the comparison becomes a precision-noise check. Cells below the
  mask are excluded; they contribute negligibly to physical
  net_cool.

Compatibility:

- Aggregator parity test updated: the "d_net_cool == 0" assertion
  is replaced with "d_net_cool = sum_c dLambda_c - sum_h dGamma_h"
  to reflect the new non-zero entries. Hand-built scratch lists in
  the legacy phase 4b parity tests are also extended with the new
  slot names (`cooling:dust:tmp_b`, `cooling:free_free:L`,
  `cooling:free_free:denom`, `cooling:lya:v`,
  `cooling:lya:d_ln_fac`).
- Lambda numerics unchanged byte-for-byte: the Phase 4b parity
  tests against `pyathena.microphysics.cool` still pass at
  rtol = 1e-12. The derivative paths are pure additions, not a
  Lambda restructuring.

Out of scope (Phase 4d-b...):

- HICollIon, HRecomb, PE, GrainRec (H ionisation family).
- Nebular, Smith21, H2Form, H2Pump, H2CollDiss.
- H2Moseley21, H2Gong17.
- CII / OI / CI / OII (frozen-q_ij analytic; level-pop).

Validation: 527 passed, 4 skipped (up from 522 / 4; +5 net new
analytic-derivative parity tests, no regressions).
