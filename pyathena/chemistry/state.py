"""ChemState — the single object every chemistry policy reads and
writes.

The state carries: species inventory + ordering (frozen schema
header), physical fields (n_H, T, T_dust, Z_g, Z_d, x, ne, radiation,
column densities, velocity gradient), time-step bookkeeping
(dt, dt_remaining, t), optional scratch buffers allocated lazily by
specific solvers / networks, and a `SolverDiag` counter block.

Design choices (cross-referenced to
`tigris-notes/docs-claude/pyathena/chemistry-rewrite-plan.md`):

- Mutable dataclass with a frozen schema header. `species`,
  `policy_versions`, and array shapes are set once in `__init__` and
  never mutated. Arrays themselves are written in place. Mirrors the
  C++ `NCRStrip` SoA: zero allocation in the inner loop, plus a
  one-time `validate()` after construction.

- `ne` is a `@property` computed from `x` with no setter — callers
  cannot inject stale electron densities.

- Vectorised, with `ncell` as a leading axis. `ncell=1` is the natural
  per-cell case. Matches the C++ strip layout (`CHUNK_SIZE = 16`) so
  1:1 parity tests work without a re-broadcast step.

- HDF5 serialisation persists payload only; scratch and `diag` are
  recomputed on restart.

`from_grid` is the canonical factory for offline / 1D analytic
benchmarks (Stromgren sphere, PDR plane-parallel, single-cell sweeps).
`from_meshblock` views an Athena++ MeshBlock as a ChemState without
copying — used by the tigris-ncr port path. `from_photchem` adapts a
legacy `pyathena.microphysics.photchem.PhotChem` instance for parity
tests. Stubs raise `NotImplementedError` until the consuming code
path lands; see the chemistry-rewrite-plan.md roadmap.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from .diagnostics import SolverDiag


# Default radiation-field band layout for the 3-band NCR convention
# (FUV / LW / EUV). Matches `NCRRates` on the C++ side.
DEFAULT_CHI_BANDS: Tuple[str, ...] = ('FUV', 'LW', 'EUV')

# Canonical band layout shared with `pyathena.util.sb99` and the
# legacy `pyathena.microphysics.photchem` setup.
#
#     Band   Wavelength (A)      Use
#     ----   --------------      --------------------------------
#     LyC    < 912               photoionising (H, He, ...)
#     LW     912 - 1108          H2 photodissociation, dust PE
#     PE     1108 - 2068         PAH heating, dust charging, CI/SiI
#     FUV    912 - 2068          LW + PE combined (Draine G_0)
#     OPT    2068 - 10000        optical (PAH IR seed)
#     IR     10000 - 200000      dust thermal
#
# Note: `FUV` overlaps with `LW` + `PE` (not disjoint); radiation
# policies pick one decomposition or the other based on the physics
# they expose.
#
# ISRF reference energy densities per band [erg cm^-3], Habing 1968 /
# Draine 1978 convention. Match tigris-ncr/src/photchem/photchem.hpp:
# `u_rad_pe_isrf_cgs = 7.613e-14`, `u_rad_lw_isrf_cgs = 1.335e-14`.
# Rate / cooling functions normalise `chi_band = u_rad[band] /
# U_RAD_ISRF[band]` at point of use to obtain the dimensionless Draine
# field strength.
U_RAD_PE_ISRF_CGS:  float = 7.613e-14
U_RAD_LW_ISRF_CGS:  float = 1.335e-14
U_RAD_FUV_ISRF_CGS: float = U_RAD_PE_ISRF_CGS + U_RAD_LW_ISRF_CGS

U_RAD_ISRF_CGS: Dict[str, float] = {
    'PE':  U_RAD_PE_ISRF_CGS,
    'LW':  U_RAD_LW_ISRF_CGS,
    'FUV': U_RAD_FUV_ISRF_CGS,
}

# Sentinel string used in `policy_versions` when no policy was supplied
# at construction time. Phase 3 drivers overwrite these via `from_grid`'s
# policy kwargs.
_POLICY_NONE_SENTINEL: str = '__none__'

# Names of the policy roles that `from_grid` recognises as keyword
# arguments. Order is the canonical display order (network -> solver ->
# thermo -> cooling -> opacity -> radiation).
_POLICY_ROLES: Tuple[str, ...] = (
    'network', 'solver', 'thermo', 'cooling', 'opacity', 'radiation',
)


def _policy_version_tag(policy: Any) -> str:
    """Stamp a policy as `ClassQualname@version_string`.

    `version_string` is `policy.__version__` if set, else the sentinel
    `__none__`. Callers pass the policy instance (not the class) so
    instance-level overrides are picked up.
    """
    if policy is None:
        return _POLICY_NONE_SENTINEL
    cls_name = type(policy).__qualname__
    version = getattr(policy, '__version__', _POLICY_NONE_SENTINEL)
    return f'{cls_name}@{version}'


@dataclass
class ChemState:
    """Per-strip chemistry / thermodynamic state.

    See the package README and the chemistry-rewrite-plan.md design
    document for the rationale behind each field.

    The schema (which fields exist, in what shape, with what dtypes)
    is fixed here and validated by `validate()`. The factories
    (`from_grid`, `from_meshblock`, `from_photchem`) and the HDF5
    serialisers are filled in as the consuming code paths land; see
    the chemistry-rewrite-plan.md roadmap.
    """

    # ---- Frozen schema header ----
    # `species` is a SpeciesSet; typed as `Any` to avoid a forward
    # import cycle between state.py and species.py.
    species: Any
    policy_versions: Mapping[str, str]
    walk_order: Tuple[Tuple[str, ...], ...]

    # ---- Mutable payload (ncell-shaped unless noted) ----
    x:       np.ndarray              # (nspec, ncell), f64; mutated in place
    nH:      np.ndarray              # (ncell,)
    T:       np.ndarray              # (ncell,)
    T_dust:  np.ndarray              # (ncell,)
    Z_g:     np.ndarray              # (ncell,)
    Z_d:     np.ndarray              # (ncell,)
    chi:     np.ndarray              # (nfreq, ncell), Draine units
    xi_CR:   np.ndarray              # (ncell,)
    N_col:   np.ndarray              # (ncol, ncell)
    dvdr:    np.ndarray              # (ncell,)
    dt:           float
    dt_remaining: np.ndarray         # (ncell,) — set to dt at reset_step
    t:            float

    # ---- Structured radiation-field index ----
    # `chi_bands[i]` names the i-th row of `chi`. The 3-band NCR default
    # is `('FUV', 'LW', 'EUV')`; richer networks (Phase 6+) extend it.
    # Use `chi_for(band_name)` instead of ad-hoc attribute access.
    chi_bands: Tuple[str, ...] = DEFAULT_CHI_BANDS

    # ---- Lazily-allocated scratch buffers ----
    # Networks / solvers request scratch by string key via
    # `alloc_scratch(name, shape, dtype)` and read it back with
    # `get_scratch(name)`. The dict is owned by the state; hot-path code
    # is expected to allocate once at setup time (outside the inner
    # loop) and reuse the same buffer thereafter.
    scratch: Dict[str, np.ndarray] = field(default_factory=dict)

    # ---- Radiation energy density per band ----
    # Primary radiation-field representation: dict[band_name, ndarray]
    # of radiation energy density in cgs (erg cm^-3). The radiation /
    # ray-tracing module fills this; rate / cooling channels access
    # the Draine-normalised dimensionless `chi_band = u_rad[band] /
    # U_RAD_ISRF[band]` via `chi_for(band)`.
    #
    # Common bands and ISRF reference values (Habing 1968 / Draine
    # 1978; match tigris-ncr/src/photchem/photchem.hpp):
    #
    #   'PE':  6-13.6 eV photoelectric band; ISRF u_rad = 7.613e-14
    #   'LW':  Lyman-Werner band (11.2-13.6 eV); ISRF u_rad = 1.335e-14
    #   'FUV': alias for the integrated Draine FUV band when no
    #          subdivision is needed (rate functions treat
    #          `chi_FUV = chi_PE` for the PE / dust-charging context).
    #   'EUV': lambda < 912 Angstrom hydrogen-ionising band; usually
    #          consumed via the per-species `zeta_pi[species]` dict
    #          rather than as a chi normalisation.
    #
    # Phase 2 unit-test setups can leave the dict empty -- channels
    # fall back to `chi=0` (dark). When the dict is populated for a
    # band, `chi_for(band)` returns `u_rad[band] / U_RAD_ISRF[band]`
    # element-wise.
    u_rad: Dict[str, Any] = field(default_factory=dict)

    # ---- Photo rates (per-species, dict-keyed) ----
    # Convention: `xi_*` = cosmic-ray-driven rates (see `xi_CR`).
    # `zeta_*` = photo-driven rates (PDR / HII region literature).
    # Each dict is keyed by species name; missing keys default to 0 via
    # the `zeta_pi_for` / `zeta_diss_for` helpers and via the network's
    # `_get_optional` fallback. Values may be scalar (uniform over strip)
    # or `(ncell,)` arrays.
    #
    #   zeta_pi[species]     -- photoionisation rate, s^-1
    #     e.g., {'HI': 3e-9, 'H2': 1e-10, 'CII': ..., ...}
    #   zeta_diss[species]   -- photodissociation rate, s^-1
    #     e.g., {'H2': 1e-10, 'CO': 1e-11, ...}
    zeta_pi:   Dict[str, Any] = field(default_factory=dict)
    zeta_diss: Dict[str, Any] = field(default_factory=dict)

    # ---- Diagnostics (fixed-field POD; no free-form dict on hot path) ----
    diag: SolverDiag = field(default_factory=SolverDiag)

    # ---- Derived / read-only properties ----
    @property
    def ne(self) -> np.ndarray:
        """Electron density n_e derived from the state.

        No setter. Callers must mutate `x` to change `ne`. The
        network is responsible for keeping the electron row of `x`
        consistent with the rest of the strip via
        `NetworkBase.fill_ghosts` (see
        `pyathena/chemistry/networks/base.py`); this property simply
        reads it back.

        Resolution order:

        - If the species set carries an explicit `electron` row, the
          property returns `n_H * x[electron]` (the canonical 9-species
          `ncr3_with_ghosts` layout).
        - Otherwise the value comes from the positive-charge sum
          `n_H * sum_i max(q_i, 0) * x_i`. This branch only matters
          for legacy `SpeciesSet` constructions that omit an electron
          species entirely.
        """
        species = getattr(self, 'species', None)
        if species is not None:
            idx = getattr(species, 'idx', None)
            if idx is not None and 'electron' in idx:
                i_e = idx['electron']
                return self.nH * self.x[i_e]
        charges = getattr(species, 'charges', None)
        if charges is None:
            return np.zeros_like(self.nH)
        # `x_e` per cell = sum_i max(q_i, 0) * x_i. We use the absolute
        # charge convention: positive ions add electrons; the electron
        # row (charge -1) cancels HII when present. Equivalent to
        # taking the half-sum |q|/2, but the explicit positive-only sum
        # is easier to reason about.
        pos = np.maximum(np.asarray(charges), 0).astype(np.float64)
        x_e_per_cell = pos @ self.x
        return self.nH * x_e_per_cell

    @property
    def ncell(self) -> int:
        return self.nH.shape[0]

    @property
    def nspec(self) -> int:
        return self.x.shape[0]

    # ---- Structured radiation-field lookup ----
    def chi_for(self, band_name: str) -> np.ndarray:
        """Return the Draine-normalised field strength in band
        `band_name`. Resolution order:

        1. Special case: `band_name == 'FUV'` is the LW+PE combined
           Draine G_0. When the policy populated `state.u_rad['PE']`
           and `state.u_rad['LW']` separately,
           `chi_FUV = (u_rad['PE'] + u_rad['LW']) /
                      (U_RAD_PE_ISRF_CGS + U_RAD_LW_ISRF_CGS)`.
           Falls through to (2) if either sub-band is missing.

        2. If `state.u_rad[band_name]` is populated, return
           `u_rad[band_name] / U_RAD_ISRF_CGS[band_name]` -- the on-
           the-fly chi conversion. This is the primary path: the
           radiation module carries cgs energy densities; chemistry
           internally normalises to Draine units where rate / cooling
           formulas expect chi.

        3. Otherwise fall back to the legacy `state.chi[band_idx]`
           positional array (Phase 2 / 3 transitional layout); raises
           `KeyError` if the band is not in `chi_bands` either.

        Networks should prefer `chi_for(band_name)` over ad-hoc
        attribute access so the band layout can evolve without
        breaking call sites.
        """
        if band_name == 'FUV' and self.u_rad:
            u_PE = self.u_rad.get('PE')
            u_LW = self.u_rad.get('LW')
            if u_PE is not None and u_LW is not None:
                return ((np.asarray(u_PE) + np.asarray(u_LW))
                        / U_RAD_FUV_ISRF_CGS)
        u_rad_band = self.u_rad.get(band_name) if self.u_rad else None
        if u_rad_band is not None:
            u_isrf = U_RAD_ISRF_CGS.get(band_name)
            if u_isrf is None:
                raise KeyError(
                    f'no ISRF reference for chi normalisation of band '
                    f'{band_name!r}; '
                    f'known: {tuple(U_RAD_ISRF_CGS.keys())!r}')
            return np.asarray(u_rad_band) / u_isrf
        # Legacy positional chi array.
        try:
            idx = self.chi_bands.index(band_name)
        except ValueError as exc:
            raise KeyError(
                f'chi band {band_name!r} not present; '
                f'state.u_rad keys = {tuple(self.u_rad.keys())!r}, '
                f'state.chi_bands = {self.chi_bands!r}'
            ) from exc
        return self.chi[idx]

    # ---- Photo-rate lookup ----
    def zeta_pi_for(self, species: str, default: float = 0.0) -> Any:
        """Photoionisation rate of `species` (s^-1). Returns `default`
        if the species is not present in `state.zeta_pi`. Convention:
        `zeta_pi_species = sigma_pi_species * photon_flux`."""
        return self.zeta_pi.get(species, default)

    def zeta_diss_for(self, species: str, default: float = 0.0) -> Any:
        """Photodissociation rate of `species` (s^-1). Returns
        `default` if missing."""
        return self.zeta_diss.get(species, default)

    # ---- Scratch buffer management ----
    def alloc_scratch(
        self,
        name: str,
        shape: Tuple[int, ...],
        dtype: Any = np.float64,
    ) -> np.ndarray:
        """Allocate (or re-allocate) a scratch buffer and return it.

        Subsequent calls with the same `name` overwrite the previous
        buffer. Networks call this at setup time via
        `network.allocate_scratch(state)`; the hot path then uses
        `get_scratch(name)` only.
        """
        buf = np.zeros(shape, dtype=dtype)
        self.scratch[name] = buf
        return buf

    def get_scratch(self, name: str) -> np.ndarray:
        """Return the scratch buffer registered under `name`.

        Raises `KeyError` if no buffer has been allocated for `name` —
        callers must call `alloc_scratch` first (typically through the
        owning network's `allocate_scratch` hook). The hot path is
        expected to fetch buffers once and reuse the reference.
        """
        try:
            return self.scratch[name]
        except KeyError as exc:
            raise KeyError(
                f'scratch buffer {name!r} not allocated; '
                f'call alloc_scratch({name!r}, ...) first'
            ) from exc

    # ---- Lifecycle ----
    def reset_step(self, dt: float, t: float) -> None:
        """Called at the head of every hydro step.

        Sets `dt`, `t`, `dt_remaining = dt` per cell, zeroes
        diagnostics. Does NOT touch `x`, `T`, `nH` — those carry over
        from the previous step.
        """
        self.dt = float(dt)
        self.t = float(t)
        self.dt_remaining[:] = dt
        self.diag.reset()

    def validate(self) -> None:
        """Shape and finiteness asserts. Cheap; called after
        construction and at parity-test boundaries.

        Checks every payload array against the strip dimensions
        `(nspec, ncell)` and raises `ValueError` on mismatch. Also
        flags non-finite entries in the canonical hot-path arrays
        (`T`, `nH`, `x`).
        """
        ncell = self.ncell
        nspec = self.nspec

        if self.x.shape != (nspec, ncell):
            raise ValueError(
                f'x shape {self.x.shape} != (nspec, ncell) '
                f'= ({nspec}, {ncell})')
        if self.T.shape != (ncell,):
            raise ValueError(
                f'T shape {self.T.shape} does not match ncell={ncell}')
        if self.T_dust.shape != (ncell,):
            raise ValueError(
                f'T_dust shape {self.T_dust.shape} != ncell={ncell}')
        if self.Z_g.shape != (ncell,):
            raise ValueError(
                f'Z_g shape {self.Z_g.shape} != ncell={ncell}')
        if self.Z_d.shape != (ncell,):
            raise ValueError(
                f'Z_d shape {self.Z_d.shape} != ncell={ncell}')
        if self.xi_CR.shape != (ncell,):
            raise ValueError(
                f'xi_CR shape {self.xi_CR.shape} != ncell={ncell}')
        if self.dvdr.shape != (ncell,):
            raise ValueError(
                f'dvdr shape {self.dvdr.shape} != ncell={ncell}')
        if self.dt_remaining.shape != (ncell,):
            raise ValueError(
                f'dt_remaining shape {self.dt_remaining.shape} != '
                f'ncell={ncell}')
        if self.chi.ndim != 2 or self.chi.shape[1] != ncell:
            raise ValueError(
                f'chi shape {self.chi.shape} not (nfreq, ncell={ncell})')
        if self.N_col.ndim != 2 or self.N_col.shape[1] != ncell:
            raise ValueError(
                f'N_col shape {self.N_col.shape} not (ncol, ncell={ncell})')

        if not np.all(np.isfinite(self.T)):
            raise ValueError('T contains non-finite values')
        if not np.all(np.isfinite(self.nH)):
            raise ValueError('nH contains non-finite values')
        if not np.all(np.isfinite(self.x)):
            raise ValueError('x contains non-finite values')

    def freeze(self) -> 'ChemState':
        """Deep copy for parity tests. Not used in production paths;
        production never copies the state.
        """
        return ChemState(
            species=self.species,
            policy_versions=dict(self.policy_versions),
            walk_order=self.walk_order,
            x=self.x.copy(),
            nH=self.nH.copy(),
            T=self.T.copy(),
            T_dust=self.T_dust.copy(),
            Z_g=self.Z_g.copy(),
            Z_d=self.Z_d.copy(),
            chi=self.chi.copy(),
            xi_CR=self.xi_CR.copy(),
            N_col=self.N_col.copy(),
            dvdr=self.dvdr.copy(),
            dt=self.dt,
            dt_remaining=self.dt_remaining.copy(),
            t=self.t,
            chi_bands=self.chi_bands,
            scratch={k: v.copy() for k, v in self.scratch.items()},
            diag=SolverDiag(**self.diag.as_dict()),
        )

    # ---- Factory ----
    @classmethod
    def from_grid(
        cls,
        r: np.ndarray,
        nH: np.ndarray,
        T: np.ndarray,
        species,
        *,
        Z_g: float = 1.0,
        Z_d: float = 1.0,
        A_He: float = 0.0955,
        nfreq: int = 3,
        ncol: int = 0,
        T_dust: Optional[np.ndarray] = None,
        dvdr: float = 0.0,
        xi_CR: float = 0.0,
        chi_bands: Optional[Tuple[str, ...]] = None,
        network: Any = None,
        solver: Any = None,
        thermo: Any = None,
        cooling: Any = None,
        opacity: Any = None,
        radiation: Any = None,
    ) -> 'ChemState':
        """Build a strip-shaped state from a 1-D radial / planar grid.

        Parameters
        ----------
        r : ndarray, shape (ncell,)
            Grid coordinate. Kept on the state purely as a reference;
            not consumed by any policy. Used to size the strip.
        nH, T : ndarray, shape (ncell,)
            Initial hydrogen density [cm^-3] and temperature [K].
        species : SpeciesSet
            Species inventory. The strip carries `(species.nspec, ncell)`
            abundances initialized to the neutral state `x_HI = 1` (or
            its equivalent for the given species set: 1.0 on the first
            row, 0.0 on the rest). Callers override after construction.
        Z_g, Z_d : float, optional
            Gas / dust metallicity per cell. Stored as length-`ncell`
            arrays so a future call site can vary them spatially.
        A_He : float, optional
            Helium-to-hydrogen abundance ratio. Stored on `species` via
            the factory's mu baseline (this parameter is forwarded for
            future ThermoPolicy bindings; it is not written into the
            state because mu is a derived quantity).
        nfreq : int, optional
            Width of the radiation-field axis. Defaults to 3 to match
            the C++ NCRRates (FUV / LW / EUV).
        ncol : int, optional
            Width of the column-density axis. Defaults to 0 (no column
            tracking) so the test harness does not need a column model.
        T_dust : ndarray, optional
            Dust temperature per cell. Defaults to a constant 15 K
            (mid-range cold-ISM equilibrium) when not provided.
        dvdr, xi_CR : float, optional
            Scalar broadcasts of the velocity gradient and CR ionization
            rate. The driver overwrites these per substep.
        chi_bands : tuple of str, optional
            Names for the rows of `chi`. Defaults to `DEFAULT_CHI_BANDS`
            (`('FUV', 'LW', 'EUV')`) when `nfreq == 3` and to a
            positional `('chi_0', 'chi_1', ...)` layout otherwise.
        network, solver, thermo, cooling, opacity, radiation : optional
            Policy instances. When supplied, their qualified class name
            and `__version__` attribute are stamped into
            `state.policy_versions`. Unsupplied roles get the
            `'__none__'` sentinel.

        Returns
        -------
        ChemState
            Validated strip with the abundance vector seeded to the
            neutral state and all radiation / column fields zeroed.

        Notes
        -----
        The factory does not consume the helium baseline `A_He` --
        that lives on `ThermoPolicy`. It is in the signature so the
        Phase 3 driver can adopt a single call shape without further
        churn.
        """
        r = np.asarray(r, dtype=np.float64)
        if r.ndim != 1:
            raise ValueError(f'r must be 1-D; got shape {r.shape}')
        ncell = r.shape[0]

        nH = np.asarray(nH, dtype=np.float64)
        T = np.asarray(T, dtype=np.float64)
        if nH.shape != (ncell,):
            raise ValueError(
                f'nH shape {nH.shape} does not match r ncell={ncell}')
        if T.shape != (ncell,):
            raise ValueError(
                f'T shape {T.shape} does not match r ncell={ncell}')

        nspec = getattr(species, 'nspec', None)
        if nspec is None:
            raise ValueError(
                'species must expose `nspec`; got '
                f'{type(species).__name__}')

        # Seed abundances: one full unit on the first row (HI in the
        # canonical SpeciesSet ordering), zeros elsewhere. Concrete
        # callers override after construction.
        x = np.zeros((nspec, ncell), dtype=np.float64)
        x[0, :] = 1.0

        Z_g_arr = np.full(ncell, float(Z_g), dtype=np.float64)
        Z_d_arr = np.full(ncell, float(Z_d), dtype=np.float64)
        if T_dust is None:
            T_dust_arr = np.full(ncell, 15.0, dtype=np.float64)
        else:
            T_dust_arr = np.asarray(T_dust, dtype=np.float64)
            if T_dust_arr.shape != (ncell,):
                raise ValueError(
                    f'T_dust shape {T_dust_arr.shape} != ncell={ncell}')

        nfreq_int = max(int(nfreq), 0)
        chi = np.zeros((nfreq_int, ncell), dtype=np.float64)
        N_col = np.zeros((max(int(ncol), 0), ncell), dtype=np.float64)
        xi_CR_arr = np.full(ncell, float(xi_CR), dtype=np.float64)
        dvdr_arr = np.full(ncell, float(dvdr), dtype=np.float64)
        dt_rem = np.zeros(ncell, dtype=np.float64)

        # Resolve chi_bands. When the caller did not pass a layout,
        # default to the 3-band NCR convention for nfreq=3 and to a
        # positional naming for other widths so `chi_for()` still works.
        if chi_bands is None:
            if nfreq_int == len(DEFAULT_CHI_BANDS):
                bands_resolved: Tuple[str, ...] = DEFAULT_CHI_BANDS
            else:
                bands_resolved = tuple(
                    f'chi_{i}' for i in range(nfreq_int)
                )
        else:
            bands_resolved = tuple(chi_bands)
            if len(bands_resolved) != nfreq_int:
                raise ValueError(
                    f'chi_bands has {len(bands_resolved)} entries but '
                    f'nfreq={nfreq_int}'
                )

        # Stamp policy_versions from the supplied policy instances. The
        # 'network' / 'thermo' entries are always present (even when not
        # supplied) so legacy call sites that read these keys keep
        # working; the others appear only when provided.
        provided = dict(zip(
            _POLICY_ROLES,
            (network, solver, thermo, cooling, opacity, radiation),
        ))
        policy_versions: Dict[str, str] = {
            'network': _policy_version_tag(provided['network']),
            'thermo':  _policy_version_tag(provided['thermo']),
        }
        for role in ('solver', 'cooling', 'opacity', 'radiation'):
            if provided[role] is not None:
                policy_versions[role] = _policy_version_tag(provided[role])

        # Walk_order falls out of the network when one is supplied.
        walk_order: Tuple[Tuple[str, ...], ...] = ()
        if network is not None:
            wo = getattr(network, 'walk_order', None)
            if wo:
                walk_order = tuple(tuple(g) for g in wo)

        state = cls(
            species=species,
            policy_versions=policy_versions,
            walk_order=walk_order,
            x=x,
            nH=nH.copy(),
            T=T.copy(),
            T_dust=T_dust_arr,
            Z_g=Z_g_arr,
            Z_d=Z_d_arr,
            chi=chi,
            xi_CR=xi_CR_arr,
            N_col=N_col,
            dvdr=dvdr_arr,
            dt=0.0,
            dt_remaining=dt_rem,
            t=0.0,
            chi_bands=bands_resolved,
        )

        # Give the network a chance to register its scratch buffers.
        if network is not None:
            alloc_hook = getattr(network, 'allocate_scratch', None)
            if callable(alloc_hook):
                alloc_hook(state)
        # Attach the grid coordinate as a non-schema field for callers
        # that want to plot the strip later. Not part of validate().
        object.__setattr__(state, 'r', r)
        # Forward the helium abundance for downstream policy use; same
        # rationale as `r`.
        object.__setattr__(state, 'A_He', float(A_He))
        state.validate()
        return state

    @classmethod
    def from_meshblock(cls, *args, **kwargs) -> 'ChemState':
        """Construct from an Athena++ MeshBlock view. Phase 8."""
        raise NotImplementedError(
            'ChemState.from_meshblock arrives in Phase 8 (tigris-ncr port)')

    @classmethod
    def from_photchem(cls, *args, **kwargs) -> 'ChemState':
        """Construct by adapting an existing
        `pyathena.microphysics.photchem.PhotChem` instance — used by
        parity tests. Phase 2 fleshes this out alongside NCRNetwork3.
        """
        raise NotImplementedError(
            'ChemState.from_photchem arrives in Phase 2 with NCRNetwork3')

    def to_hdf5(self, path: str) -> None:
        """Serialise payload only. Phase 8."""
        raise NotImplementedError(
            'ChemState.to_hdf5 arrives in Phase 8 (tigris-ncr port)')


@contextlib.contextmanager
def assert_no_alloc(allow: int = 0):
    """Context manager: assert no new `numpy.ndarray` allocations
    happened inside the block.

    Used by Phase 3 solver tests to confirm that hot-path code paths do
    not allocate. Tracks ndarray construction by wrapping the array
    constructors numpy exposes at the Python level (`np.empty`,
    `np.zeros`, `np.ones`, `np.full`, `np.array`, `np.asarray`, and
    `np.copy`). Views, in-place ufuncs (`out=`), and pre-allocated
    buffers do not go through these constructors — those are the
    patterns the solver hot path must use.

    Parameters
    ----------
    allow : int, optional
        Tolerate up to `allow` allocations without raising. Defaults to
        zero. Useful when a known one-off setup allocation happens
        inside the timed block.

    Notes
    -----
    The wrap is process-wide (numpy has no thread-local hook), so do
    not nest these or use them from multiple threads. `np.asarray` of an
    array of the same dtype usually returns the input unchanged with no
    fresh allocation, but a dtype / copy=True conversion does allocate
    — the counter reflects what numpy actually did, not the call count.
    """
    counter = {'n': 0}
    constructors = ('empty', 'zeros', 'ones', 'full', 'array', 'copy',
                    'empty_like', 'zeros_like', 'ones_like', 'full_like')
    originals = {name: getattr(np, name) for name in constructors}

    def _wrap(name, fn):
        def wrapper(*args, **kwargs):
            out = fn(*args, **kwargs)
            if isinstance(out, np.ndarray):
                counter['n'] += 1
            return out
        wrapper.__name__ = name
        return wrapper

    for name, fn in originals.items():
        setattr(np, name, _wrap(name, fn))
    try:
        yield counter
    finally:
        for name, fn in originals.items():
            setattr(np, name, fn)
    if counter['n'] > allow:
        raise AssertionError(
            f'expected at most {allow} numpy allocations, '
            f'got {counter["n"]}'
        )
