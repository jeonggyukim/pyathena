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

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple

import numpy as np

from .diagnostics import SolverDiag


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

    # ---- Optional scratch (allocated lazily when a policy requests it) ----
    C:          Optional[np.ndarray] = None    # (nspec, ncell)
    D:          Optional[np.ndarray] = None    # (nspec, ncell)
    Lambda:     Optional[np.ndarray] = None    # (nchan_cool, ncell)
    Gamma:      Optional[np.ndarray] = None    # (nchan_heat, ncell)
    metal_CT:   Optional[np.ndarray] = None    # (4, ncell)
    regime_tag: Optional[np.ndarray] = None    # (ncell,) int8 — matches enums.Regime
    nsub_est:   Optional[np.ndarray] = None    # (ncell,) int32

    # ---- Diagnostics (fixed-field POD; no free-form dict on hot path) ----
    diag: SolverDiag = field(default_factory=SolverDiag)

    # ---- Derived / read-only properties ----
    @property
    def ne(self) -> np.ndarray:
        """Electron density n_e = n_H * sum_i q_i x_i, derived from `x`.

        No setter. Callers must mutate `x` to change `ne`. Charge
        neutrality is read off `SpeciesSet.charges`; the electron row
        carries charge -1 so it cancels HII in a balanced state.

        For sets that include the electron row, this is equivalent to
        `n_H * x_e`. For sets that derive the electron implicitly (no
        electron row), the sum is over the positive-ion rows only.
        """
        species = getattr(self, 'species', None)
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
            C=None if self.C is None else self.C.copy(),
            D=None if self.D is None else self.D.copy(),
            Lambda=None if self.Lambda is None else self.Lambda.copy(),
            Gamma=None if self.Gamma is None else self.Gamma.copy(),
            metal_CT=None if self.metal_CT is None else self.metal_CT.copy(),
            regime_tag=None if self.regime_tag is None else self.regime_tag.copy(),
            nsub_est=None if self.nsub_est is None else self.nsub_est.copy(),
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

        chi = np.zeros((max(int(nfreq), 0), ncell), dtype=np.float64)
        N_col = np.zeros((max(int(ncol), 0), ncell), dtype=np.float64)
        xi_CR_arr = np.full(ncell, float(xi_CR), dtype=np.float64)
        dvdr_arr = np.full(ncell, float(dvdr), dtype=np.float64)
        dt_rem = np.zeros(ncell, dtype=np.float64)

        state = cls(
            species=species,
            policy_versions={'network': '__none__', 'thermo': '__none__'},
            walk_order=(),
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
        )
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
