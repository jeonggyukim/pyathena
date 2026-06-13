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

This file is a Phase 0 skeleton. Methods that need policy-specific
information (`from_grid`, `from_meshblock`, `from_photchem`, `to_hdf5`)
are stubs that raise `NotImplementedError`; subsequent phases fill
them in.
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

    Phase 0 note: the schema is locked here, but the factory methods
    (`from_grid`, `from_meshblock`, `from_photchem`) and the
    `to_hdf5` / `from_hdf5` serialisers are stubs. Subsequent phases
    flesh them out as concrete networks land.
    """

    # ---- Frozen schema header ----
    # `species` is a SpeciesSet placeholder for now (Phase 2 introduces
    # the real SpeciesSet class). Using Any to avoid a forward import
    # cycle in the skeleton.
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

        No setter. Callers must mutate `x` to change `ne`.

        Phase 0 stub: needs `SpeciesSet` to know per-species charges.
        Returns zeros until Phase 2 wires up the charge vector.
        """
        return np.zeros_like(self.nH)

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

        Phase 0 implementation does basic shape consistency; finer
        checks (species charge sum, abundance closure) land alongside
        the SpeciesSet implementation in Phase 2.
        """
        ncell = self.ncell
        if self.T.shape != (ncell,):
            raise ValueError(
                f'T shape {self.T.shape} does not match ncell={ncell}')
        if self.T_dust.shape != (ncell,):
            raise ValueError(
                f'T_dust shape {self.T_dust.shape} != ncell={ncell}')
        if self.dt_remaining.shape != (ncell,):
            raise ValueError(
                f'dt_remaining shape {self.dt_remaining.shape} != ncell={ncell}')
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

    # ---- Factory stubs (Phase 0 skeleton) ----
    @classmethod
    def from_grid(cls, *args, **kwargs) -> 'ChemState':
        """Construct from a 1-D radial / planar grid + initial fields.
        Phase 2 fleshes this out alongside SpeciesSet.
        """
        raise NotImplementedError(
            'ChemState.from_grid arrives in Phase 2 with SpeciesSet')

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
