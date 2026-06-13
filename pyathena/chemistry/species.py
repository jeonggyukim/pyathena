"""Species inventory + ordering used across the chemistry stack.

The two primary objects are:

- `Ion`: frozen dataclass for a single ion (e.g., HII, OIII, electron).
- `SpeciesSet`: frozen, ordered collection of species names. Every
  policy (network, solver, cooling, ...) reads the species ordering
  from here; storage in `ChemState.x` follows `SpeciesSet.names`.

Design notes (see also the chemistry-rewrite-plan.md, Section 2):

- Hydrogen carries three species: HI, HII, H2. The H2 species is
  counted as 0.5 particle per H nucleus in the mu formula
  (`n_per_particle`), so that the H mass-conservation constraint
  `x_HI + x_HII + 2*x_H2 = 1` is expressed exactly in storage units.
- The free electron is a species in its own right (final row). Its
  density is derived by `ChemState.ne` from `sum_i q_i x_i n_H`;
  storing it lets parity tests inject a reference value directly.
- The mu formula consumed by ChemState is::

    mu = muH / sum_i n_per_particle[i] * x_i

  matching `pyathena.microphysics.get_cooling.mu = muH / (1.1 + xe - x_H2)`
  for the minimal HI / HII / H2 / e set with x_He = 0.1 carried as a
  scalar inside `n_per_particle` for HI (1.0 + 0.1 = 1.1 baseline) when
  helium is not tracked as a species (see factories below for the
  trick).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Tuple

import numpy as np

# muH per Asplund09 + xHe = 9.55e-2 (Draine 2011 Table 1.4).
# Single source of truth for the mean mass per H nucleus in amu;
# matches `pyathena/microphysics/get_cooling.py:36` and
# `AbundanceSolar.get_XYZ_muH_mu` for Zprime=1, xHe=9.55e-2.
MU_H_DEFAULT: float = 1.4271
# Helium abundance per H. Used in the n_per_particle baseline for
# the minimal HI/HII/H2/e set where He is not tracked as a species.
X_HE_DEFAULT: float = 0.1


@dataclass(frozen=True)
class Ion:
    """A single ion (or atom, or molecule, or electron).

    Parameters
    ----------
    element : str
        Element symbol (`'H'`, `'He'`, `'O'`, ...). For the electron
        and molecules use the convention `'e'` and `'H2'`.
    Z : int
        Atomic number (proton count). 0 for the electron.
    N : int
        Electron count. Equal to Z for a neutral atom; 1 for the
        electron; 2 for H2 (two bound electrons per molecule, used
        purely for bookkeeping).
    charge : int
        Net charge in units of |e|. Equals `Z - N` for ions; `-1` for
        the electron.
    name : str
        Display name (`'HI'`, `'HII'`, `'H2'`, `'electron'`, `'OIII'`).
        Must be unique inside any `SpeciesSet` it joins.
    """
    element: str
    Z: int
    N: int
    charge: int
    name: str


def _ion(name: str, element: str, Z: int, N: int, charge: int) -> Ion:
    """Internal helper to build an Ion with positional args reordered
    for readability inside factory bodies.
    """
    return Ion(element=element, Z=Z, N=N, charge=charge, name=name)


# Canonical Ion instances for the species this package currently knows
# about. New entries here force a SpeciesSet update + parity test.
HI       = _ion('HI',       'H',  Z=1, N=1, charge=0)
HII      = _ion('HII',      'H',  Z=1, N=0, charge=+1)
H2       = _ion('H2',       'H2', Z=2, N=2, charge=0)   # 2 H nuclei + 2 e
HeI      = _ion('HeI',      'He', Z=2, N=2, charge=0)
HeII     = _ion('HeII',     'He', Z=2, N=1, charge=+1)
HeIII    = _ion('HeIII',    'He', Z=2, N=0, charge=+2)
ELECTRON = _ion('electron', 'e',  Z=0, N=1, charge=-1)
CI       = _ion('CI',       'C',  Z=6, N=6, charge=0)
CII      = _ion('CII',      'C',  Z=6, N=5, charge=+1)
CO       = _ion('CO',       'CO', Z=14, N=14, charge=0)  # 6 + 8; net 14 e
OI       = _ion('OI',       'O',  Z=8, N=8, charge=0)
OII      = _ion('OII',      'O',  Z=8, N=7, charge=+1)


@dataclass(frozen=True)
class SpeciesSet:
    """Ordered, frozen collection of species.

    The `names` tuple is the canonical storage order: `ChemState.x` is
    `(nspec, ncell)` with row `i` storing the abundance of `names[i]`.

    Use the factory classmethods rather than constructing by hand:

    - `SpeciesSet.minimal_HI_HII_H2()` for the 4-species NCRNetwork3
      layout (HI, HII, H2, electron) with the electron as the only
      ghost species.
    - `SpeciesSet.ncr3_with_ghosts()` extends the minimal set with the
      five algebraic ghost rows (CI, CII, CO, OI, OII) consumed by the
      NCR cooling layer (`CoolingOther` in `ncr_rates.hpp`).
    - `SpeciesSet.ncr3_plus_helium()` adds the three helium ions
      (HeI, HeII, HeIII) ahead of the electron — placeholder for the
      Phase 6 helium-tracking network.

    The vector fields (`charges`, `n_per_particle`, `mass_per_particle`,
    `is_electron`) are populated in `__post_init__` from the `Ion`
    table so that callers can write vectorised expressions without a
    per-species Python loop.

    Two additional partitioning slots are filled by the constructor
    based on the optional `evolved_names_in` / `ghost_names_in` inputs:

    - `evolved_names` / `evolved_idx` / `n_evolved`: subset of `names`
      that a solver integrates as ODE state variables.
    - `ghost_names` / `ghost_idx` / `n_ghost`: subset of `names` whose
      values a network rebuilds algebraically each substep via
      `NetworkBase.fill_ghosts`.

    The two subsets must cover `names` exactly with no overlap; the
    constructor raises `ValueError` if they do not. When neither
    partitioning input is provided, the constructor defaults to
    `evolved = names` and `ghost = ()` (back-compat with legacy
    callers that did not know about the split).
    """

    ions: Tuple[Ion, ...]

    # Optional partitioning inputs. The defaults of `None` mean
    # "treat every species as evolved and none as ghost"; concrete
    # factories pass explicit tuples.
    evolved_names_in: Optional[Tuple[str, ...]] = None
    ghost_names_in:   Optional[Tuple[str, ...]] = None

    # Filled by __post_init__; declared with init=False so the
    # constructor signature stays `SpeciesSet(ions=...)`.
    names: Tuple[str, ...]               = field(init=False)
    elements: Tuple[str, ...]            = field(init=False)
    idx: Mapping[str, int]               = field(init=False)
    charges: np.ndarray                  = field(init=False)
    n_per_particle: np.ndarray           = field(init=False)
    mass_per_particle: np.ndarray        = field(init=False)
    is_electron: np.ndarray              = field(init=False)

    # Evolved / ghost partition. Solvers iterate over `evolved_idx`;
    # networks rebuild `ghost_idx` rows in `fill_ghosts`.
    evolved_names: Tuple[str, ...]       = field(init=False)
    ghost_names: Tuple[str, ...]         = field(init=False)
    evolved_idx: np.ndarray              = field(init=False)
    ghost_idx: np.ndarray                = field(init=False)
    n_evolved: int                       = field(init=False)
    n_ghost: int                         = field(init=False)

    # Mu baseline carried by the set. For sets that do not track He
    # explicitly we fold x_He into the HI / HII / H2 n_per_particle
    # baseline so the mu formula stays self-consistent.
    mu_H: float                          = field(init=False)
    x_He: float                          = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.ions, tuple):
            object.__setattr__(self, 'ions', tuple(self.ions))

        names = tuple(ion.name for ion in self.ions)
        if len(set(names)) != len(names):
            raise ValueError(
                f'SpeciesSet has duplicate names: {names}')
        object.__setattr__(self, 'names', names)

        # Preserve insertion order while keeping uniqueness.
        seen: list = []
        for ion in self.ions:
            if ion.element not in seen:
                seen.append(ion.element)
        object.__setattr__(self, 'elements', tuple(seen))

        object.__setattr__(self, 'idx',
                           {n: i for i, n in enumerate(names)})

        # Per-species vectors.
        charges = np.array([ion.charge for ion in self.ions],
                           dtype=np.int8)
        is_e = np.array([ion.name == 'electron' for ion in self.ions],
                        dtype=bool)
        object.__setattr__(self, 'charges', charges)
        object.__setattr__(self, 'is_electron', is_e)

        # Whether the set tracks helium explicitly. Drives the mu
        # baseline below.
        tracks_he = 'He' in self.elements
        object.__setattr__(self, 'x_He',
                           0.0 if tracks_he else X_HE_DEFAULT)
        object.__setattr__(self, 'mu_H', MU_H_DEFAULT)

        # Particles per H nucleus contributed by one unit of `x_i`.
        # The mu formula consumed by `ChemState` is
        #     mu = muH / sum_i n_per_particle[i] * x_i
        # plus an additive `x_He` baseline for sets that do not track
        # helium explicitly. For HI / HII / H2 / electron each species
        # contributes one particle per x_i (one H atom, one ion, one
        # molecule, one electron); the H mass-conservation closure
        # `x_HI + x_HII + 2 x_H2 = 1` is enforced separately by the
        # solver and is *not* baked into these coefficients.
        #
        # Cross-check with the canonical NCR mu in
        # `pyathena/microphysics/get_cooling.py:36`:
        #     mu = muH / (1.1 + xe - x_H2)
        # Using the closure x_HI + x_HII = 1 - 2 x_H2,
        #     1.1 + xe - x_H2 = (1 + x_He) + xe - x_H2
        #                     = (x_HI + x_HII + 2 x_H2) + x_He + xe - x_H2
        #                     = x_HI + x_HII + x_H2 + x_He + xe
        # which matches sum_i n_per_particle[i] * x_i with each
        # baseline coefficient = 1. The `x_He` baseline is added by
        # `ChemState.mu` via `SpeciesSet.x_He`, not by `n_per_particle`,
        # so the array is symmetric across species and easy to read.
        npp = np.zeros(len(self.ions), dtype=float)
        for i, ion in enumerate(self.ions):
            if ion.name in ('HI', 'HII', 'H2', 'electron'):
                npp[i] = 1.0
            elif ion.element == 'He':
                # 1 particle per He atom regardless of charge.
                npp[i] = 1.0
            else:
                # Generic metal: 1 particle per ion. Real abundances
                # are <= 1e-3 so this is a negligible correction;
                # carrying it keeps the formula consistent.
                npp[i] = 1.0
        object.__setattr__(self, 'n_per_particle', npp)

        # Mass per H nucleus in m_H units. Used by the rho - n_H
        # conversion in ChemState; HI/HII/H2 carry the full 1.0/1.0/2.0
        # per H, He carries 4.0 per He nucleus * x_He, etc. We tabulate
        # only the per-species coefficient; abundances live in the
        # ChemistryConfig.
        masses = np.zeros(len(self.ions), dtype=float)
        for i, ion in enumerate(self.ions):
            if ion.name in ('HI', 'HII'):
                masses[i] = 1.008
            elif ion.name == 'H2':
                masses[i] = 2.016
            elif ion.element == 'He':
                masses[i] = 4.0026
            elif ion.name == 'electron':
                masses[i] = 5.4858e-4
            else:
                # Placeholder; concrete metals fill in via per-element
                # tables (Asplund09 atomic weight).
                masses[i] = 0.0
        object.__setattr__(self, 'mass_per_particle', masses)

        # Evolved / ghost partition.
        # Defaults: every species is evolved when no partition is given.
        evolved_in = self.evolved_names_in
        ghost_in = self.ghost_names_in
        if evolved_in is None and ghost_in is None:
            evolved_in = names
            ghost_in = ()
        elif evolved_in is None:
            # ghost-only declared; evolved is the complement.
            ghost_set = set(ghost_in)
            evolved_in = tuple(n for n in names if n not in ghost_set)
        elif ghost_in is None:
            evolved_set = set(evolved_in)
            ghost_in = tuple(n for n in names if n not in evolved_set)

        evolved_in = tuple(evolved_in)
        ghost_in = tuple(ghost_in)

        # Validate the partition before computing index arrays so the
        # error message describes the user input rather than an index
        # KeyError further down.
        if set(evolved_in) & set(ghost_in):
            overlap = sorted(set(evolved_in) & set(ghost_in))
            raise ValueError(
                'SpeciesSet evolved/ghost partition overlaps: '
                f'{overlap}')
        union = set(evolved_in) | set(ghost_in)
        if union != set(names):
            missing = sorted(set(names) - union)
            extra = sorted(union - set(names))
            raise ValueError(
                'SpeciesSet evolved/ghost partition does not cover '
                f'names exactly; missing={missing} extra={extra}')

        evolved_idx = np.array(
            [self.idx[n] for n in evolved_in], dtype=np.int64)
        ghost_idx = np.array(
            [self.idx[n] for n in ghost_in], dtype=np.int64)

        object.__setattr__(self, 'evolved_names', evolved_in)
        object.__setattr__(self, 'ghost_names', ghost_in)
        object.__setattr__(self, 'evolved_idx', evolved_idx)
        object.__setattr__(self, 'ghost_idx', ghost_idx)
        object.__setattr__(self, 'n_evolved', int(evolved_idx.size))
        object.__setattr__(self, 'n_ghost', int(ghost_idx.size))

        self.validate()

    # ---- Factory methods ----
    @classmethod
    def minimal_HI_HII_H2(cls) -> 'SpeciesSet':
        """Four-species NCRNetwork3 layout: HI, HII, H2, electron.

        Storage order matches `tigris-ncr` `NCRStrip` SoA — HI first,
        then HII, then H2, then the derived electron row. The H mass-
        conservation closure `x_HI + x_HII + 2 x_H2 = 1` is enforced
        downstream by the solver (RAMSES-style clipping); SpeciesSet
        carries the metadata only.

        Partition: evolved = (HI, HII, H2), ghost = (electron,). The
        electron row is rebuilt from the positive-ion charges via
        `NetworkBase.fill_ghosts`.
        """
        return cls(
            ions=(HI, HII, H2, ELECTRON),
            evolved_names_in=('HI', 'HII', 'H2'),
            ghost_names_in=('electron',),
        )

    @classmethod
    def ncr3_with_ghosts(cls) -> 'SpeciesSet':
        """Nine-species NCRNetwork3 layout with the full algebraic
        ghost set used by the NCR cooling layer.

        Order: HI, HII, H2 (evolved), then electron, CI, CII, CO, OI,
        OII (ghost). The ghost rows are populated each substep by
        `NCRNetwork3.fill_ghosts` from the evolved rows, T, and the
        radiation / metal-abundance state — they are not integrated
        as ODE variables.
        """
        return cls(
            ions=(HI, HII, H2, ELECTRON, CI, CII, CO, OI, OII),
            evolved_names_in=('HI', 'HII', 'H2'),
            ghost_names_in=('electron', 'CI', 'CII', 'CO', 'OI', 'OII'),
        )

    @classmethod
    def ncr3_plus_helium(cls) -> 'SpeciesSet':
        """Phase 6 prep: NCR3 + HeI/HeII/HeIII.

        Order: HI, HII, H2, HeI, HeII, HeIII, electron. The helium
        chemistry is not implemented yet (the network policy still
        treats x_He as a scalar); the set is provided so downstream
        code paths and parity infrastructure can be plumbed without
        another schema change later.

        Partition: evolved = (HI, HII, H2, HeI, HeII, HeIII), ghost =
        (electron,). All helium ions are integrated as ODE variables
        once Phase 6 lands; until then they are seeded by the test
        harness directly.
        """
        return cls(
            ions=(HI, HII, H2, HeI, HeII, HeIII, ELECTRON),
            evolved_names_in=('HI', 'HII', 'H2', 'HeI', 'HeII', 'HeIII'),
            ghost_names_in=('electron',),
        )

    # ---- Lookups / helpers ----
    def __len__(self) -> int:
        return len(self.ions)

    def __contains__(self, name: str) -> bool:
        return name in self.idx

    def index(self, name: str) -> int:
        """Row index for `name`. Raises KeyError on miss."""
        return self.idx[name]

    def idx_of(self, name: str) -> int:
        """Row index for `name`. Mirrors the `idx` dict lookup but as
        a method, so parametric code (`species.idx_of(target_name)`)
        does not need to introspect the dict directly.
        """
        return self.idx[name]

    @property
    def nspec(self) -> int:
        return len(self.ions)

    @property
    def electron_index(self) -> int:
        """Row index of the electron species."""
        where = np.where(self.is_electron)[0]
        if where.size != 1:
            raise ValueError(
                'SpeciesSet must contain exactly one electron species; '
                f'found {where.size}')
        return int(where[0])

    @property
    def h2_index(self):
        """Row index of H2 if the set tracks it, else `None`.

        Exposed for ThermoPolicy implementations (NCRThermo reads this
        to pick up x_H2 without re-scanning the species names).
        """
        return self.idx.get('H2', None)

    def validate(self) -> None:
        """Sanity checks. Cheap; called once in `__post_init__` and at
        parity-test boundaries.

        Checks:
        - at least one species declared,
        - exactly one electron species present,
        - `n_per_particle` strictly positive (every species contributes
          a positive particle count to the mu sum),
        - per-species mass non-negative.
        """
        if self.nspec == 0:
            raise ValueError('SpeciesSet is empty')
        if int(self.is_electron.sum()) != 1:
            raise ValueError(
                'SpeciesSet must contain exactly one electron species; '
                f'is_electron = {self.is_electron.tolist()}')
        if not np.all(np.isfinite(self.n_per_particle)):
            raise ValueError(
                f'n_per_particle has non-finite entries: '
                f'{self.n_per_particle}')
        if np.any(self.n_per_particle <= 0.0):
            raise ValueError(
                'n_per_particle must be strictly positive: '
                f'{self.n_per_particle.tolist()}')
        if np.any(self.mass_per_particle < 0.0):
            raise ValueError(
                'mass_per_particle has negative entries: '
                f'{self.mass_per_particle.tolist()}')
