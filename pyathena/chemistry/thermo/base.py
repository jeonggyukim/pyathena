"""ThermoPolicy: abstract base for the thermal-state conversions
the chemistry rewrite needs.

Every concrete policy answers four questions for a `ChemState`:

  - `mu(state, out)`   : mean molecular weight per particle, in amu.
  - `T_to_e(state, out)`: specific internal energy per unit mass, in
                          erg / g, given the current `state.T` and `x`.
  - `e_to_T(state, out)`: inverse of `T_to_e`. Writes the temperature
                          implied by a specific internal energy.
  - `pressure(state, out)`: thermal pressure n_total k_B T per cell.

All methods write into a caller-owned `out` buffer of shape
`(state.ncell,)`. None of them allocate on the hot path.

Design notes (see
`tigris-notes/docs-claude/pyathena/chemistry-rewrite-plan.md`, section 4):

- `gamma` is a class attribute, not part of the state. The adiabatic
  index is set by the network policy (5/3 for monatomic, 7/5 for the
  fully molecular limit). The Tigris hydro uses 5/3, so `NCRThermo`
  inherits that.
- `e_to_T` and `T_to_e` are mass-specific (`e` in erg/g, not erg/cm^3)
  to match the Athena++ hydro `cons[IEN] / rho` convention.
- Constants `k_B` and `m_H` are cached as module-level floats at
  import time. The hot path never reaches into `astropy.constants`.
"""
from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import numpy as np
from astropy import constants as _const

if TYPE_CHECKING:  # pragma: no cover - import-time only
    from ..state import ChemState


# Constants cached at module-import time. CGS, plain floats.
# These are the same values `pyathena.microphysics.get_cooling` uses
# (k_B in erg/K, m_H in g) and what the tigris-ncr C++ side calls
# `Constants::boltzmann_cgs` and `Constants::hydrogen_mass_cgs`.
K_B_CGS: float = float(_const.k_B.cgs.value)
M_H_CGS: float = float(_const.m_p.cgs.value)


class ThermoPolicy(abc.ABC):
    """Abstract base for thermal-state conversion policies.

    Subclasses implement `mu`, `T_to_e`, `e_to_T`, and `pressure`.
    `gamma` is overridable as a class attribute.
    """

    # Adiabatic index. 5/3 for an atomic / monatomic plasma; 7/5 for the
    # fully molecular limit. Set on the subclass.
    gamma: float = 5.0 / 3.0

    @abc.abstractmethod
    def mu(self, state: "ChemState", out: np.ndarray) -> np.ndarray:
        """Write mean molecular weight per particle (amu) into `out`.

        `out` must have shape `(state.ncell,)`. Returns `out` for
        chaining; never allocates.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def T_to_e(self, state: "ChemState", out: np.ndarray) -> np.ndarray:
        """Write specific internal energy (erg / g) into `out`.

        Uses `state.T` and the current abundances. `out` must have
        shape `(state.ncell,)`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def e_to_T(self, state: "ChemState", e: np.ndarray,
               out: np.ndarray) -> np.ndarray:
        """Write the temperature (K) implied by specific internal
        energy `e` (erg / g) into `out`.

        `e` and `out` must have shape `(state.ncell,)`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def pressure(self, state: "ChemState", out: np.ndarray) -> np.ndarray:
        """Write thermal pressure (erg / cm^3) into `out`.

        Equals `n_total * k_B * T` evaluated per cell.
        """
        raise NotImplementedError
