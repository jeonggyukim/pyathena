"""NCRThermo: the tigris-NCR mu formula and its T <-> e_internal,
P conversions.

The mu expression matches PhotochemistryNCR::GetTemperature in
`tigris-ncr/src/photchem/photchem.hpp:295-298` exactly:

    inline Real PhotochemistryNCR::GetTemperature(
        const Real rho, const Real press,
        const Real x_h2, const Real x_e) {
      return press/rho * punit->temperature_mu_cgs * mu_hyd_
             / (1.0 + x_he_ - x_h2 + x_e);
    }

Reading off the denominator: per H nucleus, the total particle count
is `1 + A_He - x_H2 + x_e`. The `1 - x_H2` term combines H atoms,
H+, and H2 molecules using H mass conservation
(x_HI + x_HII + 2 x_H2 = 1, total H-particle count
= x_HI + x_HII + x_H2 = 1 - x_H2). `A_He` is the He / H abundance
ratio; the C++ source uses `x_he_ = 0.1`; this module defaults to
`A_He = 0.0955` (Draine 2011 Table 1.4 / Asplund 2009), which is also
the value used by `pyathena.microphysics.abundance_solar` and
`pyathena.microphysics.get_cooling`.

`mu_hyd = 1 + 4 A_He` is the mean mass per H nucleus in atomic mass
units. Multiplying `mu_hyd / m_H_total_particles_per_H` gives mu in amu.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from .base import K_B_CGS, M_H_CGS, ThermoPolicy

if TYPE_CHECKING:  # pragma: no cover - import-time only
    from ..state import ChemState


class NCRThermo(ThermoPolicy):
    """Concrete ThermoPolicy for the tigris-NCR network.

    Parameters
    ----------
    A_He : float, optional
        Helium-to-hydrogen abundance ratio n_He / n_H. Defaults to
        0.0955 (Draine 2011 Table 1.4 / Asplund 2009). Kept as a
        constructor argument so a future cosmological run can sweep
        it from a non-solar composition.

    Notes
    -----
    The hot path reads `x_H2` and `x_e` from the state. Two routes
    are supported:

    - If `state.species` has an `h2_index` attribute (int or None),
      `x_H2 = state.x[h2_index]` for non-None, else 0.
    - If `state.species` has a `charges` attribute (1-D array of
      length nspec), `x_e = charges @ state.x`. If absent, `x_e = 0`.

    Both lookups happen once per call; the rest is pure NumPy
    arithmetic in caller-owned buffers, so there is no allocation in
    the hot loop.
    """

    # The Tigris hydro runs with gamma = 5/3 (monatomic). H2 is tracked
    # by the chemistry but does not enter the EOS; the energy partition
    # follows from the C++ side using gamma = 5/3 across the whole
    # plasma.
    gamma: float = 5.0 / 3.0

    def __init__(self, A_He: float = 0.0955) -> None:
        self.A_He: float = float(A_He)
        # mu_hyd = mean mass per H nucleus in amu. With only H + He
        # contributing, this is (m_H + A_He * m_He) / m_H = 1 + 4 A_He.
        self.mu_hyd: float = 1.0 + 4.0 * self.A_He

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _x_h2(state: "ChemState") -> np.ndarray:
        """Return x_H2 per cell, or a scalar 0.0 if the state's species
        set does not track H2.
        """
        species = state.species
        h2_index: Optional[int] = getattr(species, 'h2_index', None)
        if h2_index is None:
            return np.zeros(state.ncell)
        return state.x[h2_index, :]

    @staticmethod
    def _x_e(state: "ChemState") -> np.ndarray:
        """Return x_e per cell from `species.charges @ state.x`, or
        zeros if the species set does not expose charges.
        """
        species = state.species
        charges = getattr(species, 'charges', None)
        if charges is None:
            return np.zeros(state.ncell)
        return np.asarray(charges) @ state.x

    # ------------------------------------------------------------------
    # ThermoPolicy interface
    # ------------------------------------------------------------------
    def mu(self, state: "ChemState", out: np.ndarray) -> np.ndarray:
        """Write mean molecular weight per particle (amu).

        Formula (C++ port; photchem.hpp:295-298):

            mu = mu_hyd / (1 + A_He - x_H2 + x_e)
        """
        x_h2 = self._x_h2(state)
        x_e = self._x_e(state)
        # In-place: out = 1 + A_He - x_H2 + x_e, then out = mu_hyd / out.
        np.add(1.0 + self.A_He, x_e, out=out)
        np.subtract(out, x_h2, out=out)
        np.divide(self.mu_hyd, out, out=out)
        return out

    def T_to_e(self, state: "ChemState", out: np.ndarray) -> np.ndarray:
        """Write specific internal energy (erg / g) given state.T.

        Formula:

            e = (1 / (gamma - 1)) * k_B * T / (mu * m_H)

        with mu evaluated at the current `x`. Uses `out` as scratch
        for mu first, then overwrites with `e`.
        """
        # out = mu, then transform in place to e.
        self.mu(state, out)
        # Multiplicative scale on T / mu: k_B / ((gamma - 1) * m_H).
        scale = K_B_CGS / ((self.gamma - 1.0) * M_H_CGS)
        np.divide(state.T, out, out=out)   # out = T / mu
        np.multiply(out, scale, out=out)   # out = scale * T / mu
        return out

    def e_to_T(self, state: "ChemState", e: np.ndarray,
               out: np.ndarray) -> np.ndarray:
        """Write temperature (K) implied by specific internal energy
        `e` (erg / g).

        Inverse of `T_to_e`:

            T = (gamma - 1) * e * mu * m_H / k_B
        """
        self.mu(state, out)  # out = mu
        scale = (self.gamma - 1.0) * M_H_CGS / K_B_CGS
        np.multiply(out, e, out=out)
        np.multiply(out, scale, out=out)
        return out

    def pressure(self, state: "ChemState", out: np.ndarray) -> np.ndarray:
        """Write thermal pressure (erg / cm^3).

        Formula:

            P = n_total * k_B * T
              = n_H * (1 + A_He - x_H2 + x_e) * k_B * T
        """
        x_h2 = self._x_h2(state)
        x_e = self._x_e(state)
        # out = n_total / n_H = 1 + A_He - x_H2 + x_e
        np.add(1.0 + self.A_He, x_e, out=out)
        np.subtract(out, x_h2, out=out)
        # out *= nH * k_B * T
        np.multiply(out, state.nH, out=out)
        np.multiply(out, state.T, out=out)
        np.multiply(out, K_B_CGS, out=out)
        return out
