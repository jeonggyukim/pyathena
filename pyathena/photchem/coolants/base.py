"""Generic 5-level coolant module factory.

Each per-ion file under this package (e.g., `o_3.py`, `n_2.py`)
is a thin shim that:

    from .base import IonCoolant
    coolant = IonCoolant('o_3.txt', label='OIII')
    populations = coolant.populations
    cooling = coolant.cooling

Carries the per-ion atomic data load + the Upsilon interpolation
+ the 5-level steady-state solve. All ions share the same physics
path; only the data file differs.
"""

import os
import numpy as np

from ..n_level import (
    solve_5level_steady_state,
    cooling_from_populations,
    NLEV,
)
from ..data.build_chianti_tables import read_ascii


# Prefactor in Draine 2011 Eq. 17.10 / Osterbrock 2006 Eq. 3.20
# q_ji = beta * Upsilon_ji / (g_j * sqrt(T))
# beta = h^2 / [(2 pi m_e)^(3/2) * sqrt(k_B)] = 8.629e-8 (cgs)
_BETA = 8.629e-8


class IonCoolant:
    """Single-ion 5-level cooling function backed by an atomic-data
    text file in `pyathena/photchem/data/`.

    Parameters
    ----------
    filename : str
        Name of the .txt under `pyathena/photchem/data/`,
        e.g. 'o_3.txt'.
    label : str
        Pretty-print label for diagnostics (e.g., 'OIII').
    """

    def __init__(self, filename, label):
        self.label = label
        self.filename = filename
        self._table = None  # lazily loaded

    def _load(self):
        # Per-ion atomic-data files live at
        # data/microphysics/chianti_v11/ alongside the other CHIANTI
        # tables (ioneq + cool). This module is at
        # pyathena/photchem/coolants/base.py; go up three to the
        # repo root then into data/microphysics/chianti_v11/.
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', '..', '..',
            'data', 'microphysics', 'chianti_v11', self.filename,
        )
        d = read_ascii(path)
        nphys = int(d.get('nlev_phys', len(d['E_erg'])))
        # Pad to NLEV=5 with high-E dummies if the physical level
        # count is smaller (e.g., the 2-level CII case).
        if nphys < NLEV:
            E = np.full(NLEV, np.inf)
            E[:nphys] = d['E_erg']
            E[nphys:] = 1.0e7 * 1.986e-16          # ~ 1e7 cm^-1 in erg
            g = np.ones(NLEV)
            g[:nphys] = d['g']
        else:
            E = d['E_erg']
            g = d['g']
        self._table = {
            'A': d['A'],
            'E_erg': E,
            'g': g,
            'T_grid': d['T_grid'],
            'Upsilon_e': d['Upsilon_e'],
            'Upsilon_p': d.get(
                'Upsilon_p', np.zeros_like(d['Upsilon_e'])),
            'nlev_phys': nphys,
        }

    def table(self):
        if self._table is None:
            self._load()
        return self._table

    def _interp_upsilon_kind(self, T, kind):
        tab = self.table()
        T_grid = tab['T_grid']
        Y_grid = tab['Upsilon_e' if kind == 'e' else 'Upsilon_p']
        T_arr = np.atleast_1d(np.asarray(T, dtype=float))
        cell_shape = T_arr.shape
        out = np.zeros((NLEV, NLEV) + cell_shape)
        lnT = np.log(T_arr)
        lnT_grid = np.log(T_grid)
        for i in range(NLEV):
            for j in range(NLEV):
                Y_ij = Y_grid[i, j, :]
                if np.all(Y_ij == 0.0):
                    continue
                out[i, j] = np.interp(
                    lnT, lnT_grid, Y_ij,
                    left=Y_ij[0], right=Y_ij[-1])
        if np.isscalar(T):
            out = out[..., 0]
        return out

    def _collisional_q_down(self, T, n_e, n_p=None):
        """Downward collisional rate matrix [s^-1], shape
        (5, 5, ...). Electron + proton sum via Draine 17.10.
        """
        tab = self.table()
        g = tab['g']
        T_arr = np.asarray(T, dtype=float)
        sqrt_T = np.sqrt(T_arr)
        n_e_arr = np.asarray(n_e, dtype=float)
        n_p_arr = n_e_arr if n_p is None else np.asarray(
            n_p, dtype=float)
        Y_e = self._interp_upsilon_kind(T, 'e')
        Y_p = self._interp_upsilon_kind(T, 'p')
        Cdown = np.zeros_like(Y_e)
        for i in range(NLEV):
            for j in range(NLEV):
                if i <= j:
                    continue
                pref = _BETA / (g[i] * sqrt_T)
                Cdown[i, j] = pref * (
                    Y_e[i, j] * n_e_arr + Y_p[i, j] * n_p_arr)
        return Cdown

    def populations(self, T, n_e, n_p=None):
        """Return (5, ...) array of level fractional populations.
        Inputs can be scalars or numpy arrays.
        """
        tab = self.table()
        Cdown = self._collisional_q_down(T, n_e, n_p=n_p)
        return solve_5level_steady_state(
            tab['A'], tab['E_erg'], tab['g'], T, Cdown,
        )

    def cooling(self, T, n_e, n_ion, n_p=None):
        """Volumetric cooling rate [erg cm^-3 s^-1] from this ion.

        Sum over all radiative transitions in the 5-level system,
        weighted by level populations and emitted photon energy.
        """
        tab = self.table()
        f = self.populations(T, n_e, n_p=n_p)
        return n_ion * cooling_from_populations(
            f, tab['A'], tab['E_erg'])
