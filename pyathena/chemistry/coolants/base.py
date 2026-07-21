r"""Generic 5-level coolant module.

Each per-ion file under this package (e.g., `o_3.py`, `n_2.py`)
does:

    from .base import IonCoolant
    coolant = IonCoolant('o_3.txt', label='OIII')
    populations = coolant.populations
    cooling = coolant.cooling

The class loads per-ion atomic data, interpolates Upsilon, and
solves the 5-level steady-state populations. All ions use the
same code; only the data file differs.

Atomic-physics summary
======================

A bound electron is excited from level $j$ (lower) to level $i$
(upper) by collision with a thermal electron or proton. From the
upper level it either decays radiatively at rate $A_{ij}$,
emitting a photon of energy $E_i - E_j$, or is collisionally
de-excited before it can radiate.

Statistical equilibrium fixes the level populations
$f_i \equiv n_i / n_X$ by balancing every collisional and
radiative excitation against every collisional and radiative
de-excitation:

$$
\sum_{j \neq i} \left[ n_e \, q_{ji}^{\rm up} f_j
                     + n_e \, q_{ij}^{\rm down} f_i \,\delta_{j<i}
                     + A_{ji} f_j \,\delta_{j>i}
                 \right]
= \sum_{j \neq i} \left[ A_{ij} f_i \,\delta_{j<i}
                     + n_e \, q_{ij}^{\rm down} f_i \,\delta_{j<i}
                     + n_e \, q_{ji}^{\rm up} f_j
                 \right] ,
$$

with closure $\sum_i f_i = 1$. For a 5-level system this is a
$5 \times 5$ linear system per cell solved by
`n_level.solve_5level_steady_state`. Once $f_i$ is known the
volumetric cooling rate is

$$
\Lambda = n_X \sum_{i>j} f_i \, A_{ij} \, (E_i - E_j)
\quad [\text{erg cm}^{-3}\,\text{s}^{-1}] .
$$

The per-ion-per-electron cooling efficiency reported by
`cooling()` is this rate divided by $n_e \, n_X$.

Rate coefficient conventions (Draine 2011 ch. 17)
=================================================

* **Downward** (collisional de-excitation, upper $i \to$ lower $j$):

  $$
  q_{ij}^{\rm down}(T)
  = \frac{\beta \, \Upsilon_{ij}(T)}{g_i \sqrt{T}}
  \quad [\text{cm}^{3}\,\text{s}^{-1}] ,
  $$

  where $g_i = 2 J_i + 1$ is the statistical weight of the upper
  level and $\Upsilon_{ij}(T)$ is the Burgess-Tully effective
  collision strength tabulated by CHIANTI.

* **Upward** (collisional excitation, lower $j \to$ upper $i$) by
  detailed balance,

  $$
  q_{ji}^{\rm up}(T)
  = q_{ij}^{\rm down}(T)\,\frac{g_i}{g_j}\,
    \exp\!\left(-\frac{E_i - E_j}{k_B T}\right)
  \quad [\text{cm}^{3}\,\text{s}^{-1}] .
  $$

* Per-cell rate $[\text{s}^{-1}]$ is $q \, n_e$ for electron
  impact, $q \, n_p$ for proton impact; the two contributions add
  linearly (Draine 2011 Eq. 17.10).

The prefactor

$$
\beta = \frac{h^2}{\sqrt{8 \pi^3 \, m_e^3 \, k_B}}
    \simeq 8.629 \times 10^{-6}
\quad [\text{cm}^{3}\,\text{s}^{-1}\,\text{K}^{1/2}]
$$

is computed at module import from `astropy.constants` so a CODATA
update flows through without touching this file.

Dummy padding for low-physical-level ions
=========================================

`NLEV` is fixed at 5. Ions with fewer physical levels (e.g., the
2-level fine-structure ions C II ${}^2P$, Ne II ${}^2P$,
N III ${}^2P$) get the remaining rows padded with dummy levels at
$E = 10^7\,\text{cm}^{-1}$, $g = 1$, and zero rates. The n-level
solver detects them (zero row + zero column in the rate matrix)
and forces $f_{\rm dummy} = 0$ so the matrix stays non-singular.
The dummy rows contribute nothing to cooling.

Reference frame for the per-ion files
=====================================

The text files in `data/microphysics/chianti_v11/` are produced
offline by `pyathena.chemistry.tables.chianti_v11.build_atomic`
and carry, for each ion: the energy levels $E_i$ in erg, the
statistical weights $g_i$, the radiative-decay matrix $A_{ij}$ in
$\text{s}^{-1}$, the temperature grid $T_{\rm grid}$ in K, and the
electron and proton Burgess-Tully $\Upsilon_{ij}(T)$ tables.

The docstring uses MyST-flavoured Markdown math (`$...$` inline,
`$$...$$` block) so equations render in Jupyter / JupyterBook /
GitHub renderer / VS Code preview without a separate math
extension.
"""

import os
import numpy as np

from .n_level import (
    solve_5level_steady_state,
    cooling_from_populations,
    NLEV,
)
from ..tables.chianti_v11.build_atomic import read_ascii


# Prefactor in Draine 2011 Eq. 17.10 / Osterbrock 2006 Eq. 3.20
# Collisional de-excitation rate coefficient (upper -> lower):
#   q_ji^down(T) = _BETA * Upsilon / (g_upper * sqrt(T))   [cm^3 / s]
# beta = h^2 / sqrt(8 pi^3 m_e^3 k_B) in CGS (Draine 2011 Eq 17.10
# with T in K; equals 8.629e-6).
from astropy import constants as _const
_BETA = (_const.h.cgs.value ** 2
         / np.sqrt(8.0 * np.pi ** 3
                   * _const.m_e.cgs.value ** 3
                   * _const.k_B.cgs.value))


class IonCoolant:
    """Single-ion 5-level cooling function. Atomic data is read
    from a text file in `data/microphysics/chianti_v11/`.

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
        # Per-ion atomic data is stored at
        # data/microphysics/chianti_v11/ next to the other CHIANTI
        # tables (ioneq + cool). This module is at
        # pyathena/microphysics/coolants/base.py; the path goes up
        # three directories to the repo root.
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
