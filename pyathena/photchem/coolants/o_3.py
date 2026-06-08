"""O III cooling function: 5-level statistical equilibrium of the
2p^2 ground configuration (3P + 1D + 1S).

Uses the atomic data extracted offline from CHIANTI v11.0.2 in
`pyathena/photchem/data/o_3.txt`. At call time the per-T effective
collision strengths Upsilon(T) are interpolated to the local T,
converted to downward collisional rate coefficients q via Draine
2011 Eq. 17.10 (= Osterbrock 2006 Eq. 3.20), multiplied by the
electron density, and fed into the generic 5-level solver to get
populations. The cooling rate per O III ion is the standard
sum_{i>j} f_i * A[i,j] * E_ij.

LIMITATION -- PROTON IMPACT NOT INCLUDED.

CHIANTI ships proton-impact excitation rates (Psplups data) for
the 3P_J fine-structure transitions of O III. Proton rates for
these small-Delta-E transitions can equal or exceed electron rates
at HII temperatures, where n_p ~ n_e. The current implementation
uses electron impact only, so:
  - The IR fine-structure lines [O III] 52 + 88 um are
    underestimated at n_e < n_crit (~few * 1e3 cm^-3 for the FS
    transitions).
  - The dominant optical cooling channel [O III] 5008 + 4960 is
    set by electron excitation from the 3P_0 ground and is
    relatively accurate.
  - Total cooling per OIII ion can be off by up to ~3x in
    low-density HII gas (n_e ~ 1e3 cm^-3); converges to the
    correct value at n_e >> n_crit where collisions thermalize
    3P_J independent of collider.
Comparison vs ChiantiPy.populate() (which includes p + e) at
T=10^4 K, n_e=1e3 cm^-3:
  - f(3P_0): 0.97 (here) vs 0.31 (full); f(3P_2): 0.002 vs 0.20.
  - Total cooling per OIII: 4.6e-19 vs ~1.4e-18 erg / s.
Follow-up: extend the build script to extract psplups data and the
runtime to add a proton-impact contribution. Sufficient for an
HII-region SWEEP TARGET that focuses on 5008+4960 line emission;
not adequate for FIR fine-structure diagnostics.

Public API:
    cooling(T, n_e, n_OIII)
        Return the volumetric cooling rate [erg cm^-3 s^-1] from
        O III via electron impact excitation. T, n_e, n_OIII can
        be scalars or numpy arrays (broadcast together).

    populations(T, n_e)
        Return the (5, ...) array of fractional populations
        f_J=2, f_J=1, f_J=0, f_J(1D2), f_J(1S0).
"""

import os
import numpy as np

from ..n_level import (
    solve_5level_steady_state,
    cooling_from_populations,
    KB_CGS,
    NLEV,
)
from ..data.build_chianti_tables import read_ascii


# Lazy-load the atomic data at module import time.
_DATA_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'data', 'o_3.txt',
)


def _load():
    d = read_ascii(_DATA_FILE)
    # Pad to NLEV=5 if the file has fewer (not the case for O III).
    nphys = int(d.get('nlev_phys', len(d['E_erg'])))
    if nphys < NLEV:
        # Place padding levels at huge energy with zero rates.
        E = np.full(NLEV, np.inf)
        E[:nphys] = d['E_erg']
        E[nphys:] = 1.0e7 * 1.986e-16   # ~ 1e7 cm^-1 in erg
        g = np.ones(NLEV)
        g[:nphys] = d['g']
    else:
        E = d['E_erg']
        g = d['g']
    return {
        'A': d['A'],
        'E_erg': E,
        'g': g,
        'T_grid': d['T_grid'],
        'Upsilon_e': d['Upsilon_e'],
    }


# Module-level constants populated on first use.
_TABLE = None
# Prefactor `beta` in Draine 2011 Eq 17.10 for electron impact
# rates: q_ji = (beta / (g_j * sqrt(T))) * Upsilon_ji(T).
# beta = h^2 / [(2 pi m_e)^(3/2) * sqrt(k_B)] = 8.629e-8 (cgs).
_BETA = 8.629e-8


def _table():
    global _TABLE
    if _TABLE is None:
        _TABLE = _load()
    return _TABLE


def _interp_upsilon(T):
    """Interpolate the (5, 5, NT) Upsilon table to per-cell T.

    Returns an array of shape (5, 5, ...) where the trailing dims
    match the shape of T.
    """
    tab = _table()
    T_grid = tab['T_grid']
    Y_grid = tab['Upsilon_e']                       # (5, 5, NT)
    T_arr = np.atleast_1d(np.asarray(T, dtype=float))
    cell_shape = T_arr.shape
    out = np.zeros((NLEV, NLEV) + cell_shape)
    # Linear interpolation in log T for each (i, j) transition.
    lnT = np.log(T_arr)
    lnT_grid = np.log(T_grid)
    for i in range(NLEV):
        for j in range(NLEV):
            Y_ij = Y_grid[i, j, :]
            if np.all(Y_ij == 0.0):
                continue
            out[i, j] = np.interp(lnT, lnT_grid, Y_ij,
                                  left=Y_ij[0], right=Y_ij[-1])
    if np.isscalar(T):
        # Squeeze the (1,) trailing dim back out
        out = out[..., 0]
    return out


def _collisional_q_down(T, n_e):
    """Per-cell downward rate matrix `n_e * q_ij` for electron
    impact, shape (5, 5, ...).
    """
    tab = _table()
    g = tab['g']
    Y = _interp_upsilon(T)                          # (5, 5, ...)
    T_arr = np.asarray(T, dtype=float)
    sqrt_T = np.sqrt(T_arr)
    # Draine 17.10: q_ji = beta * Upsilon_ji / (g_j * sqrt(T)).
    # Note that 'j' here is the UPPER level in our convention
    # (de-excitation is i_upper -> i_lower; rate keyed by upper g).
    NL = NLEV
    Cdown = np.zeros_like(Y)
    n_e_arr = np.asarray(n_e, dtype=float)
    for i in range(NL):
        for j in range(NL):
            if i <= j:
                continue                            # only downward
            Cdown[i, j] = (_BETA / (g[i] * sqrt_T)) * Y[i, j] * n_e_arr
    return Cdown


def populations(T, n_e):
    """Return the 5-level fractional populations of O III at the
    given (T, n_e). Inputs can be scalars or arrays.
    """
    tab = _table()
    Cdown = _collisional_q_down(T, n_e)
    return solve_5level_steady_state(
        tab['A'], tab['E_erg'], tab['g'], T, Cdown,
    )


def cooling(T, n_e, n_OIII):
    """Volumetric cooling rate from O III [erg cm^-3 s^-1].

    Lambda(O III) = n_OIII * sum_{i>j} f_i * A[i,j] * (E_i - E_j),
    where f_i are level populations from the 5-level solve. The
    sum-over-emission-lines includes:
      - [O III] 5008 + 4960 (1D2 -> 3P_2 + 3P_1)
      - [O III] 4364 (1S0 -> 1D2; auroral, T_e diagnostic with 5008)
      - IR fine-structure (3P_J transitions, far-IR)
    """
    tab = _table()
    f = populations(T, n_e)
    return n_OIII * cooling_from_populations(f, tab['A'], tab['E_erg'])
