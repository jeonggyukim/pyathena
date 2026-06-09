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

Both electron and proton impact are included. CHIANTI ships
proton-impact excitation rates (Psplups data) for the small-Delta-E
fine-structure transitions of O III (3P_J -> 3P_J'). For HII
conditions (T ~ 1e4 K, n_p ~ n_e), proton impact dominates the FS
thermalization. Both contributions are summed at runtime via the
same Draine 2011 Eq. 17.10 formula on the per-collider Upsilon
tables. Pass `n_p` explicitly if it differs from `n_e`; default is
`n_p = n_e` (charge neutrality with metals being a small
correction in HII gas).

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
        'Upsilon_p': d.get('Upsilon_p',
                           np.zeros_like(d['Upsilon_e'])),
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


def _interp_upsilon_kind(T, kind):
    """Interpolate one of the (5, 5, NT) Upsilon tables to per-cell
    T. `kind` is 'e' (electron) or 'p' (proton).
    """
    tab = _table()
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
            out[i, j] = np.interp(lnT, lnT_grid, Y_ij,
                                  left=Y_ij[0], right=Y_ij[-1])
    if np.isscalar(T):
        out = out[..., 0]
    return out


def _collisional_q_down(T, n_e, n_p=None):
    """Per-cell downward rate matrix summed over collider
    contributions, shape (5, 5, ...).

    Electron impact and (optional) proton impact use the same
    Draine 2011 Eq. 17.10 conversion of Upsilon to q with the same
    prefactor `beta = 8.629e-8` (the kinematics of the electron
    cancel out in the rate-coefficient form), only the table of
    Upsilon differs. For HII gas where the protons supply most of
    the FS thermalization, leaving `n_p = None` defaults to
    `n_p = n_e` (charge neutrality with metals being a small
    correction).
    """
    tab = _table()
    g = tab['g']
    T_arr = np.asarray(T, dtype=float)
    sqrt_T = np.sqrt(T_arr)
    n_e_arr = np.asarray(n_e, dtype=float)
    if n_p is None:
        n_p_arr = n_e_arr
    else:
        n_p_arr = np.asarray(n_p, dtype=float)

    Y_e = _interp_upsilon_kind(T, 'e')
    Y_p = _interp_upsilon_kind(T, 'p')
    Cdown = np.zeros_like(Y_e)
    for i in range(NLEV):
        for j in range(NLEV):
            if i <= j:
                continue
            pref = _BETA / (g[i] * sqrt_T)
            Cdown[i, j] = pref * (Y_e[i, j] * n_e_arr
                                  + Y_p[i, j] * n_p_arr)
    return Cdown


def populations(T, n_e, n_p=None):
    """Return the 5-level fractional populations of O III at the
    given (T, n_e [, n_p]). `n_p` defaults to `n_e`.
    """
    tab = _table()
    Cdown = _collisional_q_down(T, n_e, n_p=n_p)
    return solve_5level_steady_state(
        tab['A'], tab['E_erg'], tab['g'], T, Cdown,
    )


def cooling(T, n_e, n_OIII, n_p=None):
    """Volumetric cooling rate from O III [erg cm^-3 s^-1].

    Lambda(O III) = n_OIII * sum_{i>j} f_i * A[i,j] * (E_i - E_j),
    where f_i are level populations from the 5-level solve. The
    sum-over-emission-lines includes:
      - [O III] 5008 + 4960 (1D2 -> 3P_2 + 3P_1)
      - [O III] 4364 (1S0 -> 1D2; auroral, T_e diagnostic with 5008)
      - IR fine-structure (3P_J transitions, far-IR)
    Both electron and proton impact are included; pass `n_p`
    explicitly if it differs from `n_e` (in HII regions
    `n_p = n_HII ~= n_e` is the default).
    """
    tab = _table()
    f = populations(T, n_e, n_p=n_p)
    return n_OIII * cooling_from_populations(f, tab['A'], tab['E_erg'])
