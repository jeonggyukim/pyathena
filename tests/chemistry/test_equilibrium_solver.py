"""Regression tests for `pyathena.chemistry.equilibrium_solver_v2.solve_equilibrium`."""
from __future__ import annotations

import numpy as np
import pytest

from pyathena.chemistry.species import SpeciesSet
from pyathena.chemistry.state import (
    ChemState, U_RAD_PE_ISRF_CGS, U_RAD_LW_ISRF_CGS,
)
from pyathena.chemistry.networks.ncr3 import NCRNetwork3
from pyathena.chemistry.equilibrium_solver_v2 import solve_equilibrium


_CELLS = [
    dict(label='cold mol (nH=1e3)', T=20.0, nH=1.0e3, xi_CR=2.0e-16,
         zeta_pi_HI=0.0, chi_FUV=1.0, xH2_init=1.0e-4,
         zeta_diss_H2=0.0, tlim_yr=1.0e5),
    dict(label='cold mol (nH=1e6)', T=20.0, nH=1.0e6, xi_CR=2.0e-16,
         zeta_pi_HI=0.0, chi_FUV=1.0, xH2_init=1.0e-4,
         zeta_diss_H2=0.0, tlim_yr=1.0e4),
    dict(label='CNM', T=100.0, nH=3.0e1, xi_CR=2.0e-16,
         zeta_pi_HI=0.0, chi_FUV=1.0, xH2_init=1.0e-4,
         zeta_diss_H2=5.7e-11, tlim_yr=1.0e8),
    dict(label='WNM', T=8000.0, nH=0.5, xi_CR=2.0e-16,
         zeta_pi_HI=0.0, chi_FUV=1.0, xH2_init=1.0e-4,
         zeta_diss_H2=5.7e-11, tlim_yr=5.0e7),
    dict(label='HII region', T=1.0e4, nH=1.0e1, xi_CR=2.0e-16,
         zeta_pi_HI=3.0e-9, chi_FUV=10.0, xH2_init=1.0e-4,
         zeta_diss_H2=5.7e-10, tlim_yr=5.0e4),
]

_NEWTON_MAX_ITER = 60


def _arr(cells, key):
    return np.array([c[key] for c in cells], dtype=np.float64)


def _build_strip(cells=None):
    """Build a ChemState strip from `cells` (default: all `_CELLS`).
    Each cell is the dict layout above; pass `cells=[_CELLS[k]]` for
    a 1-cell strip."""
    if cells is None:
        cells = _CELLS
    cells = list(cells)
    species = SpeciesSet.ncr3_with_ghosts()
    network = NCRNetwork3()
    state = ChemState.from_grid(
        r=np.arange(len(cells), dtype=np.float64),
        nH=_arr(cells, 'nH'), T=_arr(cells, 'T'),
        species=species, Z_d=1.0, Z_g=1.0,
    )
    chi_FUV = _arr(cells, 'chi_FUV')
    state.xi_CR = _arr(cells, 'xi_CR')
    state.zeta_pi = {'HI': _arr(cells, 'zeta_pi_HI')}
    state.zeta_diss = {'H2': _arr(cells, 'zeta_diss_H2')}
    state.u_rad = {
        'PE': chi_FUV * U_RAD_PE_ISRF_CGS,
        'LW': chi_FUV * U_RAD_LW_ISRF_CGS,
    }
    idx = species.idx
    xH2 = _arr(cells, 'xH2_init')
    xHII = np.full_like(xH2, 1.0e-3)
    state.x[idx['H2']] = xH2
    state.x[idx['HII']] = xHII
    state.x[idx['HI']] = 1.0 - 2.0 * xH2 - xHII
    state.x[idx['electron']] = 1.6e-4
    network.closure(state)
    return state, network, species


def test_solver_converges_all_cells():
    """`solve_equilibrium` reaches `tol=1e-6` on every cell within
    `_NEWTON_MAX_ITER`, seeded via `network.seed_equilibrium`."""
    state, network, _ = _build_strip()
    network.seed_equilibrium(state)
    network.closure(state)
    result = solve_equilibrium(
        network, state, max_iter=_NEWTON_MAX_ITER, tol=1.0e-6)
    assert bool(np.all(result.converged)), (
        f'{(~result.converged).sum()}/{result.converged.size} not '
        f'converged; residual_inf = {result.residual_inf:.3e}')
