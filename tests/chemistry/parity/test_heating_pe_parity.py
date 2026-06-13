"""Parity test: `PhotoelectricHeating` vs
`pyathena.microphysics.cool.heatPE`.

`heatPE(nH, T, xe, Z_d, chi_PE)` uses the Weingartner & Draine 2001
fit (Table 2; Rv = 3.1, bC = 4.0, distribution A, ISRF) plus the
WD01 charge parameter

    x = chi_PE * sqrt(T) / n_e

at `phi = 1.0`. The new `PhotoelectricHeating` channel is a literal
port of the same formula and must reproduce it bit-exactly on a
representative grid.

Tolerance: `rtol = 1e-12`, `atol = 0`.
"""
from __future__ import annotations

import numpy as np
import pytest

from pyathena.chemistry.species import SpeciesSet
from pyathena.chemistry.state import ChemState
from pyathena.chemistry.heating.photoelectric import PhotoelectricHeating
from pyathena.microphysics.cool import heatPE as _heatPE_legacy


@pytest.fixture(scope='module')
def grid():
    T_vals = np.logspace(1.0, 4.0, 24)
    nH_vals = np.logspace(-2.0, 4.0, 16)
    T_grid, nH_grid = np.meshgrid(T_vals, nH_vals, indexing='xy')
    return T_grid.ravel(), nH_grid.ravel()


def _build_state(T, nH, xe, Z_d, chi_PE):
    species = SpeciesSet.ncr3_with_ghosts()
    ncell = T.size
    state = ChemState.from_grid(
        r=np.arange(ncell, dtype=np.float64),
        nH=nH.copy(),
        T=T.copy(),
        species=species,
        Z_d=float(Z_d),
    )
    idx = species.idx
    state.x[idx['HI']] = 1.0 - xe
    state.x[idx['HII']] = xe
    state.x[idx['electron']] = xe
    state.chi[state.chi_bands.index('FUV')] = chi_PE

    state.alloc_scratch('heating:photoelectric:tmp', (ncell,))
    state.alloc_scratch('heating:photoelectric:ne_floor', (ncell,))
    state.alloc_scratch('heating:photoelectric:eps_num', (ncell,))
    return state, species


@pytest.mark.parametrize('xe,Z_d,chi_PE', [
    (0.01, 1.0, 1.0),       # warm neutral medium, solar Z_d
    (0.1, 1.0, 10.0),       # closer to an HII region edge
    (1.0e-3, 0.1, 0.3),     # low metallicity, attenuated FUV
])
def test_pe_channel_matches_heatPE(grid, xe, Z_d, chi_PE):
    T, nH = grid
    state, species = _build_state(T, nH, xe, Z_d, chi_PE)
    out = np.empty_like(T)
    channel = PhotoelectricHeating(i_electron=species.idx['electron'])
    channel.evaluate(state, out)

    expected = _heatPE_legacy(
        nH=nH, T=T,
        xe=np.full_like(T, xe),
        Z_d=Z_d,
        chi_PE=np.full_like(T, chi_PE),
    )

    np.testing.assert_allclose(out, expected, rtol=1.0e-12, atol=0.0)
