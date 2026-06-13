"""Parity test: `LymanAlphaCooling` vs `pyathena.microphysics.cool.coolHI`.

The new channel-policy implementation must reproduce the legacy
`coolHI(nH, T, xHI, xe)` function bit-exactly on a representative
(T, n_H) grid because the implementation is a literal port of the
DESPOTIC formula (Krumholz 2014, ApJS 211, 19; Draine 2011 11.32 /
11.34 / 11.36) with identical constants.

Tolerance: `rtol = 1e-12`, `atol = 0`. Anything looser would indicate
either a constant rounded differently or a missing line.
"""
from __future__ import annotations

import numpy as np
import pytest

from pyathena.chemistry.species import SpeciesSet
from pyathena.chemistry.state import ChemState
from pyathena.chemistry.cooling.lyman_alpha import LymanAlphaCooling
from pyathena.microphysics.cool import coolHI as _coolHI_legacy


@pytest.fixture(scope='module')
def grid():
    """A (n_T, n_n) Cartesian product spanning the H I cooling regime
    (T = 100 K to 1e6 K, n_H = 0.01 to 1e4 cm^-3) flattened to a
    single strip.
    """
    T_vals = np.logspace(2.0, 6.0, 30)
    nH_vals = np.logspace(-2.0, 4.0, 20)
    T_grid, nH_grid = np.meshgrid(T_vals, nH_vals, indexing='xy')
    return T_grid.ravel(), nH_grid.ravel()


def _build_state(T, nH, xHI, xe):
    """Construct a minimal ChemState that the channel can evaluate
    against. Uses `SpeciesSet.ncr3_with_ghosts()` so the electron
    ghost row carries the test xe.
    """
    species = SpeciesSet.ncr3_with_ghosts()
    ncell = T.size
    state = ChemState.from_grid(
        r=np.arange(ncell, dtype=np.float64),
        nH=nH.copy(),
        T=T.copy(),
        species=species,
    )
    idx = species.idx
    state.x[idx['HI']] = xHI
    state.x[idx['HII']] = 1.0 - xHI
    state.x[idx['H2']] = 0.0
    state.x[idx['electron']] = xe
    # Other ghost rows do not enter the Lyman-alpha formula; leave
    # them as their factory defaults.

    # Allocate the channel's owned scratch slots so evaluate() runs.
    state.alloc_scratch('cooling:lyman_alpha:tmp', (ncell,))
    state.alloc_scratch('cooling:lyman_alpha:prefac', (ncell,))
    return state, species


@pytest.mark.parametrize('xHI,xe', [
    (0.5, 0.5),    # half-ionised; bulk of HII region thermal cooling
    (0.9, 0.1),    # mostly neutral, traces of free electrons
    (0.99, 0.01),  # WNM-like
])
def test_lyman_alpha_channel_matches_coolHI(grid, xHI, xe):
    T, nH = grid
    state, species = _build_state(T, nH, xHI, xe)
    out = np.empty_like(T)
    channel = LymanAlphaCooling(
        i_HI=species.idx['HI'], i_electron=species.idx['electron'],
    )
    channel.evaluate(state, out)

    expected = _coolHI_legacy(
        nH=nH, T=T,
        xHI=np.full_like(T, xHI),
        xe=np.full_like(T, xe),
    )

    np.testing.assert_allclose(out, expected, rtol=1.0e-12, atol=0.0)


def test_lyman_alpha_d_out_is_zero():
    """Phase 4a contract: the channel writes a zero derivative when
    asked, so the dispatcher works but the semi-implicit damping is
    a weak under-estimate. Phase 4b will replace this with the
    analytic value.
    """
    T = np.array([1.0e4, 5.0e4, 1.0e5])
    nH = np.array([1.0, 10.0, 100.0])
    state, species = _build_state(T, nH, 0.5, 0.5)
    out = np.empty_like(T)
    d_out = np.full_like(T, 999.0)
    channel = LymanAlphaCooling(
        i_HI=species.idx['HI'], i_electron=species.idx['electron'],
    )
    channel.evaluate(state, out, d_out)
    np.testing.assert_array_equal(d_out, np.zeros_like(T))
