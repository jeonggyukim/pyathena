"""Grouped parity tests for the Phase 4b batch 4b channels.

Covers:

- `cooling.h2_colldiss.H2CollDissCooling`     vs `cool.coolH2colldiss`
- `cooling.oii.OIIFineStructureCooling`        vs `cool.coolOII`
- `cooling.nebular.NebularMetalLineCooling`    vs `cool.coolneb`
- `cooling.hi_smith21.HISmith21Cooling`        vs `cool.coolHISmith21`

Tolerance: rtol = 1e-12, atol = 0. One test per channel; looping
representative cases internally per the project pytest grouping
convention.
"""
from __future__ import annotations

import numpy as np

from pyathena.chemistry.species import SpeciesSet
from pyathena.chemistry.state import ChemState
from pyathena.chemistry.cooling.h2_colldiss import H2CollDissCooling
from pyathena.chemistry.cooling.oii import OIIFineStructureCooling
from pyathena.chemistry.cooling.nebular import NebularMetalLineCooling
from pyathena.chemistry.cooling.hi_smith21 import HISmith21Cooling

from pyathena.microphysics.cool import (
    coolH2colldiss as _coolH2colldiss_legacy,
    coolOII as _coolOII_legacy,
    coolneb as _coolneb_legacy,
    coolHISmith21 as _coolHISmith21_legacy,
)


_T_VALS = np.logspace(2.0, 7.0, 30)
_N_VALS = np.logspace(-2.0, 4.0, 16)
_T_GRID, _NH_GRID = np.meshgrid(_T_VALS, _N_VALS, indexing='xy')
_T = _T_GRID.ravel()
_NH = _NH_GRID.ravel()

_SCRATCH = (
    # H2 colldiss
    'cooling:h2_colldiss:Tinv', 'cooling:h2_colldiss:logT4',
    'cooling:h2_colldiss:sqrtT',
    'cooling:h2_colldiss:k9l', 'cooling:h2_colldiss:k9h',
    'cooling:h2_colldiss:k10l', 'cooling:h2_colldiss:k10h',
    'cooling:h2_colldiss:ncrH2', 'cooling:h2_colldiss:ncrHI',
    'cooling:h2_colldiss:n2ncr',
    'cooling:h2_colldiss:k_H2_HI', 'cooling:h2_colldiss:k_H2_H2',
    'cooling:h2_colldiss:tmp_a', 'cooling:h2_colldiss:tmp_b',
    'cooling:h2_colldiss:gate',
    # OII
    'cooling:oii:T4', 'cooling:oii:lnT4',
    'cooling:oii:prefac', 'cooling:oii:tmp_a',
    'cooling:oii:Omega',
    'cooling:oii:q10', 'cooling:oii:q20', 'cooling:oii:q21',
    'cooling:oii:q01', 'cooling:oii:q02', 'cooling:oii:q12',
    'cooling:oii:tmp0', 'cooling:oii:tmp1', 'cooling:oii:tmp2',
    # Nebular
    'cooling:neb:T4', 'cooling:neb:lnT4',
    'cooling:neb:poly_fit', 'cooling:neb:f_red',
    'cooling:neb:tmp_a',
    # Smith21
    'cooling:hi_smith21:Tinv',
    'cooling:hi_smith21:T6', 'cooling:hi_smith21:T6_SQR',
    'cooling:hi_smith21:T6_CUB',
    'cooling:hi_smith21:Upsilon', 'cooling:hi_smith21:u_cold',
    'cooling:hi_smith21:total', 'cooling:hi_smith21:tmp_a',
    'cooling:hi_smith21:mask_cold',
    'cooling:hi_smith21:mask_hot',
)


def _build_state(T, nH, xHI, xH2, xHII, xe, xOII=3.2e-4, Z_g=1.0):
    species = SpeciesSet.ncr3_with_ghosts()
    ncell = T.size
    state = ChemState.from_grid(
        r=np.arange(ncell, dtype=np.float64),
        nH=nH.copy(), T=T.copy(),
        species=species, Z_g=float(Z_g),
    )
    idx = species.idx
    state.x[idx['HI']] = xHI
    state.x[idx['HII']] = xHII
    state.x[idx['H2']] = xH2
    state.x[idx['electron']] = xe
    state.x[idx['OII']] = xOII
    for name in _SCRATCH:
        state.alloc_scratch(name, (ncell,))
    return state, species


_CASES = (
    # (xHI, xH2, xHII, xe)
    (0.99, 0.0, 0.005, 0.005),
    (0.40, 0.30, 0.01, 0.05),
    (0.10, 0.45, 0.00, 0.02),
    (0.50, 0.0, 0.50, 0.50),
)


def test_h2_colldiss_parity():
    """coolH2colldiss vs H2CollDissCooling, rtol = 1e-12.

    Suppresses RuntimeWarnings for log10 underflow in cold cells;
    both the channel and the legacy code emit them but the final
    output is zero by the T > 700 K gate.
    """
    for xHI, xH2, xHII, xe in _CASES:
        state, species = _build_state(_T, _NH, xHI, xH2, xHII, xe)
        out = np.empty_like(_T)
        ch = H2CollDissCooling(
            i_HI=species.idx['HI'],
            i_H2=species.idx['H2'],
        )
        with np.errstate(divide='ignore', invalid='ignore'):
            ch.evaluate(state, out)
            expected = _coolH2colldiss_legacy(
                nH=_NH, T=_T,
                xHI=np.full_like(_T, xHI),
                xH2=np.full_like(_T, xH2),
            )
        np.testing.assert_allclose(
            out, expected, rtol=1.0e-12, atol=0.0,
            err_msg=f'(xHI={xHI}, xH2={xH2})',
        )


def test_oii_fine_structure_parity():
    """coolOII vs OIIFineStructureCooling, rtol = 1e-12."""
    xOII_values = (1.0e-5, 3.2e-4)
    for xHI, xH2, xHII, xe in _CASES:
        for xOII in xOII_values:
            state, species = _build_state(
                _T, _NH, xHI, xH2, xHII, xe, xOII=xOII)
            out = np.empty_like(_T)
            ch = OIIFineStructureCooling(
                i_OII=species.idx['OII'],
                i_electron=species.idx['electron'],
            )
            ch.evaluate(state, out)
            expected = _coolOII_legacy(
                nH=_NH, T=_T,
                xe=np.full_like(_T, xe),
                xOII=np.full_like(_T, xOII),
            )
            np.testing.assert_allclose(
                out, expected, rtol=1.0e-12, atol=0.0,
                err_msg=f'(xe={xe}, xOII={xOII:.0e})',
            )


def test_nebular_parity():
    """coolneb vs NebularMetalLineCooling, rtol = 1e-12."""
    for xHI, xH2, xHII, xe in _CASES:
        for Z_g in (0.1, 1.0):
            state, species = _build_state(
                _T, _NH, xHI, xH2, xHII, xe, Z_g=Z_g)
            out = np.empty_like(_T)
            ch = NebularMetalLineCooling(
                i_HII=species.idx['HII'],
                i_electron=species.idx['electron'],
            )
            ch.evaluate(state, out)
            expected = _coolneb_legacy(
                nH=_NH, T=_T,
                xe=np.full_like(_T, xe),
                xHII=np.full_like(_T, xHII),
                Z_g=Z_g,
            )
            np.testing.assert_allclose(
                out, expected, rtol=1.0e-12, atol=1.0e-300,
                err_msg=f'(xHII={xHII}, xe={xe}, Z_g={Z_g})',
            )


def test_hi_smith21_parity():
    """coolHISmith21 vs HISmith21Cooling, rtol = 1e-12."""
    for xHI, xH2, xHII, xe in _CASES:
        state, species = _build_state(_T, _NH, xHI, xH2, xHII, xe)
        out = np.empty_like(_T)
        ch = HISmith21Cooling(
            i_HI=species.idx['HI'],
            i_electron=species.idx['electron'],
        )
        ch.evaluate(state, out)
        expected = _coolHISmith21_legacy(
            nH=_NH, T=_T,
            xe=np.full_like(_T, xe),
            xHI=np.full_like(_T, xHI),
        )
        np.testing.assert_allclose(
            out, expected, rtol=1.0e-12, atol=0.0,
            err_msg=f'(xHI={xHI}, xe={xe})',
        )
