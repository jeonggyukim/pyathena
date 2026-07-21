"""Grouped parity tests for the Phase 4b batch 2 channels.

Covers:

- `cooling.cii.CIIFineStructureCooling` vs `cool.coolCII`
- `cooling.oi.OIFineStructureCooling`  vs `cool.coolOI`
- `cooling.dust.DustGasCoupling`       vs `cool.cooldust`

Tolerance: `rtol = 1e-12`, `atol = 0`. Per the pytest grouping
convention, each channel gets one test function looping over
representative cases.
"""
from __future__ import annotations

import numpy as np

from pyathena.chemistry.species import SpeciesSet
from pyathena.chemistry.state import ChemState
from pyathena.chemistry.cooling.cii import CIIFineStructureCooling
from pyathena.chemistry.cooling.oi import OIFineStructureCooling
from pyathena.chemistry.cooling.dust import DustGasCoupling

from pyathena.microphysics.cool import (
    coolCII as _coolCII_legacy,
    coolOI as _coolOI_legacy,
    cooldust as _cooldust_legacy,
)


_T_VALS = np.logspace(1.0, 4.0, 28)
_N_VALS = np.logspace(-2.0, 4.0, 18)
_T_GRID, _NH_GRID = np.meshgrid(_T_VALS, _N_VALS, indexing='xy')
_T = _T_GRID.ravel()
_NH = _NH_GRID.ravel()


# Phase 4b batch 2 scratch namespaces required across the three
# channels.
_SCRATCH_SLOTS = (
    'cooling:cii:T2', 'cooling:cii:tmp', 'cooling:cii:tmp_b',
    'cooling:cii:k10e', 'cooling:cii:k10HI', 'cooling:cii:k10H2',
    'cooling:cii:q10', 'cooling:cii:q01',
    'cooling:cii:warm_mask', 'cooling:cii:cold_mask',
    'cooling:oi:T2', 'cooling:oi:lnT2',
    'cooling:oi:tmp', 'cooling:oi:tmp_o', 'cooling:oi:tmp_p',
    'cooling:oi:k_HI', 'cooling:oi:k_H2', 'cooling:oi:k_e',
    'cooling:oi:q10', 'cooling:oi:q20', 'cooling:oi:q21',
    'cooling:oi:q01', 'cooling:oi:q02', 'cooling:oi:q12',
    'cooling:oi:tmp0', 'cooling:oi:tmp1', 'cooling:oi:tmp2',
    'cooling:dust:tmp', 'cooling:dust:tmp_b',
)


def _build_state(T, nH, xHI, xH2, xe, T_dust=15.0, Z_d=1.0,
                 xCII=1.6e-4, xOI=3.2e-4):
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
    state.x[idx['HI']] = xHI
    state.x[idx['HII']] = 1.0 - xHI - 2.0 * xH2
    state.x[idx['H2']] = xH2
    state.x[idx['electron']] = xe
    state.x[idx['CII']] = xCII
    state.x[idx['OI']] = xOI
    state.T_dust[:] = T_dust

    for name in _SCRATCH_SLOTS:
        state.alloc_scratch(name, (ncell,))
    return state, species


_CASES = (
    # (xHI, xH2, xe)
    (0.99, 0.0, 0.01),
    (0.50, 0.0, 0.50),
    (0.40, 0.30, 0.05),
    (0.10, 0.45, 0.02),
)


def test_cii_fine_structure_parity():
    """coolCII vs CIIFineStructureCooling, rtol = 1e-12.

    The CII channel multiplies the steady-state f1 by `xCII`; sweep
    a couple of xCII values to confirm the linearity is preserved.
    """
    xCII_values = (1.0e-5, 1.6e-4, 5.0e-4)
    for xHI, xH2, xe in _CASES:
        for xCII in xCII_values:
            state, species = _build_state(
                _T, _NH, xHI, xH2, xe, xCII=xCII)
            out = np.empty_like(_T)
            ch = CIIFineStructureCooling(
                i_HI=species.idx['HI'],
                i_H2=species.idx['H2'],
                i_CII=species.idx['CII'],
                i_electron=species.idx['electron'],
            )
            ch.evaluate(state, out)
            expected = _coolCII_legacy(
                nH=_NH, T=_T,
                xe=np.full_like(_T, xe),
                xHI=np.full_like(_T, xHI),
                xH2=np.full_like(_T, xH2),
                xCII=np.full_like(_T, xCII),
            )
            np.testing.assert_allclose(
                out, expected, rtol=1.0e-12, atol=0.0,
                err_msg=f'(xHI={xHI}, xH2={xH2}, xCII={xCII:.0e})',
            )


def test_oi_fine_structure_parity():
    """coolOI vs OIFineStructureCooling, rtol = 1e-12."""
    xOI_values = (1.0e-5, 3.2e-4)
    for xHI, xH2, xe in _CASES:
        for xOI in xOI_values:
            state, species = _build_state(
                _T, _NH, xHI, xH2, xe, xOI=xOI)
            out = np.empty_like(_T)
            ch = OIFineStructureCooling(
                i_HI=species.idx['HI'],
                i_H2=species.idx['H2'],
                i_OI=species.idx['OI'],
                i_electron=species.idx['electron'],
            )
            ch.evaluate(state, out)
            expected = _coolOI_legacy(
                nH=_NH, T=_T,
                xe=np.full_like(_T, xe),
                xHI=np.full_like(_T, xHI),
                xH2=np.full_like(_T, xH2),
                xOI=np.full_like(_T, xOI),
            )
            np.testing.assert_allclose(
                out, expected, rtol=1.0e-12, atol=0.0,
                err_msg=f'(xHI={xHI}, xH2={xH2}, xOI={xOI:.0e})',
            )


def test_dust_gas_coupling_parity():
    """cooldust vs DustGasCoupling, rtol = 1e-12. Sweep dust and Z_d."""
    for xHI, xH2, xe in _CASES:
        for T_dust in (5.0, 15.0, 50.0):
            for Z_d in (0.1, 1.0):
                state, species = _build_state(
                    _T, _NH, xHI, xH2, xe,
                    T_dust=T_dust, Z_d=Z_d)
                out = np.empty_like(_T)
                ch = DustGasCoupling()
                ch.evaluate(state, out)
                expected = _cooldust_legacy(
                    nH=_NH, T=_T,
                    Td=np.full_like(_T, T_dust),
                    Z_d=Z_d,
                )
                np.testing.assert_allclose(
                    out, expected, rtol=1.0e-12, atol=0.0,
                    err_msg=f'(T_dust={T_dust}, Z_d={Z_d})',
                )
