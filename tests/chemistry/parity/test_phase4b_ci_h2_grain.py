"""Grouped parity tests for the Phase 4b batch 3 channels.

Covers (with the NCR production defaults flagged):

- `cooling.ci.CIFineStructureCooling`        vs `cool.coolCI`
- `cooling.lya.LyaCooling` (NCR default)     vs `cool.coolLya`
- `cooling.h2_gong17.H2Gong17Cooling` (alt)  vs `cool.coolH2G17`
- `cooling.h2_moseley21.H2Moseley21Cooling` (NCR default)
                                              vs `cool.coolH2rovib`
- `cooling.grain_recombination.GrainRecombinationCooling`
                                              vs `cool.coolRec`

Tolerance: `rtol = 1e-12`, `atol = 0`. One test per channel; each
loops representative cases internally.
"""
from __future__ import annotations

import numpy as np

from pyathena.chemistry.species import SpeciesSet
from pyathena.chemistry.state import ChemState
from pyathena.chemistry.cooling.ci import CIFineStructureCooling
from pyathena.chemistry.cooling.lya import LyaCooling
from pyathena.chemistry.cooling.h2_gong17 import H2Gong17Cooling
from pyathena.chemistry.cooling.h2_moseley21 import H2Moseley21Cooling
from pyathena.chemistry.cooling.grain_recombination import (
    GrainRecombinationCooling,
)

from pyathena.microphysics.cool import (
    coolCI as _coolCI_legacy,
    coolLya as _coolLya_legacy,
    coolH2G17 as _coolH2G17_legacy,
    coolH2rovib as _coolH2rovib_legacy,
    coolRec as _coolRec_legacy,
)


_T_VALS = np.logspace(1.0, 4.0, 28)
_N_VALS = np.logspace(-2.0, 4.0, 18)
_T_GRID, _NH_GRID = np.meshgrid(_T_VALS, _N_VALS, indexing='xy')
_T = _T_GRID.ravel()
_NH = _NH_GRID.ravel()


# Phase 4b batch 3 scratch slots.
_SCRATCH_SLOTS = (
    # CI
    'cooling:ci:T2', 'cooling:ci:lnT', 'cooling:ci:lnT2',
    'cooling:ci:tmp_a', 'cooling:ci:tmp_b',
    'cooling:ci:mask_cold', 'cooling:ci:mask_warm',
    'cooling:ci:gamma10', 'cooling:ci:gamma20', 'cooling:ci:gamma21',
    'cooling:ci:k_e_10', 'cooling:ci:k_e_20', 'cooling:ci:k_e_21',
    'cooling:ci:k_HI_10', 'cooling:ci:k_HI_20', 'cooling:ci:k_HI_21',
    'cooling:ci:k_H2_10', 'cooling:ci:k_H2_20', 'cooling:ci:k_H2_21',
    'cooling:ci:q10', 'cooling:ci:q20', 'cooling:ci:q21',
    'cooling:ci:q01', 'cooling:ci:q02', 'cooling:ci:q12',
    'cooling:ci:tmp0', 'cooling:ci:tmp1', 'cooling:ci:tmp2',
    'cooling:ci:fac',
    # H2 G17
    'cooling:h2_g17:T_eff', 'cooling:h2_g17:T3', 'cooling:h2_g17:y',
    'cooling:h2_g17:tmp', 'cooling:h2_g17:tmp_b',
    'cooling:h2_g17:Lpartner', 'cooling:h2_g17:Gamma_n0',
    'cooling:h2_g17:Gamma_LTE',
    'cooling:h2_g17:mask_cold', 'cooling:h2_g17:mask_warm',
    'cooling:h2_g17:mask_hot', 'cooling:h2_g17:mask_T_floor',
    # Grain rec
    'cooling:grain_rec:tmp', 'cooling:grain_rec:ne_floor',
    'cooling:grain_rec:lnx',
    # Lya
    'cooling:lya:T4', 'cooling:lya:fac', 'cooling:lya:ne',
    'cooling:lya:tmp_a', 'cooling:lya:tmp_b',
    'cooling:lya:v', 'cooling:lya:d_ln_fac',
    # H2 Moseley21
    'cooling:h2_moseley:T3', 'cooling:h2_moseley:T3inv',
    'cooling:h2_moseley:sqrtT3', 'cooling:h2_moseley:nHI',
    'cooling:h2_moseley:nH2', 'cooling:h2_moseley:tmp',
    'cooling:h2_moseley:x_eff', 'cooling:h2_moseley:accum',
    'cooling:h2_moseley:term', 'cooling:h2_moseley:ratio_lo',
    'cooling:h2_moseley:ratio_hi',
)


def _build_state(T, nH, xHI, xH2, xHII, xe, xCI=1.6e-4, Z_d=1.0,
                 chi_PE=1.0):
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
    state.x[idx['HII']] = xHII
    state.x[idx['H2']] = xH2
    state.x[idx['electron']] = xe
    state.x[idx['CI']] = xCI
    state.chi[state.chi_bands.index('FUV')] = chi_PE
    for name in _SCRATCH_SLOTS:
        state.alloc_scratch(name, (ncell,))
    return state, species


_CASES = (
    # (xHI, xH2, xHII, xe)
    (0.99, 0.0, 0.005, 0.005),    # cold neutral
    (0.40, 0.30, 0.01, 0.05),     # half molecular
    (0.10, 0.45, 0.00, 0.02),     # mostly molecular
    (0.50, 0.0, 0.50, 0.50),      # HII region edge
)


def test_ci_fine_structure_parity():
    """coolCI vs CIFineStructureCooling, rtol = 1e-12."""
    xCI_values = (1.0e-5, 1.6e-4)
    for xHI, xH2, xHII, xe in _CASES:
        for xCI in xCI_values:
            state, species = _build_state(
                _T, _NH, xHI, xH2, xHII, xe, xCI=xCI)
            out = np.empty_like(_T)
            ch = CIFineStructureCooling(
                i_HI=species.idx['HI'],
                i_H2=species.idx['H2'],
                i_CI=species.idx['CI'],
                i_electron=species.idx['electron'],
            )
            ch.evaluate(state, out)
            expected = _coolCI_legacy(
                nH=_NH, T=_T,
                xe=np.full_like(_T, xe),
                xHI=np.full_like(_T, xHI),
                xH2=np.full_like(_T, xH2),
                xCI=np.full_like(_T, xCI),
            )
            np.testing.assert_allclose(
                out, expected, rtol=1.0e-12, atol=0.0,
                err_msg=f'(xHI={xHI}, xH2={xH2}, xCI={xCI:.0e})',
            )


def test_h2_gong17_parity():
    """coolH2G17 vs H2Gong17Cooling, rtol = 1e-12. Alternative H2
    cooling form; the NCR default is H2Moseley21Cooling.
    """
    for xHI, xH2, xHII, xe in _CASES:
        state, species = _build_state(
            _T, _NH, xHI, xH2, xHII, xe)
        out = np.empty_like(_T)
        ch = H2Gong17Cooling(
            i_HI=species.idx['HI'],
            i_HII=species.idx['HII'],
            i_H2=species.idx['H2'],
            i_electron=species.idx['electron'],
        )
        ch.evaluate(state, out)
        expected = _coolH2G17_legacy(
            nH=_NH, T=_T,
            xHI=np.full_like(_T, xHI),
            xH2=np.full_like(_T, xH2),
            xHII=np.full_like(_T, xHII),
            xe=np.full_like(_T, xe),
        )
        np.testing.assert_allclose(
            out, expected, rtol=1.0e-12, atol=0.0,
            err_msg=f'(xHI={xHI}, xH2={xH2}, xHII={xHII})',
        )


def test_h2_moseley21_parity():
    """coolH2rovib vs H2Moseley21Cooling, rtol = 1e-12. NCR default
    H2 cooling form.
    """
    for xHI, xH2, xHII, xe in _CASES:
        state, species = _build_state(
            _T, _NH, xHI, xH2, xHII, xe)
        out = np.empty_like(_T)
        ch = H2Moseley21Cooling(
            i_HI=species.idx['HI'], i_H2=species.idx['H2'],
        )
        ch.evaluate(state, out)
        expected = _coolH2rovib_legacy(
            nH=_NH, T=_T,
            xHI=np.full_like(_T, xHI),
            xH2=np.full_like(_T, xH2),
        )
        np.testing.assert_allclose(
            out, expected, rtol=1.0e-12, atol=0.0,
            err_msg=f'(xHI={xHI}, xH2={xH2})',
        )


def test_lya_parity():
    """coolLya vs LyaCooling, rtol = 1e-12. NCR default H I cooling form."""
    for xHI, xH2, xHII, xe in _CASES:
        state, species = _build_state(
            _T, _NH, xHI, xH2, xHII, xe)
        out = np.empty_like(_T)
        ch = LyaCooling(
            i_HI=species.idx['HI'],
            i_electron=species.idx['electron'],
        )
        ch.evaluate(state, out)
        expected = _coolLya_legacy(
            nH=_NH, T=_T,
            xe=np.full_like(_T, xe),
            xHI=np.full_like(_T, xHI),
        )
        np.testing.assert_allclose(
            out, expected, rtol=1.0e-12, atol=0.0,
            err_msg=f'(xHI={xHI}, xe={xe})',
        )


def test_grain_recombination_parity():
    """coolRec vs GrainRecombinationCooling, rtol = 1e-12. Sweep
    Z_d and chi_PE because the WD01 fit is non-linear in both.
    """
    for xHI, xH2, xHII, xe in _CASES:
        for Z_d in (0.1, 1.0):
            for chi_PE in (0.3, 1.0, 10.0):
                state, species = _build_state(
                    _T, _NH, xHI, xH2, xHII, xe,
                    Z_d=Z_d, chi_PE=chi_PE)
                out = np.empty_like(_T)
                ch = GrainRecombinationCooling(
                    i_electron=species.idx['electron'])
                ch.evaluate(state, out)
                expected = _coolRec_legacy(
                    nH=_NH, T=_T,
                    xe=np.full_like(_T, xe),
                    Z_d=Z_d,
                    chi_PE=np.full_like(_T, chi_PE),
                )
                np.testing.assert_allclose(
                    out, expected, rtol=1.0e-12, atol=0.0,
                    err_msg=f'(Z_d={Z_d}, chi_PE={chi_PE}, xe={xe})',
                )
