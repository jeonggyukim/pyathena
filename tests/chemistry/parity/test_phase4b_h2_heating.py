"""Grouped parity tests for the Phase 4b batch 4a H2 heating channels.

Covers:

- `heating.h2_formation.H2FormationHeating`
- `heating.h2_photodissociation.H2DissociationHeating`
- `heating.h2_photodissociation.H2PumpHeating`

The legacy `pyathena.microphysics.cool.heatH2` returns the
form/diss/pump triple together, gated by an `iH2heating` flag (1 =
V18 / Sternberg+2014, 2 = HM79). The channel ports follow the NCR
production path (`tigris-ncr/src/photchem/ncr_rates.hpp:1545-1558`
and `Athena-TIGRESS/src/microphysics/cool_tigress.c:1132-1152`)
which uses HM79 by default (`iH2heating = 2`). Tolerance
`rtol = 1e-12`, `atol = 0`.
"""
from __future__ import annotations

import numpy as np

from pyathena.chemistry.species import SpeciesSet
from pyathena.chemistry.state import ChemState
from pyathena.chemistry.heating.h2_formation import H2FormationHeating
from pyathena.chemistry.heating.h2_photodissociation import (
    H2DissociationHeating, H2PumpHeating,
)

# Reference implementations inline. We do NOT call the legacy
# `pyathena.microphysics.cool.heatH2` because it has a latent typo
# (`sqrt(T2)` instead of `np.sqrt(T2)` at cool.py line 151) on the
# `ikgr_H2 == 1` path. The reference here mirrors the C++ NCR
# production source at `tigris-ncr/src/photchem/ncr_rates.hpp:
# 905-916, 950-970, 1370-1383, 1545-1557` byte-for-byte.

_EV_CGS = 1.602176634e-12


def _kgr_HM79(T, Z_d, kgr_H2):
    T2 = T * 1.0e-2
    sqrtT2 = np.sqrt(T2)
    denom = 1.0 + 0.4 * sqrtT2 + 0.2 * T2 + 0.08 * T2 * T2
    return kgr_H2 * Z_d * sqrtT2 * 2.0 / denom


def _ncrit_HM79(T, xHI, xH2):
    de = (1.6 * xHI * np.exp(-(400.0 / T) ** 2)
          + 1.4 * xH2 * np.exp(-12000.0 / (T + 1200.0)))
    return 1.0e6 / np.sqrt(T) / de


def _heatH2_form_ref(nH, T, xHI, xH2, Z_d, kgr_H2):
    kgr = _kgr_HM79(T, Z_d, kgr_H2)
    ncrit = _ncrit_HM79(T, xHI, xH2)
    return (kgr * nH * xHI * (0.2 + 4.2 / (1.0 + ncrit / nH))
            * _EV_CGS)


def _heatH2_diss_ref(xH2, xi_diss_H2):
    return xi_diss_H2 * xH2 * 0.4 * _EV_CGS


def _heatH2_pump_HM79_ref(nH, T, xHI, xH2, xi_diss_H2):
    ncrit = _ncrit_HM79(T, xHI, xH2)
    f = 1.0 / (1.0 + ncrit / nH)
    return xi_diss_H2 * xH2 * 9.0 * 2.2 * f * _EV_CGS


_T_VALS = np.logspace(1.0, 4.0, 28)
_N_VALS = np.logspace(-2.0, 4.0, 18)
_T_GRID, _NH_GRID = np.meshgrid(_T_VALS, _N_VALS, indexing='xy')
_T = _T_GRID.ravel()
_NH = _NH_GRID.ravel()

_SCRATCH = (
    # H2 formation
    'heating:h2_form:T2', 'heating:h2_form:sqrtT2',
    'heating:h2_form:kgr', 'heating:h2_form:ncrit',
    'heating:h2_form:tmp_a', 'heating:h2_form:tmp_b',
    'heating:h2_form:f',
    # H2 pump
    'heating:h2_pump:tmp_a', 'heating:h2_pump:tmp_b',
    'heating:h2_pump:ncrit', 'heating:h2_pump:f',
)


def _build_state(T, nH, xHI, xH2, xe=0.0, Z_d=1.0):
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
    for name in _SCRATCH:
        state.alloc_scratch(name, (ncell,))
    return state, species


_CASES = (
    # (xHI, xH2)
    (0.99, 0.0),
    (0.50, 0.20),
    (0.10, 0.45),
    (0.01, 0.49),
)


def test_h2_formation_heating_parity():
    """H2FormationHeating vs HM79 closed-form reference (NCR default)."""
    kgr_H2 = 3.0e-17
    for xHI, xH2 in _CASES:
        for Z_d in (0.5, 1.0):
            state, species = _build_state(_T, _NH, xHI, xH2, Z_d=Z_d)
            out = np.empty_like(_T)
            ch = H2FormationHeating(
                i_HI=species.idx['HI'],
                i_H2=species.idx['H2'],
                kgr_H2=kgr_H2,
                temperature_dependent_kgr=True,
            )
            with np.errstate(divide='ignore', invalid='ignore'):
                ch.evaluate(state, out)
                expected = _heatH2_form_ref(
                    nH=_NH, T=_T,
                    xHI=np.full_like(_T, xHI),
                    xH2=np.full_like(_T, xH2),
                    Z_d=Z_d, kgr_H2=kgr_H2,
                )
            np.testing.assert_allclose(
                out, expected, rtol=1.0e-12, atol=0.0,
                err_msg=f'(xHI={xHI}, xH2={xH2}, Z_d={Z_d})',
            )


def test_h2_dissociation_heating_parity():
    """H2DissociationHeating vs constant-multiply reference."""
    for xHI, xH2 in _CASES:
        for xi_diss_H2 in (1.0e-13, 1.0e-11):
            state, species = _build_state(_T, _NH, xHI, xH2)
            out = np.empty_like(_T)
            ch = H2DissociationHeating(
                i_H2=species.idx['H2'],
                xi_diss_H2=xi_diss_H2,
            )
            ch.evaluate(state, out)
            expected = _heatH2_diss_ref(np.full_like(_T, xH2),
                                         xi_diss_H2)
            np.testing.assert_allclose(
                out, expected, rtol=1.0e-12, atol=0.0,
                err_msg=f'(xH2={xH2}, xi_diss={xi_diss_H2:.0e})',
            )


def test_h2_pump_heating_parity():
    """H2PumpHeating (HM79 form) vs closed-form reference (NCR default)."""
    for xHI, xH2 in _CASES:
        for xi_diss_H2 in (1.0e-13, 1.0e-11):
            state, species = _build_state(_T, _NH, xHI, xH2)
            out = np.empty_like(_T)
            ch = H2PumpHeating(
                i_HI=species.idx['HI'],
                i_H2=species.idx['H2'],
                xi_diss_H2=xi_diss_H2,
                form='HM79',
            )
            with np.errstate(divide='ignore', invalid='ignore'):
                ch.evaluate(state, out)
                expected = _heatH2_pump_HM79_ref(
                    nH=_NH, T=_T,
                    xHI=np.full_like(_T, xHI),
                    xH2=np.full_like(_T, xH2),
                    xi_diss_H2=xi_diss_H2,
                )
            np.testing.assert_allclose(
                out, expected, rtol=1.0e-12, atol=0.0,
                err_msg=f'(xHI={xHI}, xH2={xH2}, '
                f'xi_diss={xi_diss_H2:.0e})',
            )
