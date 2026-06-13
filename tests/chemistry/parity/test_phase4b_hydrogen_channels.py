"""Grouped parity tests for the Phase 4b hydrogen channels.

Covers:

- `cooling.hi_collisional_ionization.HICollisionalIonizationCooling`
  vs `pyathena.microphysics.cool.coolHIion`
- `cooling.recombination_hydrogen.HRecombinationCooling`
  vs `pyathena.microphysics.cool.coolrecH`
- `cooling.free_free.FreeFreeHCooling`
  vs `pyathena.microphysics.cool.coolffH`
- `heating.cosmic_ray.CosmicRayHeating`
  vs `pyathena.microphysics.cool.heatCR`

Per the project's pytest grouping rule (memory file
`feedback_pytest_test_count_grouping.md`), each comparison loops
internal cases inside one test function rather than parametrising.
Tolerance: `rtol = 1e-12`, `atol = 0` everywhere.
"""
from __future__ import annotations

import numpy as np

from pyathena.chemistry.species import SpeciesSet
from pyathena.chemistry.state import ChemState
from pyathena.chemistry.cooling.hi_collisional_ionization import (
    HICollisionalIonizationCooling,
)
from pyathena.chemistry.cooling.recombination_hydrogen import (
    HRecombinationCooling,
)
from pyathena.chemistry.cooling.free_free import FreeFreeHCooling
from pyathena.chemistry.heating.cosmic_ray import CosmicRayHeating

from pyathena.microphysics.cool import (
    coolHIion as _coolHIion_legacy,
    coolrecH as _coolrecH_legacy,
    coolffH as _coolffH_legacy,
    heatCR as _heatCR_legacy,
)


_T_VALS = np.logspace(2.0, 6.0, 25)
_N_VALS = np.logspace(-2.0, 4.0, 16)
_T_GRID, _NH_GRID = np.meshgrid(_T_VALS, _N_VALS, indexing='xy')
_T = _T_GRID.ravel()
_NH = _NH_GRID.ravel()


def _build_state(T, nH, xHI, xHII, xH2, xe):
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
    state.x[idx['HII']] = xHII
    state.x[idx['H2']] = xH2
    state.x[idx['electron']] = xe

    for name in (
        'cooling:hi_coll_ion:tmp', 'cooling:hi_coll_ion:y',
        'cooling:hi_coll_ion:kcoll',
        'cooling:h_rec:tmp', 'cooling:h_rec:E_rr_B',
        'cooling:free_free:tmp', 'cooling:free_free:gff',
        'cooling:free_free:L', 'cooling:free_free:denom',
        'heating:cosmic_ray:tmp', 'heating:cosmic_ray:qHI',
        'heating:cosmic_ray:qH2', 'heating:cosmic_ray:log_nH',
        'heating:cosmic_ray:mask_b1',
    ):
        state.alloc_scratch(name, (ncell,))
    return state, species


# Four representative ionisation states. Same case set drives every
# channel to keep the parametrisation in one place.
_CASES = (
    # (xHI, xHII, xH2, xe)
    (0.99, 0.01, 0.0,  0.01),    # WNM-like
    (0.50, 0.50, 0.0,  0.50),    # HII region boundary
    (0.10, 0.90, 0.0,  0.90),    # ionised
    (0.10, 0.05, 0.40, 0.05),    # half-molecular CNM
)


def test_hi_collisional_ionization_parity():
    """coolHIion vs HICollisionalIonizationCooling, rtol = 1e-12."""
    for xHI, xHII, xH2, xe in _CASES:
        state, species = _build_state(_T, _NH, xHI, xHII, xH2, xe)
        out = np.empty_like(_T)
        ch = HICollisionalIonizationCooling(
            i_HI=species.idx['HI'],
            i_electron=species.idx['electron'],
        )
        ch.evaluate(state, out)
        expected = _coolHIion_legacy(
            nH=_NH, T=_T,
            xe=np.full_like(_T, xe),
            xHI=np.full_like(_T, xHI),
        )
        np.testing.assert_allclose(
            out, expected, rtol=1.0e-12, atol=0.0,
            err_msg=f'(xHI={xHI}, xe={xe})',
        )


def test_h_recombination_parity():
    """coolrecH vs HRecombinationCooling, rtol = 1e-12."""
    for xHI, xHII, xH2, xe in _CASES:
        state, species = _build_state(_T, _NH, xHI, xHII, xH2, xe)
        out = np.empty_like(_T)
        ch = HRecombinationCooling(
            i_HII=species.idx['HII'],
            i_electron=species.idx['electron'],
        )
        ch.evaluate(state, out)
        expected = _coolrecH_legacy(
            nH=_NH, T=_T,
            xe=np.full_like(_T, xe),
            xHII=np.full_like(_T, xHII),
        )
        np.testing.assert_allclose(
            out, expected, rtol=1.0e-12, atol=0.0,
            err_msg=f'(xHII={xHII}, xe={xe})',
        )


def test_free_free_h_parity():
    """coolffH vs FreeFreeHCooling, rtol = 1e-12."""
    for xHI, xHII, xH2, xe in _CASES:
        state, species = _build_state(_T, _NH, xHI, xHII, xH2, xe)
        out = np.empty_like(_T)
        ch = FreeFreeHCooling(
            i_HII=species.idx['HII'],
            i_electron=species.idx['electron'],
        )
        ch.evaluate(state, out)
        expected = _coolffH_legacy(
            nH=_NH, T=_T,
            xe=np.full_like(_T, xe),
            xHII=np.full_like(_T, xHII),
        )
        np.testing.assert_allclose(
            out, expected, rtol=1.0e-12, atol=0.0,
            err_msg=f'(xHII={xHII}, xe={xe})',
        )


def test_cosmic_ray_heating_parity():
    """heatCR vs CosmicRayHeating, rtol = 1e-12. Sweeps xi_CR too
    because the multiplicative prefactor exposes any per-channel
    constant drift."""
    xi_cr_values = (1.0e-17, 2.0e-16, 1.0e-15)
    for xHI, xHII, xH2, xe in _CASES:
        for xi_CR in xi_cr_values:
            state, species = _build_state(_T, _NH, xHI, xHII, xH2, xe)
            out = np.empty_like(_T)
            ch = CosmicRayHeating(
                i_HI=species.idx['HI'],
                i_H2=species.idx['H2'],
                i_electron=species.idx['electron'],
                xi_CR=xi_CR,
            )
            ch.evaluate(state, out)
            expected = _heatCR_legacy(
                nH=_NH, xe=np.full_like(_T, xe),
                xHI=np.full_like(_T, xHI),
                xH2=np.full_like(_T, xH2),
                xi_CR=xi_CR,
            )
            np.testing.assert_allclose(
                out, expected, rtol=1.0e-12, atol=0.0,
                err_msg=f'(xHI={xHI}, xH2={xH2}, xi_CR={xi_CR:.0e})',
            )
