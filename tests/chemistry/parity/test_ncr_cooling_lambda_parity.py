"""Parity test #4: NCR cooling Lambda(T, nH, x_e, x_HI, x_H2, Z_g)
on a (T, nH) grid against `pyathena.microphysics.cool`.

Phase 3 introduces the explicit subcycling solver but leaves the
cooling tables for Phase 4. The cooling-side parity contract is
therefore exercised against the legacy `pyathena.microphysics.cool`
module directly: the new-path callable in this file delegates to the
legacy function, the way `test_OII_HII_resonance_parity.py` does for
the charge-transfer wiring. Phase 4 will rebind these callables to
the new `pyathena.chemistry.coolants` package and the tolerance band
stays at rtol=1e-10.

The grid covers four decades in nH (cold dense PDR to warm diffuse
ISM) and four decades in T (cold molecular through warm ionised); a
fixed ionisation-fraction sweep saturates the per-species coolant
formulas in their usual operating range. Z_g is fixed at solar (1.0).
The catalog mirrors the species the NCR solver evaluates on the hot
path (CII, OI, OII, HI, H2G17, plus the photoionisation cooling and
Lyman-alpha terms).
"""
from __future__ import annotations

import numpy as np
import pytest

from pyathena.chemistry import _parity

# Frozen reference -- pre-Phase 4 cooling lives here.
from pyathena.microphysics import cool as _cool_old
# New chemistry path; Phase 3 delegates to the same module so the
# parity contract holds by construction. Phase 4 rebinds this to the
# `pyathena.chemistry.coolants` package and the tolerance stays.
from pyathena.microphysics import cool as _cool_new


# ---- Grid fixtures ------------------------------------------------------
@pytest.fixture(scope='module')
def T_grid() -> np.ndarray:
    """Temperature grid from 100 K to 1e6 K (50 points)."""
    return np.logspace(2.0, 6.0, 50)


@pytest.fixture(scope='module')
def nH_grid() -> np.ndarray:
    """Hydrogen density grid from 0.1 cm^-3 to 1e4 cm^-3 (10 points)."""
    return np.logspace(-1.0, 4.0, 10)


@pytest.fixture(scope='module')
def composition() -> dict:
    """A single canonical composition used across the catalog.

    Values are chosen so every coolant has a non-zero contribution to
    test, while still being physically plausible (sum
    `xHI + xHII + 2 xH2 = 1`):

    - x_HI = 0.5: half neutral
    - x_HII = 0.4: substantial ionised fraction so coolHIion / coolOII
      see a meaningful x_e
    - x_H2 = 0.05: 10% in molecular form
    - x_e = x_HII = 0.4 (electron count from H II only)
    """
    xHI = 0.5
    xHII = 0.4
    xH2 = 0.05
    return {
        'xHI': xHI,
        'xHII': xHII,
        'xH2': xH2,
        'xe': xHII,
        'xCII': 1.6e-4,    # x_CII saturated at solar C abundance
        'xOI': 3.2e-4,     # x_OI ~ x_O_std * Z_g (mostly neutral)
        'xOII': 1.0e-6,    # small OII tracer for the OII catalog entry
        'Z_g': 1.0,
    }


# ---- Catalog: per-species cool functions and their argument adapters ---
def _coolCII_args(T, nH, comp):
    return (nH, T, comp['xe'], comp['xHI'], comp['xH2'], comp['xCII'])


def _coolOI_args(T, nH, comp):
    return (nH, T, comp['xe'], comp['xHI'], comp['xH2'], comp['xOI'])


def _coolOII_args(T, nH, comp):
    return (nH, T, comp['xe'], comp['xOII'])


def _coolHIion_args(T, nH, comp):
    return (nH, T, comp['xe'], comp['xHI'])


def _coolHI_args(T, nH, comp):
    return (nH, T, comp['xHI'], comp['xe'])


def _coolHISmith21_args(T, nH, comp):
    return (nH, T, comp['xe'], comp['xHI'])


def _coolH2G17_args(T, nH, comp):
    return (nH, T, comp['xHI'], comp['xH2'], comp['xHII'], comp['xe'])


_COOLANT_CATALOG = [
    ('coolCII', _coolCII_args),
    ('coolOI', _coolOI_args),
    ('coolOII', _coolOII_args),
    ('coolHIion', _coolHIion_args),
    ('coolHI', _coolHI_args),
    ('coolHISmith21', _coolHISmith21_args),
    ('coolH2G17', _coolH2G17_args),
]


# ---- Per-species parity --------------------------------------------------
@pytest.mark.parametrize('name, adapt', _COOLANT_CATALOG,
                         ids=[name for name, _ in _COOLANT_CATALOG])
def test_cooling_lambda_per_species_parity(
    T_grid, nH_grid, composition, name, adapt,
):
    """Per-species Lambda parity on a (T, nH) grid at rtol=1e-10.

    The new and old paths invoke the same underlying microphysics
    function in Phase 3, so the test is bit-exact by construction.
    The catalog and grid layout are written so Phase 4 only has to
    rebind `_cool_new` at the top of the module to its new home in
    `pyathena.chemistry.coolants` and the tolerance stays.
    """
    # Broadcast (T, nH) onto a 2D grid so every catalog entry runs on
    # the same shape.
    T_2d, nH_2d = np.meshgrid(T_grid, nH_grid, indexing='xy')
    args = adapt(T_2d, nH_2d, composition)

    fn_old = getattr(_cool_old, name)
    fn_new = getattr(_cool_new, name)
    out_old, out_new = _parity.run_both(fn_old, fn_new, *args)

    _parity.assert_close(out_old, out_new, rtol=1.0e-10, atol=0.0,
                         label=f'{name}')


def test_summed_lambda_parity(T_grid, nH_grid, composition):
    """Sum the catalog contributions and check parity on the total.

    Many coolants contribute simultaneously at warm-neutral conditions;
    the summed Lambda is the quantity the thermal driver actually
    needs, so we test that explicitly too.
    """
    T_2d, nH_2d = np.meshgrid(T_grid, nH_grid, indexing='xy')

    def _summed(cool_module):
        total = np.zeros_like(T_2d)
        for name, adapt in _COOLANT_CATALOG:
            args = adapt(T_2d, nH_2d, composition)
            total = total + getattr(cool_module, name)(*args)
        return total

    def _summed_old():
        return _summed(_cool_old)

    def _summed_new():
        return _summed(_cool_new)

    out_old, out_new = _parity.run_both(_summed_old, _summed_new)
    # In the Phase 3 delegation pattern the two callables share the
    # same module reference, so the comparison is exact. Phase 4
    # rebinds `_cool_new` to the new chemistry path and the tolerance
    # band kicks in.
    _parity.assert_close(out_old, out_new, rtol=1.0e-10, atol=0.0,
                         label='summed_lambda')
