"""Parity test: Verner+96 photoionization cross sections.

Phase 1 strangler-pattern port of `pyathena.microphysics.photx`.
Asserts the new `pyathena.chemistry.rates.photx.PhotX` matches the
frozen microphysics reference at every (Z, N, E) we care about, to
`rtol=1e-12, atol=0`.

The constructor reads `data/microphysics/verner96_photx.dat` (which
did NOT move); call-time output is a pure NumPy expression on the
loaded fit-parameter row, so byte-identical agreement is the right
bar. `get_sigma_pi_H2` is a module-level piecewise expression and is
also covered here.
"""
from __future__ import annotations

import numpy as np
import pytest

from pyathena.chemistry import _parity
from pyathena.chemistry.rates.photx import PhotX as PhotXNew
from pyathena.chemistry.rates.photx import get_sigma_pi_H2 as get_sigma_pi_H2_new
from pyathena.microphysics.photx import PhotX as PhotXOld
from pyathena.microphysics.photx import get_sigma_pi_H2 as get_sigma_pi_H2_old


# (Z, N, label) -- (Z, N) labels the ion BEFORE ionization. Mirrors
# `tests/microphysics/test_photx_sigma.py:PHOTOION_CATALOG`, covering
# the followed coolant set plus H, He, Ne.
PHOTOION_CATALOG = [
    (1, 1, "H I"),
    (2, 2, "He I"),
    (2, 1, "He II"),
    (6, 6, "C I"),
    (6, 5, "C II"),
    (7, 7, "N I"),
    (7, 6, "N II"),
    (8, 8, "O I"),
    (8, 7, "O II"),
    (8, 6, "O III"),
    (10, 10, "Ne I"),
    (16, 16, "S I"),
    (16, 15, "S II"),
    (16, 14, "S III"),
]


@pytest.fixture(scope="module")
def px_old():
    return PhotXOld()


@pytest.fixture(scope="module")
def px_new():
    return PhotXNew()


@pytest.fixture(scope="module")
def E_grid():
    """Sub-threshold through hard-X-ray energies (1 eV to 10 keV).
    Spans the `E < Eth` cutoff branch and the high-E asymptote."""
    return np.logspace(0.0, 4.0, 50)


@pytest.mark.parametrize("Z,N,label", PHOTOION_CATALOG)
def test_parity_get_sigma_E_grid(px_old, px_new, E_grid, Z, N, label):
    """Array-valued E input: every (Z, N) reactant in the followed
    coolant set agrees bit-for-bit on a 50-point log E grid spanning
    the sub-threshold cutoff branch through the high-E tail.
    """
    out_old, out_new = _parity.run_both(
        px_old.get_sigma, px_new.get_sigma, Z, N, E_grid)
    _parity.assert_close(out_old, out_new, rtol=1.0e-12, atol=0.0,
                         label=f'get_sigma[{label}, E_grid]')


@pytest.mark.parametrize("Z,N,label", PHOTOION_CATALOG)
def test_parity_get_Eth_eV(px_old, px_new, Z, N, label):
    """Threshold energy lookup in eV: a direct table read, so the
    match is exact (rtol=0, atol=0)."""
    out_old, out_new = _parity.run_both(
        px_old.get_Eth, px_new.get_Eth, Z, N)
    _parity.assert_close(out_old, out_new, rtol=0.0, atol=0.0,
                         label=f'get_Eth[{label}, eV]')


@pytest.mark.parametrize("Z,N,label", [
    (1, 1, "H I"),
    (2, 1, "He II"),
    (8, 8, "O I"),
])
def test_parity_get_Eth_Angstrom(px_old, px_new, Z, N, label):
    """Threshold wavelength via the astropy unit conversion path."""
    out_old, out_new = _parity.run_both(
        px_old.get_Eth, px_new.get_Eth, Z, N, unit='Angstrom')
    _parity.assert_close(out_old, out_new, rtol=1.0e-12, atol=0.0,
                         label=f'get_Eth[{label}, Angstrom]')


def test_parity_loaded_table_arrays(px_old, px_new):
    """The eleven vectors loaded from `verner96_photx.dat` must match
    element-wise: any path-resolution drift that pointed at a
    different file would surface here, not via the cross-section call.
    """
    for attr in ('Z', 'N', 'Eth', 'Emax', 'E0',
                 'sigma0', 'ya', 'P', 'yw', 'y0', 'y1'):
        _parity.assert_close(getattr(px_old, attr),
                             getattr(px_new, attr),
                             rtol=0.0, atol=0.0,
                             label=f'PhotX.{attr}')


def test_parity_get_sigma_pi_H2():
    """`get_sigma_pi_H2(E)` is a module-level piecewise function:
    exercise the full piecewise grid (below 15.2 eV cutoff, every
    constant interval, the analytic E^-3 tail above 18.1 eV)."""
    E = np.array([10.0, 15.0, 15.3, 15.5, 15.8, 16.0, 16.3, 16.5,
                  16.75, 16.9, 17.1, 17.4, 17.8, 18.0, 18.5, 25.0,
                  50.0, 100.0])
    out_old, out_new = _parity.run_both(
        get_sigma_pi_H2_old, get_sigma_pi_H2_new, E)
    _parity.assert_close(out_old, out_new, rtol=1.0e-12, atol=0.0,
                         label='get_sigma_pi_H2')
