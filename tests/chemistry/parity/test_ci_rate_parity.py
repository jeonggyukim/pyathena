"""Parity test: Voronov 1997 collisional ionization rate fits.

Phase 1 strangler-pattern port of `pyathena.microphysics.ci_rate`.
Asserts the new `pyathena.chemistry.rates.ci_rate.CollIonRate`
matches the frozen microphysics reference at every (Z, N, T) we
care about, to `rtol=1e-12, atol=0`.

The constructor reads `data/microphysics/cloudy/coll_ion.dat` (which
did NOT move); call-time output is a pure NumPy expression on the
loaded `(A, P, X, K, dE_Kel)` row, so byte-identical agreement is
the right bar.
"""
from __future__ import annotations

import numpy as np
import pytest

from pyathena.chemistry import _parity
from pyathena.chemistry.rates.ci_rate import CollIonRate as CollIonRateNew
from pyathena.microphysics.ci_rate import CollIonRate as CollIonRateOld


# (Z, N_reactant, label) -- (Z, N) labels the ion BEFORE ionization.
# Mirrors `tests/microphysics/test_ci_rate.py:CI_CATALOG`, covering
# the followed coolant set plus H, He.
CI_CATALOG = [
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
def ci_old():
    return CollIonRateOld()


@pytest.fixture(scope="module")
def ci_new():
    return CollIonRateNew()


@pytest.fixture(scope="module")
def T_grid():
    """Warm-to-hot grid covering the U > 80 cutoff branch (cold end)
    through the unsuppressed regime (T >> dE/k_B)."""
    return np.logspace(2.0, 8.0, 25)


@pytest.mark.parametrize("Z,N,label", CI_CATALOG)
def test_parity_get_ci_rate_T_grid(ci_old, ci_new, T_grid, Z, N, label):
    """Array-valued T input: every (Z, N) reactant in the followed
    coolant set agrees bit-for-bit (modulo numpy float ordering) on
    a 25-point log T grid spanning the cold cutoff branch through
    the high-T plateau.
    """
    out_old, out_new = _parity.run_both(
        ci_old.get_ci_rate, ci_new.get_ci_rate, Z, N, T_grid)
    _parity.assert_close(out_old, out_new, rtol=1.0e-12, atol=0.0,
                         label=f'get_ci_rate[{label}, T_grid]')


@pytest.mark.parametrize("Z,N,label", [
    (1, 1, "H I"),
    (8, 8, "O I"),
    (16, 15, "S II"),
])
def test_parity_get_ci_rate_scalar_T(ci_old, ci_new, Z, N, label):
    """Scalar T input: exercises the `np.where` broadcast path
    where T is a 0-d input. H I, O I, S II picked to hit three
    different dE_Kel rows.
    """
    for T in (1.0e3, 1.0e4, 2.0e5, 1.0e6):
        out_old, out_new = _parity.run_both(
            ci_old.get_ci_rate, ci_new.get_ci_rate, Z, N, T)
        _parity.assert_close(out_old, out_new, rtol=1.0e-12, atol=0.0,
                             label=f'get_ci_rate[{label}, T={T:g}]')


def test_parity_loaded_table_arrays(ci_old, ci_new):
    """The five vectors loaded from `coll_ion.dat` must match
    element-wise: any path-resolution drift that pointed at a
    different file would surface here, not via the rate call.
    """
    for attr in ('N', 'Z', 'dE_Kel', 'P', 'A', 'X', 'K'):
        _parity.assert_close(getattr(ci_old, attr),
                             getattr(ci_new, attr),
                             rtol=0.0, atol=0.0,
                             label=f'CollIonRate.{attr}')
