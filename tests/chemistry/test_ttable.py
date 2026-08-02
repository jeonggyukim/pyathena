"""Unit tests for the shared temperature-table helper.

`TemperatureTable` holds one implementation of the build-and-look-up
code that `ci_rate`, `rec_rate` and `ct_rate` each grew separately, so
cooling channels can tabulate their temperature-only factors without a
fourth copy.
"""
from __future__ import annotations

import numpy as np
import pytest

from pyathena.chemistry.enums import InterpMode
from pyathena.chemistry.rates._ttable import TAB_N, TemperatureTable
from pyathena.chemistry.state import assert_no_alloc


_MODES = (InterpMode.LogLog, InterpMode.Nqt2, InterpMode.Nqt1)
_MODE_IDS = ('LogLog', 'Nqt2', 'Nqt1')
_T = np.logspace(1.0, 8.0, 3000)


def _free_free(T):
    """The temperature-only factor of `FreeFreeHCooling`.

    Smooth and positive over the whole grid, so it isolates the
    interpolation error from anything the fit itself does.
    """
    gff = 1.0 + 0.44 / (1.0 + 0.058 * np.log(T / 10.0 ** 5.4) ** 2)
    return 1.422e-25 * gff * np.sqrt(T / 1.0e4)


@pytest.mark.parametrize(
    'mode,rtol', list(zip(_MODES, (1.0e-5, 5.0e-3, 5.0e-3))), ids=_MODE_IDS)
def test_smooth_function_within_tolerance(mode, rtol):
    """LogLog is limited by interpolation, the NQT modes by encoding."""
    table = TemperatureTable(_free_free, mode)
    err = np.abs(table(_T) / _free_free(_T) - 1.0).max()
    assert err <= rtol, f'{mode.name}: max rtol = {err:.3e}'


def test_exact_mode_is_rejected():
    """`Exact` is not a table; the caller handles it analytically."""
    with pytest.raises(ValueError, match='InterpMode.Exact'):
        TemperatureTable(_free_free, InterpMode.Exact)


@pytest.mark.parametrize('mode', _MODES, ids=_MODE_IDS)
def test_identically_zero_column_returns_exact_zeros(mode):
    """A vanishing column must not come back at the log-encoding floor."""
    table = TemperatureTable(lambda T: np.zeros_like(T), mode)
    assert np.all(table(_T) == 0.0)


@pytest.mark.parametrize('mode', _MODES, ids=_MODE_IDS)
def test_clamp_holds_the_value_flat_outside_the_window(mode):
    """Outside `clamp` the result is the value at the window edge.

    The caller passes an unclamped `fn` plus the window, and lookup
    clips `T` into it. That keeps the slope discontinuity out of the
    interpolant while reproducing the analytic clamp exactly.
    """
    table = TemperatureTable(lambda T: T ** 2.0, mode,
                             clamp=(1.0e3, 1.0e5))
    below = table(np.array([1.0, 10.0, 500.0]))
    above = table(np.array([2.0e5, 1.0e7, 1.0e9]))
    np.testing.assert_allclose(below, 1.0e6, rtol=5.0e-3)
    np.testing.assert_allclose(above, 1.0e10, rtol=5.0e-3)
    # Flat, not merely close: every entry equals its neighbours.
    np.testing.assert_allclose(below, below[0], rtol=1.0e-12)
    np.testing.assert_allclose(above, above[0], rtol=1.0e-12)


def test_clamp_beats_tabulating_the_clamped_form():
    """The reason the clamp is applied at lookup rather than baked in.

    Tabulating an already-clamped function puts a slope discontinuity
    inside a grid bin, and interpolating across it is what cost the
    charge-transfer tables 8.9e-2 before this pattern was adopted.
    """
    lo, hi = 5.0e3, 1.0e5

    def unclamped(T):
        return T ** 2.0

    def clamped(T):
        return np.clip(T, lo, hi) ** 2.0

    aware = TemperatureTable(unclamped, InterpMode.LogLog, clamp=(lo, hi))
    naive = TemperatureTable(clamped, InterpMode.LogLog)

    # Sample tightly around the lower edge, where the kink sits.
    T = np.logspace(np.log10(lo) - 0.02, np.log10(lo) + 0.02, 2000)
    ref = clamped(T)
    err_aware = np.abs(aware(T) / ref - 1.0).max()
    err_naive = np.abs(naive(T) / ref - 1.0).max()
    assert err_aware < err_naive
    assert err_aware <= 1.0e-5


@pytest.mark.parametrize('mode', _MODES, ids=_MODE_IDS)
def test_lookup_is_allocation_free(mode):
    """`lookup` is what the substep loop calls, so it must not allocate."""
    table = TemperatureTable(_free_free, mode)
    n = 64
    T = np.logspace(1.0, 8.0, n)
    out = np.empty(n)
    scratch = (np.empty(n, dtype=np.int64), np.empty(n),
               np.empty(n), np.empty(n))
    table.lookup(T, out, scratch)          # warm any lazy import
    with assert_no_alloc():
        table.lookup(T, out, scratch)


def test_lookup_matches_the_convenience_form():
    """The allocating and allocation-free paths agree bit for bit."""
    table = TemperatureTable(_free_free, InterpMode.LogLog)
    n = _T.size
    out = np.empty(n)
    scratch = (np.empty(n, dtype=np.int64), np.empty(n),
               np.empty(n), np.empty(n))
    table.lookup(_T, out, scratch)
    np.testing.assert_array_equal(out, table(_T))


def test_out_argument_is_written_in_place():
    table = TemperatureTable(_free_free, InterpMode.LogLog)
    out = np.empty(_T.size)
    returned = table(_T, out=out)
    assert returned is out


def test_grid_endpoints_are_reproduced():
    """Interpolation is exact at the first and last grid node."""
    table = TemperatureTable(_free_free, InterpMode.LogLog)
    edges = table.T_grid[[0, TAB_N - 1]]
    np.testing.assert_allclose(table(edges), _free_free(edges),
                               rtol=1.0e-12)


def test_locate_is_shared_across_tables():
    """One `locate` drives many `interp` calls.

    This is the property that makes tabulation pay: finding the axis
    position costs a transcendental, interpolating at a known position
    does not. A cell reading N quantities should pay the position once.
    """
    t1 = TemperatureTable(_free_free, InterpMode.LogLog)
    t2 = TemperatureTable(lambda T: np.exp(-11.84 / (T * 1.0e-4)),
                          InterpMode.LogLog)
    n = _T.size
    idx = np.empty(n, dtype=np.int64)
    frac = np.empty(n)
    lo, hi = np.empty(n), np.empty(n)
    o1, o2 = np.empty(n), np.empty(n)

    t1.locate(_T, idx, frac)
    t1.interp(idx, frac, o1, lo, hi)
    t2.interp(idx, frac, o2, lo, hi)

    # Identical to each table resolving the position on its own.
    np.testing.assert_array_equal(o1, t1(_T))
    np.testing.assert_array_equal(o2, t2(_T))


def test_locate_and_interp_are_allocation_free():
    t1 = TemperatureTable(_free_free, InterpMode.LogLog)
    t2 = TemperatureTable(lambda T: np.sqrt(T), InterpMode.LogLog)
    n = 64
    T = np.logspace(1.0, 8.0, n)
    idx = np.empty(n, dtype=np.int64)
    frac = np.empty(n)
    lo, hi = np.empty(n), np.empty(n)
    o1, o2 = np.empty(n), np.empty(n)

    t1.locate(T, idx, frac)                 # warm
    t1.interp(idx, frac, o1, lo, hi)
    with assert_no_alloc():
        t1.locate(T, idx, frac)
        t1.interp(idx, frac, o1, lo, hi)
        t2.interp(idx, frac, o2, lo, hi)
