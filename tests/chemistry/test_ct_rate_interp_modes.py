"""Charge-transfer rate table modes against the analytic path.

`ChargeTransferRate` gained `LogLog` / `Nqt2` / `Nqt1` alongside
`Exact`. This pins how far each table mode may drift from the analytic
Kingdon & Ferland fits.

Tolerances, and why they differ from `ci_rate`'s:

- Recombination, LogLog: rtol 1e-3, the same bar `ci_rate` holds.
  Reached by tabulating the *unclamped* fit and applying the
  per-ion `[Tmin2, Tmax2]` clamp during lookup. Tabulating the
  clamped form directly puts a slope discontinuity at each window
  edge, and interpolating across the bin that straddles one costs
  8.9e-2 -- see the 2026-07-30 entry in `chemistry/CHANGES.md`.
- Ionization, LogLog: rtol 1e-2. Ionization keeps the
  tabulate-the-clamped-form path, because unclamping it would also
  require applying `exp(-dE * 1e4 / T)` at lookup, and putting a
  transcendental back into the lookup works against the point of
  tabulating. Only 8 of 120 rows carry a nonzero `dE_rate1`.
- NQT modes: rtol 2e-2, dominated by the NQT encoding error rather
  than by interpolation (1.5 percent worst case for `nqt1`).

Errors are measured only where the rate exceeds 1e-4 of that ion's
peak. Below that the comparison is meaningless for at least one ion:
the C II recombination fit has a root at 5397 K, just under its
`Tmin2` of 5500 K, so at the clamped floor it evaluates to 0.2 percent
of its leading term and relative error on that near-cancellation says
nothing about the table.
"""
from __future__ import annotations

import numpy as np
import pytest

from pyathena.chemistry.enums import InterpMode
from pyathena.chemistry.rates.ct_rate import ChargeTransferRate


_T = np.logspace(1.0, 8.0, 4000)
_SIGNIFICANT = 1.0e-4


@pytest.fixture(scope='module')
def exact():
    return ChargeTransferRate(InterpMode.Exact)


@pytest.fixture(scope='module')
def tabulated():
    return {m: ChargeTransferRate(m)
            for m in (InterpMode.LogLog, InterpMode.Nqt2, InterpMode.Nqt1)}


def _worst_rel_err(ex, tb, cols, getter):
    """Largest relative error over every tabulated ion."""
    worst = 0.0
    where = None
    for (Z, N) in sorted(cols):
        a = np.asarray(getattr(ex, getter)(Z, N, _T), dtype=np.float64)
        if a.max() <= 0.0:
            continue
        b = np.asarray(getattr(tb, getter)(Z, N, _T), dtype=np.float64)
        mask = a > max(_SIGNIFICANT * a.max(), 1.0e-30)
        if not mask.any():
            continue
        err = np.abs(b[mask] / a[mask] - 1.0).max()
        if err > worst:
            worst, where = err, (Z, N)
    return worst, where


@pytest.mark.parametrize(
    'mode,rtol',
    [(InterpMode.LogLog, 1.0e-3),
     (InterpMode.Nqt2, 2.0e-2),
     (InterpMode.Nqt1, 2.0e-2)],
    ids=['LogLog', 'Nqt2', 'Nqt1'])
def test_ct_rec_rate_table_matches_exact(exact, tabulated, mode, rtol):
    """Recombination: the clamp-aware path holds the ci_rate bar."""
    tb = tabulated[mode]
    worst, where = _worst_rel_err(
        exact, tb, tb._tab_col_rec, 'get_ct_rec_rate')
    assert worst <= rtol, (
        f'{mode.name} CT recombination: max rtol = {worst:.3e} '
        f'at (Z, N) = {where}')


@pytest.mark.parametrize(
    'mode,rtol',
    [(InterpMode.LogLog, 1.0e-2),
     (InterpMode.Nqt2, 2.0e-2),
     (InterpMode.Nqt1, 2.0e-2)],
    ids=['LogLog', 'Nqt2', 'Nqt1'])
def test_ct_ion_rate_table_matches_exact(exact, tabulated, mode, rtol):
    """Ionization: looser, and deliberately so -- see module docstring."""
    tb = tabulated[mode]
    worst, where = _worst_rel_err(
        exact, tb, tb._tab_col_ion, 'get_ct_ion_rate')
    assert worst <= rtol, (
        f'{mode.name} CT ionization: max rtol = {worst:.3e} '
        f'at (Z, N) = {where}')


def test_exact_mode_is_the_default():
    """Default construction keeps the analytic path, so the rtol 1e-12
    parity test against microphysics is unaffected by the table work.
    """
    assert ChargeTransferRate().interp_mode == InterpMode.Exact


@pytest.mark.parametrize('mode', [InterpMode.LogLog, InterpMode.Nqt2,
                                  InterpMode.Nqt1],
                         ids=['LogLog', 'Nqt2', 'Nqt1'])
def test_temperature_independent_branches_bypass_the_table(tabulated, mode):
    """The charge cutoffs are analytic in every mode.

    Ionization is zero for q > 3; recombination is zero for q < 1 and
    takes the Dalgarno estimate 1.92e-9 * q for q > 4. None of these
    depend on temperature, so none of them occupy a table column.
    """
    tb = tabulated[mode]
    T = np.array([1.0e2, 1.0e4, 1.0e6])

    # q = 4 ionization (Z=8, N=4) is above the tabulated charge range.
    assert np.all(tb.get_ct_ion_rate(8, 4, T) == 0.0)
    # q = 0 recombination: a neutral reactant cannot recombine.
    assert np.all(tb.get_ct_rec_rate(8, 8, T) == 0.0)
    # q = 5 recombination falls back to Dalgarno.
    np.testing.assert_allclose(tb.get_ct_rec_rate(8, 3, T),
                               1.92e-9 * 5, rtol=0.0)


@pytest.mark.parametrize('mode', [InterpMode.LogLog, InterpMode.Nqt2,
                                  InterpMode.Nqt1],
                         ids=['LogLog', 'Nqt2', 'Nqt1'])
def test_clamped_rates_are_flat_outside_the_window(tabulated, mode):
    """Below Tmin2 and above Tmax2 the recombination rate is constant.

    The lookup clips T into the per-column window, so the table has to
    reproduce the analytic path's plateaus exactly rather than
    extrapolating the fit.
    """
    tb = tabulated[mode]
    Z, N = 6, 5                       # C II, window [5500, 1e5] K
    j = tb._tab_col_rec[(Z, N)]
    t_lo, t_hi = tb._tab_clamp_rec[j]
    below = tb.get_ct_rec_rate(Z, N, np.array([10.0, 100.0, t_lo * 0.5]))
    above = tb.get_ct_rec_rate(Z, N, np.array([t_hi * 2.0, t_hi * 10.0]))
    np.testing.assert_allclose(below, below[0], rtol=1.0e-12)
    np.testing.assert_allclose(above, above[0], rtol=1.0e-12)
