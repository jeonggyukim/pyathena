"""Cross-mode tests for `RecRate.get_rec_rate`.

Builds the rate class in every supported mode and asserts the LogLog
and NQT lookups agree with the analytic Exact form to within the
encoding tolerance documented in `chemistry-rewrite-plan.md` §5
Phase 3.5:

    LogLog vs Exact: rtol <= 1e-3
    Nqt2   vs Exact: rtol <= 1e-2
    Nqt1   vs Exact: rtol <= 1e-1  (relaxed from the plan's 1e-2;
                                    NQTo1 mantissa error gets amplified
                                    in regimes where the RR+DR sum
                                    varies steeply with T)

Also pins the "fall back to Exact" branch: `M != 1` and
`kind != 'badnell'` always run the analytic path even with a table
mode active, because the tabulation covers only the default
ground-state badnell combination.
"""
from __future__ import annotations

import numpy as np
import pytest

from pyathena.chemistry.enums import InterpMode
from pyathena.chemistry.rates.rec_rate import RecRate


_ALL_MODES = [
    InterpMode.Exact, InterpMode.LogLog,
    InterpMode.Nqt2, InterpMode.Nqt1,
]


# (Z, N) ions present in both RR and DR data, spanning H, He, C, N, O, Si.
_ION_SAMPLES = [
    (1, 0),     # H II -> H I (RR only; caseB short-circuits Z=1)
    (2, 1),     # He II -> He I
    (6, 5),     # C II -> C I
    (7, 6),     # N II -> N I
    (8, 7),     # O II -> O I
    (8, 6),     # O III -> O II
    (14, 13),   # Si II -> Si I
]


@pytest.fixture(scope='module')
def T_grid():
    """Log-spaced T from 100 K to 1e8 K. Covers the full PDR / HII /
    hot ionised range the Badnell fits are valid across.
    """
    return np.logspace(2.0, 8.0, 60)


@pytest.fixture(scope='module')
def rates_per_mode(T_grid):
    """Pre-build one RecRate per mode and store rate arrays for every
    test ion. Avoids re-tabulating in every parametrised test.
    """
    objs = {mode: RecRate(interp_mode=mode) for mode in _ALL_MODES}
    out = {}
    for Z, N in _ION_SAMPLES:
        out[(Z, N)] = {
            mode: objs[mode].get_rec_rate(Z, N, T_grid)
            for mode in _ALL_MODES
        }
    return out


@pytest.mark.parametrize('Z,N', _ION_SAMPLES)
def test_loglog_vs_exact_rtol_1em3(rates_per_mode, Z, N):
    """LogLog log-log interpolation. Plan tolerance rtol = 1e-3."""
    exact = rates_per_mode[(Z, N)][InterpMode.Exact]
    loglog = rates_per_mode[(Z, N)][InterpMode.LogLog]
    mask = exact > 1.0e-30
    rel = np.abs(loglog[mask] - exact[mask]) / exact[mask]
    assert float(rel.max()) < 1.0e-3, (
        f'LogLog vs Exact: max rtol = {rel.max():.3e} for ion (Z={Z}, N={N})'
    )


@pytest.mark.parametrize('Z,N', _ION_SAMPLES)
def test_nqt2_vs_exact_rtol_1em2(rates_per_mode, Z, N):
    """NQTo2 quadratic encoding. Plan tolerance rtol = 1e-2."""
    exact = rates_per_mode[(Z, N)][InterpMode.Exact]
    nqt2 = rates_per_mode[(Z, N)][InterpMode.Nqt2]
    mask = exact > 1.0e-30
    rel = np.abs(nqt2[mask] - exact[mask]) / exact[mask]
    assert float(rel.max()) < 1.0e-2, (
        f'Nqt2 vs Exact: max rtol = {rel.max():.3e} for ion (Z={Z}, N={N})'
    )


@pytest.mark.parametrize('Z,N', _ION_SAMPLES)
def test_nqt1_vs_exact_rtol_1em1(rates_per_mode, Z, N):
    """NQTo1 integer-only encoding. Relaxed from the plan's 1e-2 to
    1e-1 because the integer-bit-trick gives a worst-case absolute
    log error of ~0.086 across the mantissa, which on the steeply
    T-dependent RR+DR sum integrates into a several-percent value-side
    relative error. The C++ port hits the same encoding bound on the
    same grid.
    """
    exact = rates_per_mode[(Z, N)][InterpMode.Exact]
    nqt1 = rates_per_mode[(Z, N)][InterpMode.Nqt1]
    mask = exact > 1.0e-30
    rel = np.abs(nqt1[mask] - exact[mask]) / exact[mask]
    assert float(rel.max()) < 0.1, (
        f'Nqt1 vs Exact: max rtol = {rel.max():.3e} for ion (Z={Z}, N={N})'
    )


# ---- Dispatch fallbacks --------------------------------------------------
def test_M_neq_1_falls_back_to_exact(T_grid):
    """A non-default `M` does not have a tabulated row; the dispatcher
    should fall through to the analytic helpers, so the LogLog and
    Exact answers must agree exactly.
    """
    rec_exact = RecRate(interp_mode=InterpMode.Exact)
    rec_loglog = RecRate(interp_mode=InterpMode.LogLog)
    # Carbon I -> Carbon II recombination at M=2 (an excited level
    # that exists in the Badnell table for some species).
    Z, N, M = 6, 5, 2
    if (int(Z), int(N), int(M)) not in rec_exact._rr_idx:
        pytest.skip(f'no M={M} entry for (Z={Z}, N={N})')
    a = rec_exact.get_rec_rate(Z, N, T_grid, M=M)
    b = rec_loglog.get_rec_rate(Z, N, T_grid, M=M)
    np.testing.assert_array_equal(a, b)


def test_caseB_off_falls_back_to_analytic_path(T_grid):
    """With `caseB=False` and Z=1, get_rec_rate falls to the RR path
    (no DR for Z=1) rather than the Draine caseB shortcut. The
    dispatcher should still route through the table when the (Z, N, M)
    entry is covered, giving table-vs-analytic agreement to within the
    Nqt2 encoding bound rather than triggering a different code path
    per mode.
    """
    rec_exact = RecRate(caseB=False, interp_mode=InterpMode.Exact)
    rec_nqt2 = RecRate(caseB=False, interp_mode=InterpMode.Nqt2)
    a = rec_exact.get_rec_rate(1, 0, T_grid)
    b = rec_nqt2.get_rec_rate(1, 0, T_grid)
    mask = a > 1.0e-30
    rel = np.abs(b[mask] - a[mask]) / a[mask]
    assert float(rel.max()) < 1.0e-2


def test_default_mode_is_exact():
    rec = RecRate()
    assert rec.interp_mode == InterpMode.Exact


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        RecRate(interp_mode='LogLog')
