"""Bit-parity + round-trip tests for the NQT log/exp helpers.

The helpers in `pyathena.chemistry.rates._nqt` mirror the C++ static
inline helpers at `tigris-ncr/src/photchem/ncr_rates.hpp:557-592`.
The numerical recipe is fully specified by the magic constants
(`0x3FF0000000000000`, `2**52`, the mantissa mask). These tests pin
the contract:

1. `nqt1_log(1.0) == 0.0`, `nqt1_log(2.0) == 1.0`, etc. — exact
   values for IEEE 754 integers of 2.
2. `nqt1_exp(nqt1_log(x)) ~= x` and `nqt2_exp(nqt2_log(x)) ~= x` —
   round-trip bijectivity to within the encoding precision.
3. Hammond+2025-reported error bounds — ~1.5% on NQTo1 and ~0.1% on
   NQTo2 across a representative log-T grid.
4. Vectorized form matches the per-element scalar form bit-for-bit.
"""
from __future__ import annotations

import numpy as np
import pytest

from pyathena.chemistry.rates._nqt import (
    nqt1_log, nqt1_exp, nqt2_log, nqt2_exp,
)


# ---- Exact values at powers of two ---------------------------------------
@pytest.mark.parametrize('x, expected', [
    (1.0, 0.0),
    (2.0, 1.0),
    (4.0, 2.0),
    (0.5, -1.0),
    (1024.0, 10.0),
    (1.0 / 1024.0, -10.0),
])
def test_nqt1_log_exact_at_powers_of_two(x, expected):
    """Powers of 2 have zero mantissa fraction, so nqt1_log is exact."""
    arr = np.array([x], dtype=np.float64)
    result = nqt1_log(arr)
    assert result[0] == expected


@pytest.mark.parametrize('x, expected', [
    (1.0, 0.0),
    (2.0, 1.0),
    (4.0, 2.0),
    (0.5, -1.0),
])
def test_nqt2_log_exact_at_powers_of_two(x, expected):
    """Powers of 2 have zero mantissa fraction; the quadratic
    correction `b * (1 - b) / 3` vanishes, so nqt2_log is also exact.
    """
    arr = np.array([x], dtype=np.float64)
    result = nqt2_log(arr)
    assert result[0] == expected


# ---- Round-trip bijectivity ----------------------------------------------
def test_nqt1_round_trip_within_float_precision():
    """nqt1_exp(nqt1_log(x)) recovers x to within float64 round-off.
    The encoding is bijective in exact arithmetic; the small residual
    comes from the int-to-float cast inside `nqt1_log` losing one or
    two ULP when `(as_int(x) - as_int(1.0))` exceeds 2**53.
    """
    x = np.logspace(-100.0, 100.0, num=401, dtype=np.float64)
    round_trip = nqt1_exp(nqt1_log(x))
    np.testing.assert_allclose(round_trip, x, rtol=1.0e-13, atol=0.0)


def test_nqt2_round_trip_within_float_precision():
    """nqt2_exp(nqt2_log(x)) recovers x to within float64 round-off.
    The sqrt-based inversion adds a few ULP on top of the int-cast
    residual already present in nqt1.
    """
    x = np.logspace(-100.0, 100.0, num=401, dtype=np.float64)
    round_trip = nqt2_exp(nqt2_log(x))
    np.testing.assert_allclose(round_trip, x, rtol=1.0e-13, atol=0.0)


# ---- Hammond+2025 error bounds (absolute error on log2(x)) ---------------
def _ref_log2(x: np.ndarray) -> np.ndarray:
    return np.log2(x)


def test_nqt1_log_absolute_error_under_0p09():
    """NQTo1 is piecewise linear in the mantissa fraction `b` over
    each octave; the worst absolute error on log2(x) is
    max_b |b - log2(1 + b)| ~= 0.086, attained near b ~ 1/ln(2) - 1.
    Use 0.09 as the test threshold for headroom.
    """
    x = np.logspace(0.0, 9.0, num=901, dtype=np.float64)
    approx = nqt1_log(x)
    exact = _ref_log2(x)
    abs_err = np.abs(approx - exact)
    assert float(abs_err.max()) < 0.09, (
        f'NQTo1 worst-case absolute error {abs_err.max()} '
        f'exceeded 0.09 (Hammond+2025 ~0.086)'
    )


def test_nqt2_log_absolute_error_under_0p012():
    """NQTo2 adds the quadratic mantissa correction `b * (1 - b) / 3`.
    Worst-case absolute error on log2(x) is attained near `b ~ 0.25`,
    where `log2(1.25) - (0.25 + 0.25 * 0.75 / 3) ~ 0.0094`. Use 0.012
    as the test threshold for headroom.
    """
    x = np.logspace(0.0, 9.0, num=901, dtype=np.float64)
    approx = nqt2_log(x)
    exact = _ref_log2(x)
    abs_err = np.abs(approx - exact)
    assert float(abs_err.max()) < 0.012, (
        f'NQTo2 worst-case absolute error {abs_err.max()} '
        f'exceeded 0.012 (analytic max ~0.0094)'
    )


# ---- Scalar consistency check --------------------------------------------
def test_nqt_vectorized_matches_scalar_form():
    """The helpers process arrays as a single bitcast; this test
    asserts that the vector path agrees with a per-element loop using
    the same arithmetic.
    """
    x = np.array([0.5, 1.0, 1.5, 2.0, 10.0, 1.0e4, 1.0e-30], dtype=np.float64)
    vec_log = nqt1_log(x)
    vec_exp = nqt1_exp(nqt1_log(x))
    # Scalar fallback: apply nqt1_log to a 1-element array per entry.
    scalar_log = np.array(
        [nqt1_log(np.array([xi]))[0] for xi in x], dtype=np.float64
    )
    scalar_exp = np.array(
        [nqt1_exp(nqt1_log(np.array([xi])))[0] for xi in x],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(vec_log, scalar_log)
    np.testing.assert_array_equal(vec_exp, scalar_exp)
