"""Vectorised NQT (Not Quite Transcendental) helpers.

Mirrors the four scalar `static inline` helpers in
`tigris-ncr/src/photchem/ncr_rates.hpp:554-592`. The encoding follows
Hammond et al. 2025 ApJS 277, 65 (Listing 1) and approximates log2 /
exp2 with IEEE 754 bit manipulation:

- `nqt1_log(x)` and `nqt1_exp(y)` use integer bit operations only;
  worst-case relative error ~1.5% across the full positive float64
  range.
- `nqt2_log(x)` and `nqt2_exp(y)` add a quadratic mantissa correction;
  worst-case relative error ~0.1% but `nqt2_exp` costs one `sqrt`.

The Python helpers are bit-identical to the C++ helpers on every
finite positive float64 input: both forms cast the IEEE 754 bit
pattern through int64, do plain integer arithmetic, and cast back.
This lets the rate-table interpolation modes (`InterpMode.Nqt1`,
`InterpMode.Nqt2`) match the C++ side to the last bit when the same
T-grid and rate-floor conventions are used.

Magic constants:

- `_AS_INT_ONE` = 0x3FF0000000000000 = 4607182418800017408. IEEE 754
  bit pattern of `1.0` interpreted as int64. Subtracting from the
  bit pattern of `x` extracts the "biased" log2 representation.
- `_TWO_POW_52` = 4503599627370496. Equals `as_int(2.0) - as_int(1.0)`
  and equals 2 ** 52. Dividing by it scales the biased log2 into
  natural-log-like units that approximate log2(x).
- `_MANTISSA_MASK` = 0x000FFFFFFFFFFFFF. The 52 low bits of an int64,
  i.e. the mantissa of a double. Used by `nqt2_log` to recover the
  fractional mantissa for the quadratic correction.

Allocation contract: every helper returns a freshly allocated array.
That is fine because these are called at table-build time (rate-class
`__init__`), never inside the substep hot path. The lookup path that
will consume the NQT tables does its own in-place ops with
caller-owned `out=` buffers.
"""
from __future__ import annotations

import numpy as np

__all__ = ['nqt1_log', 'nqt1_exp', 'nqt2_log', 'nqt2_exp']

# IEEE 754 double bit patterns. See module docstring.
_AS_INT_ONE: int = 4607182418800017408  # 0x3FF0000000000000
_TWO_POW_52: int = 4503599627370496      # 2 ** 52 = as_int(2.0) - as_int(1.0)
_MANTISSA_MASK: int = 0x000FFFFFFFFFFFFF
_TWO_POW_52_F: float = float(_TWO_POW_52)
_EXP_BIAS: int = 1023


def _as_int(x: np.ndarray) -> np.ndarray:
    """Bit-reinterpret a float64 array as int64. Zero-copy view."""
    arr = np.ascontiguousarray(x, dtype=np.float64)
    return arr.view(np.int64)


def _as_float(i: np.ndarray) -> np.ndarray:
    """Bit-reinterpret an int64 array as float64. Zero-copy view."""
    arr = np.ascontiguousarray(i, dtype=np.int64)
    return arr.view(np.float64)


def nqt1_log(x: np.ndarray) -> np.ndarray:
    """NQTo1 approximation of log2(x).

    Closed form: `(as_int(x) - as_int(1.0)) / 2**52`. Linear in the
    biased exponent; piecewise-linear in the mantissa with a worst-case
    ~1.5% relative error. Mirrors `nqt1_log` at
    `tigris-ncr/src/photchem/ncr_rates.hpp:557`.

    Parameters
    ----------
    x : ndarray
        Strictly positive float64 input.

    Returns
    -------
    ndarray
        Approximate log2(x), shape matching `x`.
    """
    i = _as_int(x)
    return (i - _AS_INT_ONE).astype(np.float64) / _TWO_POW_52_F


def nqt1_exp(y: np.ndarray) -> np.ndarray:
    """Inverse of `nqt1_log`. NQTo1 approximation of 2**y.

    Closed form: `as_float(int(y * 2**52) + as_int(1.0))`. Cheap
    integer bit cast; piecewise-linear approximant of the mantissa.
    Mirrors `nqt1_exp` at `tigris-ncr/src/photchem/ncr_rates.hpp:563`.

    Parameters
    ----------
    y : ndarray
        Real-valued float64 input. Allowed range is roughly
        `[-1022, 1023]`; outside it the int64 cast overflows the
        IEEE 754 exponent.

    Returns
    -------
    ndarray
        Approximate 2**y, shape matching `y`.
    """
    y_arr = np.ascontiguousarray(y, dtype=np.float64)
    # int(y * 2**52) — the C++ side uses `static_cast<int64>` which is
    # truncation toward zero. Use np.floor to match the nqt2_exp
    # convention; the difference vanishes for non-negative y and is
    # one ULP for negative y at the bin boundary, which is fine for
    # the integer-only path that never feeds into a sqrt.
    i = (y_arr * _TWO_POW_52_F).astype(np.int64) + _AS_INT_ONE
    return _as_float(i).copy()


def nqt2_log(x: np.ndarray) -> np.ndarray:
    """NQTo2 approximation of log2(x).

    Adds a quadratic mantissa correction `b * (1 - b) / 3` on top of
    `nqt1_log`, where `b` is the fractional mantissa of `x` in
    `[0, 1)`. Worst-case relative error ~0.1%. Mirrors `nqt2_log` at
    `tigris-ncr/src/photchem/ncr_rates.hpp:573`.

    Parameters
    ----------
    x : ndarray
        Strictly positive float64 input.

    Returns
    -------
    ndarray
        Approximate log2(x), shape matching `x`.
    """
    i = _as_int(x)
    b = (i & _MANTISSA_MASK).astype(np.float64) / _TWO_POW_52_F
    base = (i - _AS_INT_ONE).astype(np.float64) / _TWO_POW_52_F
    return base + b * (1.0 - b) / 3.0


def nqt2_exp(y: np.ndarray) -> np.ndarray:
    """Inverse of `nqt2_log`. NQTo2 approximation of 2**y.

    Inverts the quadratic by solving `b + b * (1 - b) / 3 = f` for the
    mantissa fraction `f`, giving `b = 2 - sqrt(4 - 3 * f)`. Uses
    `np.floor` (not truncation toward zero) so large-negative `y` —
    the kind that appears for stored rates clipped to ~1e-100 — gives
    the correct integer/mantissa split rather than NaN inside the
    sqrt. Mirrors `nqt2_exp` at
    `tigris-ncr/src/photchem/ncr_rates.hpp:580`.

    Parameters
    ----------
    y : ndarray
        Real-valued float64 input.

    Returns
    -------
    ndarray
        Approximate 2**y, shape matching `y`.
    """
    y_arr = np.ascontiguousarray(y, dtype=np.float64)
    # True floor (NOT truncation toward zero) so the fractional part
    # of large-negative y is non-negative and 4 - 3f stays positive.
    e_real = np.floor(y_arr)
    e = e_real.astype(np.int64)
    f = y_arr - e_real
    # Solve b + b*(1 - b)/3 = f  ->  b**2 - 4b + 3f = 0  ->
    # b = 2 - sqrt(4 - 3f); the other root b = 2 + sqrt(...) sits
    # outside [0, 1).
    b = 2.0 - np.sqrt(4.0 - 3.0 * f)
    mantissa = (b * _TWO_POW_52_F).astype(np.int64)
    i = ((e + _EXP_BIAS) << 52) | mantissa
    return _as_float(i).copy()
