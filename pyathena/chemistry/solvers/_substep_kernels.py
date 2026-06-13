"""Pure-function vectorised kernels for the explicit subcycling solver.

These helpers operate on whole `(ncell,)` (or larger) strips and write
into caller-owned `out=` buffers so they do not allocate inside the
solver hot path. Each function is a thin wrapper over a NumPy
expression and is exposed at module scope to make unit testing
straightforward without instantiating the solver class.

Conventions:

- Every kernel takes preallocated output buffers and writes results in
  place. None return a freshly allocated array.
- Branching over the strip uses `np.where`; no Python `if` over the
  strip axis. Temperature gates (e.g. `T >= temp_hot1`) are passed in
  as precomputed boolean masks where applicable.
- All arithmetic happens in float64. No silent dtype promotion.
"""
from __future__ import annotations

import numpy as np


def semi_implicit_x_update(
    x: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    dt: float,
    out: np.ndarray,
    tmp: np.ndarray,
) -> np.ndarray:
    """Apply one semi-implicit backward-Euler step to `dx/dt = C - D x`.

    "Semi-implicit" because the rate coefficients `C` and `D` are
    evaluated at the start of the substep and held fixed for the duration
    of this kernel, while the state `x` is solved implicitly. For the
    linearised problem with frozen rates the closed form is
    `x_new = (x + C * dt) / (1 + D * dt)`. The update is unconditionally
    positivity-preserving as long as `x`, `C`, `D` are non-negative.

    Parameters
    ----------
    x, C, D : ndarray
        Operands of identical shape. `x` is the current abundance row,
        `C` and `D` are the semi-implicit creation rate and destruction
        frequency. The kernel does not mutate any of them.
    dt : float
        Substep length.
    out : ndarray
        Output buffer of the same shape as `x`. Written in place.
    tmp : ndarray
        Scratch buffer of the same shape as `x`. Written in place; the
        value after return is unspecified.

    Returns
    -------
    ndarray
        `out`, returned for chaining convenience.
    """
    # out = x + C * dt
    np.multiply(C, dt, out=out)
    np.add(x, out, out=out)
    # tmp = 1 + D * dt
    np.multiply(D, dt, out=tmp)
    np.add(tmp, 1.0, out=tmp)
    # out = out / tmp
    np.divide(out, tmp, out=out)
    return out


def semi_implicit_T_update(
    T: np.ndarray,
    net_cool: np.ndarray,
    d_net_cool_dT: np.ndarray,
    inv_heat_capacity: np.ndarray,
    dt: float,
    out: np.ndarray,
    tmp: np.ndarray,
) -> np.ndarray:
    """Apply one semi-implicit Euler step to the temperature equation.

    The temperature evolves as `dT/dt = -inv_heat_capacity * net_cool`,
    where `net_cool = cool - heat` is the net cooling rate per unit
    volume [erg / s / cm^3] and `inv_heat_capacity` is the local
    `(gamma - 1) / (n_total * k_B)` so the product has units of `K / s`.
    Following Kim+2023 Eq. 59 the semi-implicit Euler update is

        T_new = T - inv_heat_capacity * net_cool * dt / (1 - deriv)

    with `deriv = -inv_heat_capacity * d_net_cool_dT * dt`. The
    denominator damps the response when the rate derivative w.r.t.
    temperature is large, which is the stability anchor that makes
    the explicit subcycle behave at small dt without going implicit.

    Parameters
    ----------
    T : ndarray
        Current temperature per cell.
    net_cool : ndarray
        `(cool - heat)` per cell [erg / s / cm^3].
    d_net_cool_dT : ndarray
        Finite-difference derivative `d(cool - heat) / dT` per cell.
    inv_heat_capacity : ndarray
        Per-cell `(gamma - 1) / (n_total * k_B)` so the product
        `inv_heat_capacity * net_cool` has units of K / s.
    dt : float
        Substep length.
    out : ndarray
        Output buffer for `T_new`.
    tmp : ndarray
        Scratch buffer of the same shape used for the denominator
        assembly; written in place.

    Returns
    -------
    ndarray
        `out` containing `T_new`.
    """
    # tmp = 1 + inv_heat_capacity * d_net_cool_dT * dt
    # (the -- sign on deriv flips because we are subtracting a
    # cool-positive derivative from T)
    np.multiply(inv_heat_capacity, d_net_cool_dT, out=tmp)
    np.multiply(tmp, dt, out=tmp)
    np.add(tmp, 1.0, out=tmp)
    # out = inv_heat_capacity * net_cool * dt / tmp
    np.multiply(inv_heat_capacity, net_cool, out=out)
    np.multiply(out, dt, out=out)
    np.divide(out, tmp, out=out)
    # out = T - out
    np.subtract(T, out, out=out)
    return out
