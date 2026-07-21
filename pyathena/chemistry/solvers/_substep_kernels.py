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


def semi_implicit_temp_mu_update(
    temp_mu: np.ndarray,
    net_cool: np.ndarray,
    d_net_cool_d_temp_mu: np.ndarray,
    inv_heat_cap_per_temp_mu: np.ndarray,
    dt: float,
    out: np.ndarray,
    tmp: np.ndarray,
) -> np.ndarray:
    """Apply one semi-implicit Euler step to the T/mu equation.

    The conserved thermodynamic variable across the substep is
    `temp_mu = T / mu`, not `T`. Mirroring the tigris-ncr C++ side
    (`src/photchem/ncr_solver.hpp::UpdateTemperature`) and mini-RAMSES
    (`cooling/neq_cooling_module.f90::cool_step`, `T2 = T/mu`), the
    cooling sub-step updates `temp_mu` with `mu` held fixed; the
    chemistry sub-step then changes `mu` without changing `temp_mu`.
    The driver rescales `state.T = mu_new * temp_mu` at the end of the
    substep so the per-cell internal energy budget stays consistent
    when species evolve.

    The equation of evolution is

        d(temp_mu) / dt = -inv_heat_cap_per_temp_mu * net_cool

    where `net_cool = cool - heat` is the net cooling rate per unit
    volume [erg / s / cm^3] and `inv_heat_cap_per_temp_mu` is the local
    `(gamma - 1) / (n_H * mu_hyd * k_B)` so the product has units of
    K / s. The factor is mu-independent: `n_H * mu_hyd` is the total
    particle count per unit volume that `n_total = n_H * (1 + A_He -
    x_H2 + x_e) = n_H * mu_hyd / mu` reduces to once the mu factor is
    pulled into the conserved variable.

    Following Kim+2023 Eq. 59 the semi-implicit Euler update is

        temp_mu_new = temp_mu
                      - inv_heat_cap_per_temp_mu * net_cool * dt
                        / (1 - deriv)

    with `deriv = -inv_heat_cap_per_temp_mu * d_net_cool_d_temp_mu *
    dt`. The denominator damps the response when the rate derivative
    w.r.t. temp_mu is large, which is the stability anchor that makes
    the explicit subcycle behave at small dt without going implicit.

    Parameters
    ----------
    temp_mu : ndarray
        Current `T / mu` per cell [K].
    net_cool : ndarray
        `(cool - heat)` per cell [erg / s / cm^3].
    d_net_cool_d_temp_mu : ndarray
        Finite-difference derivative `d(cool - heat) / d(T/mu)` per
        cell.
    inv_heat_cap_per_temp_mu : ndarray
        Per-cell `(gamma - 1) / (n_H * mu_hyd * k_B)`. Multiplying by
        `net_cool` gives the K / s rate of `temp_mu`.
    dt : float
        Substep length.
    out : ndarray
        Output buffer for `temp_mu_new`.
    tmp : ndarray
        Scratch buffer of the same shape used for the denominator
        assembly; written in place.

    Returns
    -------
    ndarray
        `out` containing `temp_mu_new`.
    """
    # tmp = 1 + inv_heat_cap_per_temp_mu * d_net_cool_d_temp_mu * dt
    np.multiply(inv_heat_cap_per_temp_mu, d_net_cool_d_temp_mu, out=tmp)
    np.multiply(tmp, dt, out=tmp)
    np.add(tmp, 1.0, out=tmp)
    # out = inv_heat_cap_per_temp_mu * net_cool * dt / tmp
    np.multiply(inv_heat_cap_per_temp_mu, net_cool, out=out)
    np.multiply(out, dt, out=out)
    np.divide(out, tmp, out=out)
    # out = temp_mu - out
    np.subtract(temp_mu, out, out=out)
    return out
