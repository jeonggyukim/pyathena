"""Rosenbrock23 (Shampine 1982) verbatim port of
`OrdinaryDiffEq.jl/lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_perform_step.jl`
(constant-cache form, autonomous ODE, mass_matrix = I).

Reference:
  - Shampine 1982 SINUM
  - OrdinaryDiffEq.jl Rosenbrock23ConstantCache + perform_step!
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np


_SQRT2 = float(np.sqrt(2.0))
_D = 1.0 / (2.0 + _SQRT2)
_C32 = 6.0 + _SQRT2


@dataclass
class RosenbrockResult:
    n_accept: int
    n_reject: int
    t_final: np.ndarray
    h_final: np.ndarray
    converged: np.ndarray


def _solve_per_cell(W: np.ndarray, b: np.ndarray) -> np.ndarray:
    n, _, ncell = W.shape
    out = np.empty_like(b)
    for c in range(ncell):
        out[:, c] = np.linalg.solve(W[:, :, c], b[:, c])
    return out


def integrate_rosenbrock23(
    f: Callable[[np.ndarray], np.ndarray],
    jac: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y0: np.ndarray,
    t_target: float,
    *,
    h_init: Optional[float] = None,
    h_min: Optional[float] = None,
    h_max: Optional[float] = None,
    atol: float = 1.0e-12,
    rtol: float = 1.0e-6,
    max_steps: int = 1000,
    y_lo: Optional[np.ndarray] = None,
    y_hi: Optional[np.ndarray] = None,
    t_init: Optional[np.ndarray] = None,
    h_init_arr: Optional[np.ndarray] = None,
) -> tuple:
    """Integrate `dy/dt = f(y)` from t=0 to t=t_target per cell using
    Rosenbrock23 with per-cell adaptive h.

    Parameters
    ----------
    f : callable
        `f(y) -> ndarray (nvar, ncell)`. Autonomous RHS.
    jac : callable
        `jac(y, f0) -> ndarray (nvar, nvar, ncell)`. `f0 = f(y)`
        is passed in so analytic Jacobians can reuse it; FD impls
        compute it via `(f(y + dy) - f0) / dy`.
    y0 : ndarray
        Shape (nvar, ncell). Initial condition; mutated to final.
    t_target : float
    h_init, h_min, h_max : float, optional
        Step-size bounds. If None, defaults scale with the
        integration window: `h_init = 1e-6 * t_target`,
        `h_min = 1e-12 * t_target`, `h_max = t_target` (natural
        ceiling: no single step can cover more than one full
        window). Pass explicit values to override, or supply a
        per-cell starting h via `h_init_arr`.
    atol, rtol : float
        Embedded error estimate tolerances.
    max_steps : int
    y_lo, y_hi : ndarray (nvar,) or scalar, optional
        Element-wise clip applied after each accepted step (for
        positivity / physical bounds).

    Returns
    -------
    (y_final, RosenbrockResult)
    """
    if h_init is None:
        h_init = 1.0e-6 * t_target
    if h_min is None:
        h_min = 1.0e-12 * t_target
    if h_max is None:
        h_max = t_target
    nvar, ncell = y0.shape
    y = y0.copy()
    t = (np.zeros(ncell) if t_init is None
         else np.asarray(t_init, dtype=np.float64).copy())
    h = (np.full(ncell, h_init) if h_init_arr is None
         else np.asarray(h_init_arr, dtype=np.float64).copy())
    eye = np.broadcast_to(
        np.eye(nvar)[:, :, None], (nvar, nvar, ncell)).copy()

    done = np.zeros(ncell, dtype=bool)
    n_accept = 0
    n_reject = 0

    for _step in range(max_steps):
        if bool(np.all(done)):
            break

        h_step = np.minimum(h, t_target - t)
        # Inactive (done) cells: leave y unchanged via h_step = 0.
        h_step = np.where(done, 0.0, h_step)

        # Stage 1.
        f0 = f(y)
        J = jac(y, f0)
        dtg = h_step * _D
        # Avoid division by zero for done cells.
        inv_dtg = np.where(dtg > 0.0, 1.0 / dtg, 0.0)
        # W[:, :, c] = J[:, :, c] - I / dtg[c].
        W = J - eye * inv_dtg[None, None, :]
        # k1 = -W^{-1} f0 / dtg.
        z1 = _solve_per_cell(W, f0)
        k1 = -z1 * inv_dtg[None, :]

        # Stage 2.
        y_tmp = y + 0.5 * h_step[None, :] * k1
        if y_lo is not None or y_hi is not None:
            y_tmp = np.clip(y_tmp,
                            y_lo if y_lo is not None else -np.inf,
                            y_hi if y_hi is not None else np.inf)
        f1 = f(y_tmp)
        z2 = _solve_per_cell(W, f1 - k1)
        k2 = -z2 * inv_dtg[None, :] + k1

        # New solution.
        y_new = y + h_step[None, :] * k2
        if y_lo is not None or y_hi is not None:
            y_new = np.clip(y_new,
                            y_lo if y_lo is not None else -np.inf,
                            y_hi if y_hi is not None else np.inf)

        # Stage 3 for embedded error estimate.
        f2 = f(y_new)
        rhs3 = f2 - _C32 * (k2 - f1) - 2.0 * (k1 - f0)
        z3 = _solve_per_cell(W, rhs3)
        k3 = -z3 * inv_dtg[None, :]
        utilde = (h_step[None, :] / 6.0) * (k1 - 2.0 * k2 + k3)

        sc = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
        # Per-cell error norm (max over species).
        err_per = np.abs(utilde) / np.maximum(sc, 1.0e-300)
        err_norm = np.max(err_per, axis=0)
        # Done cells: never reject.
        err_norm = np.where(done, 0.0, err_norm)

        accept = (err_norm <= 1.0) | (h_step <= h_min * 1.0001) | done
        reject = ~accept
        n_accept += int(np.sum(accept & ~done))
        n_reject += int(np.sum(reject))

        # Update accepted cells.
        for k in range(nvar):
            y[k] = np.where(accept, y_new[k], y[k])
        t = np.where(accept, t + h_step, t)

        # Adaptive h: PI controller exponent 1/3 for order 2 (the
        # propagated solution order; embedded estimate is order 3).
        safe = err_norm > 0.0
        factor = np.where(safe, 0.9 * np.power(err_norm,
                                               -1.0 / 3.0), 5.0)
        factor = np.clip(factor, 0.1, 5.0)
        h = np.clip(h * factor, h_min, h_max)
        # Reset h_step's done cells' h to current (no advancement).
        h = np.where(done, h_init, h)

        done = t >= t_target

    return y, RosenbrockResult(
        n_accept=n_accept,
        n_reject=n_reject,
        t_final=t,
        h_final=h,
        converged=done,
    )
