"""Adapter that exposes a chemistry network's `evaluate_CD` +
`closure` pair as a generic autonomous ODE `f(y) = C(y) - D(y) y` for
the Rosenbrock23 integrator. Lets `rosenbrock.py` stay agnostic about
chemistry."""
from __future__ import annotations

from typing import Any, Callable, Tuple

import numpy as np

from .rosenbrock import integrate_rosenbrock23, RosenbrockResult


def make_chemistry_f_jac(network: Any, state: Any,
                         eps_J: float = 1.0e-3) -> Tuple[
                             np.ndarray, Callable, Callable]:
    """Return `(newton_idx, f, jac)` for the Newton variables of
    `network` on `state`.

    The closures snapshot the per-cell index/scratch handles and
    call `network.evaluate_CD` + `network.closure(state)`. `f(y)`
    overwrites `state.x[newton]` with the supplied `y`, applies
    closure (rebuilds closure_species + ghosts), then reads back the
    residual `C - D y`.

    FD Jacobian uses linear-space perturbation with an x_typical
    floor at 1e-12 so the FD signal is informative even when a
    Newton variable is near the species floor.
    """
    closure_set = set(network.closure_species)
    newton_species = tuple(s for s in network.evolved
                           if s not in closure_set)
    idx = state.species.idx
    newton_idx = np.array(
        [idx[s] for s in newton_species], dtype=np.int64)
    nvar = newton_idx.size
    ncell = state.nH.shape[0]
    nspec_state = state.x.shape[0]
    # Allocate scratch buffers C and D (re-used per call).
    if 'solver:C' not in state.scratch:
        state.alloc_scratch('solver:C', (nspec_state, ncell))
    if 'solver:D' not in state.scratch:
        state.alloc_scratch('solver:D', (nspec_state, ncell))
    C = state.scratch['solver:C']
    D = state.scratch['solver:D']

    def _write_y_and_close(y):
        for k in range(nvar):
            state.x[int(newton_idx[k])] = y[k]
        network.closure(state)

    def f(y):
        _write_y_and_close(y)
        network.evaluate_CD(state, C, D)
        out = np.empty((nvar, ncell), dtype=np.float64)
        for k in range(nvar):
            i_sp = int(newton_idx[k])
            out[k] = C[i_sp] - D[i_sp] * state.x[i_sp]
        return out

    def jac(y, f0):
        floor_scale = 1.0e-12
        Jmat = np.zeros((nvar, nvar, ncell), dtype=np.float64)
        for k in range(nvar):
            y_pert = y.copy()
            dy_k = eps_J * np.maximum(np.abs(y[k]), floor_scale)
            y_pert[k] = y[k] + dy_k
            f_p = f(y_pert)
            for j in range(nvar):
                Jmat[j, k] = (f_p[j] - f0[j]) / dy_k
        # Restore state to y.
        _write_y_and_close(y)
        return Jmat

    return newton_idx, f, jac


def integrate_chemistry_to_equilibrium(
    network: Any,
    state: Any,
    t_target: float,
    *,
    h_init: float = 1.0e3,
    h_min: float = 1.0e-6,
    h_max: float = 1.0e20,
    atol: float = 1.0e-14,
    rtol: float = 1.0e-6,
    max_steps: int = 2000,
    eps_J: float = 1.0e-3,
    x_floor: float = 1.0e-20,
    x_ceil: float = 1.0 - 1.0e-12,
) -> RosenbrockResult:
    """Drive `state.x` to chemical equilibrium by integrating
    `dy/dt = C(y) - D(y) y` for the Newton-variable subset of
    `network.evolved` to `t = t_target`. Closure species and ghost
    rows are rebuilt by `network.closure` after every change to a
    Newton variable.
    """
    newton_idx, f, jac = make_chemistry_f_jac(network, state, eps_J)
    nvar = newton_idx.size
    ncell = state.nH.shape[0]
    y0 = np.empty((nvar, ncell), dtype=np.float64)
    for k in range(nvar):
        y0[k] = state.x[int(newton_idx[k])].copy()
    y_final, result = integrate_rosenbrock23(
        f=f, jac=jac, y0=y0, t_target=t_target,
        h_init=h_init, h_min=h_min, h_max=h_max,
        atol=atol, rtol=rtol, max_steps=max_steps,
        y_lo=x_floor, y_hi=x_ceil,
    )
    # Final write-back through closure.
    for k in range(nvar):
        state.x[int(newton_idx[k])] = y_final[k]
    network.closure(state)
    return result
