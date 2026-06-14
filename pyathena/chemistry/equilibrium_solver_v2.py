"""Network-agnostic equilibrium solver: log-space Newton + Armijo.

Solves `dx_i/dt = C_i(x) - D_i(x) x_i = 0` for the Newton variables
of the network. Newton variables are `network.evolved` minus
`network.closure_species` -- the closure species are rebuilt by
`network.closure(state)` from a conservation relation.

The Newton variable is `y_k = log10(x_k)` rather than `x_k`
directly. Working in log-space:

  - Newton steps in `y` are naturally bounded: a step of 1.0 in y is
    a factor of 10 in x, so the linearisation cannot over-extrapolate
    multi-decade species (cold-molecular xH2 spans 1e-4 -> 0.5;
    HII-region xHII spans 1e-20 -> 1).
  - Non-negativity is built in: `x = 10^y > 0` always.
  - The closure constraint stays in linear-x space and is enforced
    by `network.closure(state)` between each change to a Newton
    variable.

Algorithm:

  1. Evaluate `F = C - D x` at current `state.x`.
  2. Test `||F||_inf / scale < tol` per cell (on Newton species only).
  3. Build the per-cell Jacobian `dF/dy` via one-sided FD: perturb
     `y_k -> y_k + eps_J`, set `x_k = 10^y_k`, call closure, evaluate.
  4. Solve `J * delta_y = -F` per cell with `lstsq` (handles
     decoupled cells: `delta_y = 0`).
  5. Armijo backtracking on `phi = ||F||_inf^2_sum`: try
     `alpha in {1, 1/2, ...}` until `phi(y + alpha delta_y) < phi(y)`.
     Cap `y_new` at `[log10(x_floor), log10(x_ceil)]` so the
     non-negativity constraint stays explicit.

Anything chemistry-specific (species names, closure formulas, ghost-
species algebra, initial seed) is in the network, not here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


_TINY = 1.0e-30
_LN10 = float(np.log(10.0))


@dataclass
class EqResult:
    n_iter: int                    # outer iterations executed
    converged: np.ndarray          # (ncell,) bool mask
    residual_inf: float            # max relative residual over strip
    armijo_backtracks_total: int   # diagnostic
    method: str = 'newton_log_armijo'


def _allocate_CD(state: Any, nspec: int) -> None:
    ncell = state.nH.shape[0]
    if 'solver:C' not in state.scratch:
        state.alloc_scratch('solver:C', (nspec, ncell))
    if 'solver:D' not in state.scratch:
        state.alloc_scratch('solver:D', (nspec, ncell))


def _F_and_scale(network: Any, state: Any, newton_idx: np.ndarray,
                 C: np.ndarray, D: np.ndarray) -> tuple:
    network.evaluate_CD(state, C, D)
    n_newton = newton_idx.size
    ncell = C.shape[1]
    F = np.empty((n_newton, ncell), dtype=np.float64)
    scale = np.empty((n_newton, ncell), dtype=np.float64)
    for k in range(n_newton):
        i_sp = int(newton_idx[k])
        x = state.x[i_sp]
        F[k] = C[i_sp] - D[i_sp] * x
        scale[k] = np.maximum(
            np.maximum(np.abs(C[i_sp]), np.abs(D[i_sp] * x)),
            _TINY)
    return F, scale


def _solve_linear_per_cell(J: np.ndarray, b: np.ndarray) -> np.ndarray:
    n, _, ncell = J.shape
    out = np.zeros_like(b)
    for c in range(ncell):
        Jc = J[:, :, c]
        bc = b[:, c]
        if not np.any(np.abs(Jc) > _TINY):
            continue
        sol, *_ = np.linalg.lstsq(Jc, bc, rcond=None)
        out[:, c] = sol
    return out


def solve_equilibrium(
    network: Any,
    state: Any,
    *,
    max_iter: int = 30,
    tol: float = 1.0e-6,
    eps_J: float = 1.0e-3,             # FD on log10(x); 1e-3 is 0.0023 in x
    armijo_min_alpha: float = 1.0e-6,
    max_log_step: float = 2.0,         # cap |delta log10 x| per iter
    x_floor: Optional[float] = None,
    x_ceil: float = 1.0 - 1.0e-12,
) -> EqResult:
    """Network-agnostic log-space Newton + Armijo equilibrium solver.

    The Newton variable is `log10(x_k)` for each species in
    `network.evolved \\ network.closure_species`. Working in log
    space gives naturally bounded steps for multi-decade species.

    The solver reads only:
      - `network.evolved`, `network.closure_species` (metadata)
      - `network.evaluate_CD(state, C, D)`
      - `network.closure(state)`
      - `state.species.idx`, `state.x`

    Returns an `EqResult`.
    """
    if x_floor is None:
        x_floor = float(getattr(network, 'x_floor', 1.0e-20))
    y_floor = float(np.log10(x_floor))
    y_ceil = float(np.log10(x_ceil))

    closure_set = set(network.closure_species)
    newton_species = tuple(s for s in network.evolved
                           if s not in closure_set)
    idx = state.species.idx
    newton_idx = np.array(
        [idx[s] for s in newton_species], dtype=np.int64)
    n_newton = newton_idx.size
    ncell = state.nH.shape[0]
    nspec_state = state.x.shape[0]

    if n_newton == 0:
        network.closure(state)
        return EqResult(
            n_iter=0,
            converged=np.ones(ncell, dtype=bool),
            residual_inf=0.0,
            armijo_backtracks_total=0,
        )

    _allocate_CD(state, nspec_state)
    C = state.scratch['solver:C']
    D = state.scratch['solver:D']
    network.closure(state)

    backtracks_total = 0

    for it in range(max_iter):
        # Step 1: F on Newton species at current state.
        F0, scale0 = _F_and_scale(network, state, newton_idx, C, D)
        rel0 = np.abs(F0) / scale0
        converged = np.all(rel0 < tol, axis=0)
        if bool(np.all(converged)):
            return EqResult(
                n_iter=it,
                converged=converged,
                residual_inf=float(np.max(rel0)),
                armijo_backtracks_total=backtracks_total,
            )
        merit_0_sum = float(np.sum(rel0[:, ~converged]
                                   * rel0[:, ~converged]))

        # Step 2: FD Jacobian dF/dy_k via perturbation in log-space.
        # Snapshot y_orig (= log10 x_orig).
        y_orig = np.empty((n_newton, ncell), dtype=np.float64)
        for k in range(n_newton):
            x_k = state.x[int(newton_idx[k])]
            y_orig[k] = np.log10(np.maximum(x_k, x_floor))

        J = np.zeros((n_newton, n_newton, ncell), dtype=np.float64)
        for k in range(n_newton):
            i_sp = int(newton_idx[k])
            y_perturbed = y_orig[k] + eps_J
            state.x[i_sp] = np.clip(10.0 ** y_perturbed, x_floor, x_ceil)
            network.closure(state)
            network.evaluate_CD(state, C, D)
            for j in range(n_newton):
                j_sp = int(newton_idx[j])
                F_jk = C[j_sp] - D[j_sp] * state.x[j_sp]
                J[j, k] = (F_jk - F0[j]) / eps_J
            # Restore.
            state.x[i_sp] = np.clip(10.0 ** y_orig[k], x_floor, x_ceil)
        network.closure(state)

        # Step 3: Newton direction in log-space.
        delta_y = _solve_linear_per_cell(J, -F0)
        # Trust-region cap on |delta_y| per iter so a single log-space
        # Newton step can't fling a variable >100x out of physical range
        # (the linearisation may over-extrapolate when current x is far
        # from the equilibrium x).
        np.clip(delta_y, -max_log_step, max_log_step, out=delta_y)
        # Zero out converged cells so the merit is not bumped by
        # side-effect updates on already-finished cells.
        delta_y[:, converged] = 0.0

        # Step 4: Armijo backtracking on log-space step.
        alpha = 1.0
        accepted = False
        while alpha >= armijo_min_alpha:
            for k in range(n_newton):
                i_sp = int(newton_idx[k])
                y_new = np.clip(y_orig[k] + alpha * delta_y[k],
                                y_floor, y_ceil)
                state.x[i_sp] = 10.0 ** y_new
            network.closure(state)
            F_a, scale_a = _F_and_scale(
                network, state, newton_idx, C, D)
            rel_a = np.abs(F_a) / scale_a
            merit_a_sum = float(np.sum(rel_a[:, ~converged]
                                       * rel_a[:, ~converged]))
            if merit_a_sum < merit_0_sum:
                accepted = True
                break
            alpha *= 0.5
            backtracks_total += 1
        if not accepted:
            for k in range(n_newton):
                state.x[int(newton_idx[k])] = np.clip(
                    10.0 ** y_orig[k], x_floor, x_ceil)
            network.closure(state)
            break

    F_f, scale_f = _F_and_scale(network, state, newton_idx, C, D)
    rel_f = np.abs(F_f) / scale_f
    converged = np.all(rel_f < tol, axis=0)
    return EqResult(
        n_iter=it + 1,
        converged=converged,
        residual_inf=float(np.max(rel_f)),
        armijo_backtracks_total=backtracks_total,
    )
