"""Equilibrium solvers for the NCR chemistry network.

Distinct from the time-dependent integrator (`ExplicitSubcyclingSolver`,
backward-Euler-per-substep) used during production runs. These
routines solve `dx/dt = 0` directly for the steady-state composition
at given (T, nH, ionising-rate inputs), used by:

  - problem-generator initial conditions (`photchem_equil` iprob=1,
    `radiative_snr` HII region setup, ...), where the time-integrated
    BE-marched-to-equilibrium path is correct but slow.
  - thermal/chemical equilibrium regression tests.
  - as a known-good reference when benchmarking time-dependent
    solver convergence to steady state.

Candidate methods (this module hosts the prototypes; the eventual
production routine is the one that wins the benchmark):

  GaussSeidel
      Repeatedly substitute x_i = C_i / D_i + apply closure. Cheapest
      possible step; linear convergence rate set by the (C, D) self-
      coupling. Match to the BE-marched baseline in the high-dt
      limit.

  Newton
      Newton-Raphson on `F(x) = C(x) - D(x) * x = 0` for the evolved
      species (HII, H2); HI from closure. Numerical 2x2 Jacobian via
      one-sided FD on `x` (relative perturbation `eps_J`). Cramer
      solve. Quadratic convergence near the root.

  NewtonLog
      Same as Newton but in log-space variables `log x_H2`,
      `log x_HII`. Built-in positivity.

  BackwardEulerLarge
      Calls `ExplicitSubcyclingSolver.step(dt_large)` with a huge
      `dt` so the BE step approaches the steady-state limit. Baseline
      against which the other methods are benchmarked.

Each method returns an `EqSolverResult` carrying the converged state,
iteration count, and the final residual norm `||C - D * x||_inf` over
the strip. The state in-place receives the converged abundances.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class EqSolverResult:
    n_iter: int                  # outer iteration count
    converged: np.ndarray        # per-cell bool mask
    residual_inf: float          # max ||C - D*x||_inf over strip
    f_xH2_inf: float             # max relative xH2 change in final iter
    f_xHII_inf: float            # max relative xHII change in final iter
    method: str


_TINY = 1.0e-30


def _allocate_CD(state: Any) -> None:
    """Make sure the (C, D) scratch buffers exist for the network's
    `evaluate_CD` to write into."""
    ncell = state.nH.shape[0]
    nspec = state.x.shape[0]
    if 'solver:C' not in state.scratch:
        state.alloc_scratch('solver:C', (nspec, ncell))
    if 'solver:D' not in state.scratch:
        state.alloc_scratch('solver:D', (nspec, ncell))


def _residual(network: Any, state: Any) -> float:
    """Compute ||C - D*x||_inf for the evolved species (HII, H2)."""
    _allocate_CD(state)
    C = state.scratch['solver:C']
    D = state.scratch['solver:D']
    network.evaluate_CD(state, C, D)
    idx = state.species.idx
    i_HII = idx['HII']
    i_H2 = idx['H2']
    r_HII = C[i_HII] - D[i_HII] * state.x[i_HII]
    r_H2 = C[i_H2] - D[i_H2] * state.x[i_H2]
    return float(max(np.max(np.abs(r_HII)), np.max(np.abs(r_H2))))


def solve_gauss_seidel(
    network: Any,
    state: Any,
    max_iter: int = 200,
    tol: float = 1.0e-6,
) -> EqSolverResult:
    """Fixed-point iteration `x_new = C / D`, then closure.

    Cheapest possible step. One `evaluate_CD` + one `closure` per
    iteration. Convergence rate set by the (C, D) cross-coupling
    (matches the high-dt limit of backward Euler in the (C, D) split).
    """
    _allocate_CD(state)
    idx = state.species.idx
    i_HI = idx['HI']
    i_HII = idx['HII']
    i_H2 = idx['H2']
    C = state.scratch['solver:C']
    D = state.scratch['solver:D']

    f_xH2 = np.zeros_like(state.x[i_H2])
    f_xHII = np.zeros_like(state.x[i_HII])
    converged = np.zeros_like(state.x[i_HII], dtype=bool)

    for it in range(max_iter):
        network.evaluate_CD(state, C, D)
        xH2_prev = state.x[i_H2].copy()
        xHII_prev = state.x[i_HII].copy()

        # Fixed-point update on the evolved species.
        state.x[i_H2] = C[i_H2] / np.maximum(D[i_H2], _TINY)
        state.x[i_HII] = C[i_HII] / np.maximum(D[i_HII], _TINY)
        # Closure: clamp + xHI = 1 - 2*xH2 - xHII + rebuild ghosts.
        network.closure(state)

        f_xH2 = np.abs(state.x[i_H2] - xH2_prev) / np.maximum(
            np.abs(state.x[i_H2]), _TINY)
        f_xHII = np.abs(state.x[i_HII] - xHII_prev) / np.maximum(
            np.abs(state.x[i_HII]), _TINY)
        converged = (f_xH2 < tol) & (f_xHII < tol)
        if bool(np.all(converged)):
            break

    return EqSolverResult(
        n_iter=it + 1,
        converged=converged,
        residual_inf=_residual(network, state),
        f_xH2_inf=float(np.max(f_xH2)),
        f_xHII_inf=float(np.max(f_xHII)),
        method='gauss_seidel',
    )


def solve_newton(
    network: Any,
    state: Any,
    max_iter: int = 50,
    tol: float = 1.0e-6,
    eps_J: float = 1.0e-6,
    damping: float = 1.0,
    mode: str = 'chem_only',
    cooling: Optional[Any] = None,
    switch_threshold: float = 0.9,
    seed: bool = True,
) -> EqSolverResult:
    """Newton on `F(x_H2, x_minor_H) = (C - D*x)[H2, HII] = 0`.

    The hydrogen minority species (xHII when neutral-dominated, xHI
    when ionised-dominated) is the Newton unknown; the majority is
    rebuilt from closure each iteration. The switch is per-cell:
    `xHII > switch_threshold` => use xHI as the unknown (default
    threshold 0.9). Avoids catastrophic cancellation in the
    `1 - xHII - 2*xH2` closure when the gas is nearly fully ionised.

    2x2 numerical Jacobian via one-sided FD with relative perturbation
    `eps_J`. Cramer solve. `damping` = step-size factor (1.0 = full
    Newton, < 1.0 = damped).

    Convergence test: relative residual
    `|F_i| / max(|C_i|, |D_i x_i|) < tol` for each species, ensuring
    cells where the Newton step is clipped by a trust region or by a
    singular Jacobian do NOT register as converged.

    `mode`: 'chem_only' fixes T; 'chem_thermal' adds T as a Newton
    variable, requires `cooling` callable that writes per-cell
    (heat, cool) into provided output arrays. The thermal-equilibrium
    residual is F_T = (heat - cool) / (heat + cool) following the
    `chem_only=false` branch of `ncr_solver.hpp:SolveToEquilibrium`.
    """
    if mode == 'chem_thermal':
        if cooling is None:
            raise ValueError(
                "solve_newton(mode='chem_thermal') needs a `cooling` "
                'arg that writes (heat, cool) into out arrays')
        return _solve_newton_3x3(
            network, state, cooling, max_iter, tol, eps_J, damping)
    elif mode != 'chem_only':
        raise ValueError(
            f"unknown mode {mode!r}; expected 'chem_only' or "
            "'chem_thermal'")
    _allocate_CD(state)
    idx = state.species.idx
    i_HI = idx['HI']
    i_HII = idx['HII']
    i_H2 = idx['H2']
    C = state.scratch['solver:C']
    D = state.scratch['solver:D']
    floor = network.x_floor

    # Pre-seed xHII / xCII / xe from the analytic HII + CII
    # equilibrium at the caller's xH2 (Saha-like x_OII filled by
    # fill_ghosts). Starts Newton near the root; subsequent steps
    # only need to polish for the H2 coupling. Eliminates the
    # bad-initial-guess overshoot that pushes cold cells to floor.
    if seed:
        from pyathena.chemistry.equilibrium_seed import eq_xHII_xe
        chi_FUV = float(getattr(state, 'chi_FUV', 0.0)
                        if not hasattr(state, 'chi_FUV')
                           or getattr(state, 'chi_FUV', None) is None
                        else 0.0)
        xi_CR_arr = np.asarray(state.xi_CR)
        xi_CR_scalar = float(xi_CR_arr.max()) if xi_CR_arr.size else 0.0
        xHII_seed, _xCII_seed, xe_seed, xHI_seed = eq_xHII_xe(
            T=state.T, nH=state.nH, xH2=state.x[i_H2],
            xi_CR=xi_CR_scalar, G_PE=chi_FUV, G_CI=chi_FUV,
            Z_d=float(np.asarray(state.Z_d).max()),
            Z_g=float(np.asarray(state.Z_g).max()))
        state.x[i_HII] = xHII_seed
        state.x[i_HI] = xHI_seed
        i_e = idx.get('electron')
        if i_e is not None:
            state.x[i_e] = xe_seed
        network.closure(state)

    def _set_H_from_unknown(xH2_arr, x_unknown_arr, use_xHI):
        """Write (xHI, xHII) into state given xH2 + the chosen
        H-minority unknown. `use_xHI` is a bool mask (True => the
        unknown is xHI; False => the unknown is xHII)."""
        # If unknown is xHI: x_unknown_arr -> xHI; xHII = closure.
        # Else: x_unknown_arr -> xHII; xHI = closure.
        state.x[i_HI] = np.where(
            use_xHI,
            np.maximum(x_unknown_arr, floor),
            np.maximum(1.0 - 2.0 * xH2_arr - x_unknown_arr, floor))
        state.x[i_HII] = np.where(
            use_xHI,
            np.maximum(1.0 - 2.0 * xH2_arr - x_unknown_arr, floor),
            np.maximum(x_unknown_arr, floor))

    # Initial unknown selection: xHI as unknown for cells where
    # xHII > switch_threshold; xHII otherwise.
    use_xHI = state.x[i_HII] > switch_threshold
    x_unknown = np.where(use_xHI, state.x[i_HI], state.x[i_HII]).copy()
    xH2 = state.x[i_H2].copy()
    _set_H_from_unknown(xH2, x_unknown, use_xHI)
    network.fill_ghosts(state)

    res_H2 = np.zeros_like(state.x[i_H2])
    res_HII = np.zeros_like(state.x[i_HII])
    converged = np.zeros_like(state.x[i_HII], dtype=bool)

    for it in range(max_iter):
        # Re-select unknown each iter (rare; only when crossing
        # threshold after a Newton step). Hysteresis is fine because
        # the residual check governs final convergence.
        use_xHI_new = state.x[i_HII] > switch_threshold
        if np.any(use_xHI_new != use_xHI):
            use_xHI = use_xHI_new
            x_unknown = np.where(
                use_xHI, state.x[i_HI], state.x[i_HII]).copy()

        # Base residual at current x.
        network.evaluate_CD(state, C, D)
        F_H2_0 = C[i_H2] - D[i_H2] * state.x[i_H2]
        F_HII_0 = C[i_HII] - D[i_HII] * state.x[i_HII]

        xH2_0 = state.x[i_H2].copy()
        x_unknown_0 = x_unknown.copy()

        # Column 1: perturb xH2.
        dxH2 = eps_J * np.maximum(np.abs(xH2_0), _TINY)
        state.x[i_H2] = xH2_0 + dxH2
        _set_H_from_unknown(state.x[i_H2], x_unknown_0, use_xHI)
        network.fill_ghosts(state)
        network.evaluate_CD(state, C, D)
        F_H2_a = C[i_H2] - D[i_H2] * state.x[i_H2]
        F_HII_a = C[i_HII] - D[i_HII] * state.x[i_HII]
        J11 = (F_H2_a - F_H2_0) / dxH2
        J21 = (F_HII_a - F_HII_0) / dxH2

        # Restore xH2; perturb the H minority unknown.
        state.x[i_H2] = xH2_0
        dxu = eps_J * np.maximum(np.abs(x_unknown_0), _TINY)
        _set_H_from_unknown(xH2_0, x_unknown_0 + dxu, use_xHI)
        network.fill_ghosts(state)
        network.evaluate_CD(state, C, D)
        F_H2_b = C[i_H2] - D[i_H2] * state.x[i_H2]
        F_HII_b = C[i_HII] - D[i_HII] * state.x[i_HII]
        J12 = (F_H2_b - F_H2_0) / dxu
        J22 = (F_HII_b - F_HII_0) / dxu

        # Restore state to (xH2_0, x_unknown_0).
        _set_H_from_unknown(xH2_0, x_unknown_0, use_xHI)
        network.fill_ghosts(state)

        # Detect degenerate H2 row (hot_mask zeroes C_H2 = D_H2 in
        # NCR; J11 = J12 = 0 there). In those cells xH2 is forced to
        # the floor and only F_HII matters; the system decouples to
        # a 1-D Newton on the H-minority unknown.
        h2_degenerate = (np.abs(J11) + np.abs(J12)) < 1.0e-60
        # Standard 2x2 Cramer for non-degenerate cells.
        det = J11 * J22 - J12 * J21
        safe = np.abs(det) > 1.0e-50
        d_xH2_2x2 = np.where(
            safe, -(J22 * F_H2_0 - J12 * F_HII_0) / det, 0.0)
        d_xu_2x2 = np.where(
            safe, -(J11 * F_HII_0 - J21 * F_H2_0) / det, 0.0)
        # 1-D Newton fallback when H2 row degenerate.
        d_xu_1d = np.where(
            np.abs(J22) > 1.0e-50, -F_HII_0 / J22, 0.0)
        d_xH2 = np.where(h2_degenerate, 0.0, d_xH2_2x2)
        d_xu = np.where(h2_degenerate, d_xu_1d, d_xu_2x2)

        xH2 = xH2_0 + damping * d_xH2
        x_unknown = x_unknown_0 + damping * d_xu
        # Clamp to (floor, 1) so closure stays valid.
        xH2 = np.clip(xH2, floor, 0.5)
        x_unknown = np.clip(x_unknown, floor, 1.0)

        state.x[i_H2] = xH2
        _set_H_from_unknown(xH2, x_unknown, use_xHI)
        network.fill_ghosts(state)

        # Residual-based convergence test (relative). |F| / max(|C|,
        # |D*x|) gives the relative imbalance between source and sink;
        # passes 0 only when the equation is satisfied, regardless of
        # whether the Newton step was clipped or singular. Species
        # whose C and D BOTH vanish (e.g. H2 in T > T_HOT1 cells where
        # the NCR network masks H2 chemistry off) are decoupled from
        # the equilibrium and count as trivially converged for that
        # row.
        network.evaluate_CD(state, C, D)
        F_H2_new = C[i_H2] - D[i_H2] * state.x[i_H2]
        F_HII_new = C[i_HII] - D[i_HII] * state.x[i_HII]
        scale_H2 = np.maximum(np.abs(C[i_H2]),
                              np.abs(D[i_H2] * state.x[i_H2]))
        scale_HII = np.maximum(np.abs(C[i_HII]),
                               np.abs(D[i_HII] * state.x[i_HII]))
        H2_decoupled = scale_H2 < _TINY
        HII_decoupled = scale_HII < _TINY
        res_H2 = np.where(
            H2_decoupled, 0.0, np.abs(F_H2_new) / np.maximum(scale_H2, _TINY))
        res_HII = np.where(
            HII_decoupled, 0.0,
            np.abs(F_HII_new) / np.maximum(scale_HII, _TINY))
        converged = (res_H2 < tol) & (res_HII < tol)
        # Step-based info for the result struct.
        f_xH2 = np.abs(state.x[i_H2] - xH2_0) / np.maximum(
            np.abs(state.x[i_H2]), _TINY)
        f_xHII = np.abs(state.x[i_HII] - (np.where(
            use_xHI,
            1.0 - 2.0 * xH2_0 - x_unknown_0, x_unknown_0))) / np.maximum(
                np.abs(state.x[i_HII]), _TINY)
        if bool(np.all(converged)):
            break

    return EqSolverResult(
        n_iter=it + 1,
        converged=converged,
        residual_inf=_residual(network, state),
        f_xH2_inf=float(np.max(f_xH2)),
        f_xHII_inf=float(np.max(f_xHII)),
        method='newton',
    )


def _solve_newton_3x3(
    network: Any,
    state: Any,
    cooling: Any,
    max_iter: int,
    tol: float,
    eps_J: float,
    damping: float,
) -> EqSolverResult:
    """3x3 Newton for chem+thermal equilibrium.

    Variables: (x_H2, x_HII, log T). Residuals:
       F_H2  = C[H2]  - D[H2]  * x_H2
       F_HII = C[HII] - D[HII] * x_HII
       F_T   = (heat - cool) / (heat + cool)

    Jacobian: 3x3 numerical FD. log T variable so temperature stays
    positive without an explicit clip.
    """
    _allocate_CD(state)
    idx = state.species.idx
    i_HI = idx['HI']
    i_HII = idx['HII']
    i_H2 = idx['H2']
    C = state.scratch['solver:C']
    D = state.scratch['solver:D']

    ncell = state.nH.shape[0]
    heat = np.empty(ncell)
    cool = np.empty(ncell)
    heat_p = np.empty(ncell)
    cool_p = np.empty(ncell)

    def F_at_state():
        network.evaluate_CD(state, C, D)
        cooling(state, heat_p, cool_p)
        F_H2 = C[i_H2] - D[i_H2] * state.x[i_H2]
        F_HII = C[i_HII] - D[i_HII] * state.x[i_HII]
        F_T = (heat_p - cool_p) / np.maximum(heat_p + cool_p, _TINY)
        return F_H2.copy(), F_HII.copy(), F_T.copy()

    f_xH2 = np.zeros(ncell)
    f_xHII = np.zeros(ncell)
    f_T = np.zeros(ncell)
    converged = np.zeros(ncell, dtype=bool)
    floor = network.x_floor

    for it in range(max_iter):
        # Snapshot.
        xH2_0 = state.x[i_H2].copy()
        xHII_0 = state.x[i_HII].copy()
        T_0 = state.T.copy()

        F_H2_0, F_HII_0, F_T_0 = F_at_state()

        # Column 1: perturb xH2.
        dxH2 = eps_J * np.maximum(np.abs(xH2_0), _TINY)
        state.x[i_H2] = xH2_0 + dxH2
        state.x[i_HI] = np.maximum(
            1.0 - 2.0 * state.x[i_H2] - xHII_0, floor)
        network.fill_ghosts(state)
        F_H2_a, F_HII_a, F_T_a = F_at_state()
        J11 = (F_H2_a - F_H2_0) / dxH2
        J21 = (F_HII_a - F_HII_0) / dxH2
        J31 = (F_T_a - F_T_0) / dxH2

        state.x[i_H2] = xH2_0

        # Column 2: perturb xHII.
        dxHII = eps_J * np.maximum(np.abs(xHII_0), _TINY)
        state.x[i_HII] = xHII_0 + dxHII
        state.x[i_HI] = np.maximum(
            1.0 - 2.0 * xH2_0 - state.x[i_HII], floor)
        network.fill_ghosts(state)
        F_H2_b, F_HII_b, F_T_b = F_at_state()
        J12 = (F_H2_b - F_H2_0) / dxHII
        J22 = (F_HII_b - F_HII_0) / dxHII
        J32 = (F_T_b - F_T_0) / dxHII

        state.x[i_HII] = xHII_0
        state.x[i_HI] = np.maximum(1.0 - 2.0 * xH2_0 - xHII_0, floor)
        network.fill_ghosts(state)

        # Column 3: perturb log T (T_new = T_0 * (1 + eps_J)).
        state.T = T_0 * (1.0 + eps_J)
        F_H2_c, F_HII_c, F_T_c = F_at_state()
        J13 = (F_H2_c - F_H2_0) / eps_J
        J23 = (F_HII_c - F_HII_0) / eps_J
        J33 = (F_T_c - F_T_0) / eps_J

        state.T = T_0

        # 3x3 Cramer solve for delta = -J^-1 F. Compute det via
        # expansion across the first row.
        det = (J11 * (J22 * J33 - J23 * J32)
               - J12 * (J21 * J33 - J23 * J31)
               + J13 * (J21 * J32 - J22 * J31))
        safe = np.abs(det) > 1.0e-60

        # b = -F. Solve via cofactor matrix / det.
        bH2 = -F_H2_0
        bHII = -F_HII_0
        bT = -F_T_0
        # Cramer for each unknown: replace column k in J with b.
        det_x1 = (bH2 * (J22 * J33 - J23 * J32)
                  - J12 * (bHII * J33 - J23 * bT)
                  + J13 * (bHII * J32 - J22 * bT))
        det_x2 = (J11 * (bHII * J33 - J23 * bT)
                  - bH2 * (J21 * J33 - J23 * J31)
                  + J13 * (J21 * bT - bHII * J31))
        det_x3 = (J11 * (J22 * bT - bHII * J32)
                  - J12 * (J21 * bT - bHII * J31)
                  + bH2 * (J21 * J32 - J22 * J31))

        dxH2_step = np.where(safe, det_x1 / det, 0.0)
        dxHII_step = np.where(safe, det_x2 / det, 0.0)
        dlogT_step = np.where(safe, det_x3 / det, 0.0)

        # Trust-region cap on log T step.
        dlogT_step = np.clip(damping * dlogT_step, -1.0, 1.0)

        state.x[i_H2] = xH2_0 + damping * dxH2_step
        state.x[i_HII] = xHII_0 + damping * dxHII_step
        state.T = T_0 * np.exp(dlogT_step)
        network.closure(state)

        f_xH2 = np.abs(state.x[i_H2] - xH2_0) / np.maximum(
            np.abs(state.x[i_H2]), _TINY)
        f_xHII = np.abs(state.x[i_HII] - xHII_0) / np.maximum(
            np.abs(state.x[i_HII]), _TINY)
        f_T = np.abs(state.T - T_0) / np.maximum(np.abs(state.T), _TINY)
        converged = (f_xH2 < tol) & (f_xHII < tol) & (f_T < tol)
        if bool(np.all(converged)):
            break

    return EqSolverResult(
        n_iter=it + 1,
        converged=converged,
        residual_inf=_residual(network, state),
        f_xH2_inf=float(np.max(f_xH2)),
        f_xHII_inf=float(np.max(f_xHII)),
        method='newton_chem_thermal',
    )


def solve_newton_log(
    network: Any,
    state: Any,
    max_iter: int = 50,
    tol: float = 1.0e-6,
    eps_J: float = 1.0e-6,
    damping: float = 1.0,
) -> EqSolverResult:
    """Newton in log-space `(log x_H2, log x_HII)`. Built-in
    positivity. Same structure as `solve_newton` but the variables
    are log-fractions.
    """
    _allocate_CD(state)
    idx = state.species.idx
    i_HI = idx['HI']
    i_HII = idx['HII']
    i_H2 = idx['H2']
    C = state.scratch['solver:C']
    D = state.scratch['solver:D']

    f_xH2 = np.zeros_like(state.x[i_H2])
    f_xHII = np.zeros_like(state.x[i_HII])
    converged = np.zeros_like(state.x[i_HII], dtype=bool)
    floor = network.x_floor

    for it in range(max_iter):
        network.evaluate_CD(state, C, D)
        F_H2_0 = C[i_H2] - D[i_H2] * state.x[i_H2]
        F_HII_0 = C[i_HII] - D[i_HII] * state.x[i_HII]
        xH2_0 = state.x[i_H2].copy()
        xHII_0 = state.x[i_HII].copy()

        # Perturb log xH2 -> xH2 * (1 + eps_J).
        state.x[i_H2] = xH2_0 * (1.0 + eps_J)
        state.x[i_HI] = np.maximum(
            1.0 - 2.0 * state.x[i_H2] - xHII_0, floor)
        network.fill_ghosts(state)
        network.evaluate_CD(state, C, D)
        F_H2_xH2 = C[i_H2] - D[i_H2] * state.x[i_H2]
        F_HII_xH2 = C[i_HII] - D[i_HII] * state.x[i_HII]
        # dF / d(log xH2) = (F_perturbed - F_0) / eps_J
        dF_H2_dLxH2 = (F_H2_xH2 - F_H2_0) / eps_J
        dF_HII_dLxH2 = (F_HII_xH2 - F_HII_0) / eps_J

        # Perturb log xHII.
        state.x[i_H2] = xH2_0
        state.x[i_HII] = xHII_0 * (1.0 + eps_J)
        state.x[i_HI] = np.maximum(
            1.0 - 2.0 * xH2_0 - state.x[i_HII], floor)
        network.fill_ghosts(state)
        network.evaluate_CD(state, C, D)
        F_H2_xHII = C[i_H2] - D[i_H2] * state.x[i_H2]
        F_HII_xHII = C[i_HII] - D[i_HII] * state.x[i_HII]
        dF_H2_dLxHII = (F_H2_xHII - F_H2_0) / eps_J
        dF_HII_dLxHII = (F_HII_xHII - F_HII_0) / eps_J

        state.x[i_HII] = xHII_0
        state.x[i_HI] = np.maximum(1.0 - 2.0 * xH2_0 - xHII_0, floor)
        network.fill_ghosts(state)

        # 2x2 Cramer for delta_log = -J^-1 F.
        det = (dF_H2_dLxH2 * dF_HII_dLxHII
               - dF_H2_dLxHII * dF_HII_dLxH2)
        safe = np.abs(det) > 1.0e-50
        dlog_xH2 = np.where(
            safe,
            -(dF_HII_dLxHII * F_H2_0 - dF_H2_dLxHII * F_HII_0) / det,
            0.0)
        dlog_xHII = np.where(
            safe,
            -(dF_H2_dLxH2 * F_HII_0 - dF_HII_dLxH2 * F_H2_0) / det,
            0.0)

        # Limit per-step log change to avoid wild swings (Newton with
        # a trust-region cap; the typical stiff-chemistry safeguard).
        dlog_max = 2.0  # at most factor of e^2 per step
        dlog_xH2 = np.clip(damping * dlog_xH2, -dlog_max, dlog_max)
        dlog_xHII = np.clip(damping * dlog_xHII, -dlog_max, dlog_max)

        state.x[i_H2] = xH2_0 * np.exp(dlog_xH2)
        state.x[i_HII] = xHII_0 * np.exp(dlog_xHII)
        network.closure(state)

        f_xH2 = np.abs(state.x[i_H2] - xH2_0) / np.maximum(
            np.abs(state.x[i_H2]), _TINY)
        f_xHII = np.abs(state.x[i_HII] - xHII_0) / np.maximum(
            np.abs(state.x[i_HII]), _TINY)
        converged = (f_xH2 < tol) & (f_xHII < tol)
        if bool(np.all(converged)):
            break

    return EqSolverResult(
        n_iter=it + 1,
        converged=converged,
        residual_inf=_residual(network, state),
        f_xH2_inf=float(np.max(f_xH2)),
        f_xHII_inf=float(np.max(f_xHII)),
        method='newton_log',
    )


# ----------------------------------------------------------------------
# Picard outer loop (QSSA-projection + 1-D Newton on xH2)
# ----------------------------------------------------------------------


def _read_zeta(state: Any, category: str, species: str,
               shape: tuple) -> Any:
    """Mirror of `pyathena.chemistry.networks.ncr3._get_rate`: pull a
    per-species photo-rate scalar / array out of the per-category dict
    on state, defaulting to zero. Avoids importing the network's
    private helper from this module."""
    cat = getattr(state, category, None)
    if not cat:
        return 0.0
    val = cat.get(species)
    if val is None:
        return 0.0
    return val


def solve_picard_xH2(
    network: Any,
    state: Any,
    max_iter: int = 30,
    tol: float = 1.0e-6,
    eps_J: float = 1.0e-6,
    damping: float = 1.0,
    log_step_clip: float = 2.0,
) -> EqSolverResult:
    """Picard outer loop with quasi-steady-state HII projection plus
    1-D Newton on xH2.

    Outer iteration:

      Step 1.  Project xHII / xCII / xe / xHI onto the analytic
               H + C ionisation equilibrium for the current xH2 via
               `eq_xHII_xe(T, nH, xH2, xi_CR, G_PE = chi_FUV, G_CI =
               chi_FUV, zeta_pi = zeta_pi['HI'])`. xHII tracks xH2
               exactly at every iteration -- removes the 2-D Newton's
               cross-coupling sign-flip overshoot.

      Step 2.  Evaluate the H2 row residual F_H2 = C_H2 - D_H2 * xH2
               via `network.evaluate_CD`.

      Step 3.  One-sided FD on F_H2(xH2_perturbed) gives df/dxH2; the
               1-D Newton step in log xH2 (trust-region clipped at
               `log_step_clip` decades) drives xH2 toward its root.

    Convergence: relative residual `|F_H2| / max(|C_H2|, |D_H2 x_H2|)`
    falling below `tol` for all cells (the H2 channel is the only
    Newton variable; xHII is at QSSA by construction so F_HII is zero
    up to the eq_xHII_xe inner tolerance).
    """
    from pyathena.chemistry.equilibrium_seed import eq_xHII_xe

    _allocate_CD(state)
    idx = state.species.idx
    i_HI = idx['HI']
    i_HII = idx['HII']
    i_H2 = idx['H2']
    i_e = idx.get('electron')
    C = state.scratch['solver:C']
    D = state.scratch['solver:D']
    floor = network.x_floor

    T_arr = np.asarray(state.T)
    nH_arr = np.asarray(state.nH)
    shape = T_arr.shape

    # Cache the radiation inputs the seed needs. The network reads the
    # same dicts at evaluate_CD time so the two stay consistent.
    xi_CR = state.xi_CR
    zeta_pi_HI = _read_zeta(state, 'zeta_pi', 'HI', shape)
    chi_FUV = state.chi_for('FUV') if (state.u_rad
                                       or state.chi.size) else 0.0

    Z_d = np.asarray(state.Z_d)
    Z_g = np.asarray(state.Z_g)

    def _project_xHII(xH2_arr):
        """Set xHII / xCII / xe / xHI on state from the analytic
        equilibrium at the supplied xH2."""
        xHII_eq, xCII_eq, xe_eq, xHI_eq = eq_xHII_xe(
            T=T_arr, nH=nH_arr,
            xH2=np.broadcast_to(xH2_arr, shape).copy(),
            xi_CR=xi_CR, G_PE=chi_FUV, G_CI=chi_FUV,
            Z_d=Z_d, Z_g=Z_g, zeta_pi=zeta_pi_HI,
        )
        state.x[i_HII] = np.maximum(xHII_eq, floor)
        state.x[i_HI] = np.maximum(xHI_eq, floor)
        if i_e is not None:
            state.x[i_e] = np.maximum(xe_eq, floor)
        # Refill metal ghosts (CII / OI / OII / CO / CI) consistently
        # with the new xHII. closure() would clamp xH2 to the floor,
        # but we want to control xH2 explicitly here, so only refill
        # the ghost rows directly.
        network.fill_ghosts(state)

    converged = np.zeros(shape, dtype=bool)
    res_H2 = np.zeros(shape)

    for it in range(max_iter):
        xH2_prev = state.x[i_H2].copy()

        # Step 1: QSSA project xHII onto its eq for current xH2.
        _project_xHII(xH2_prev)

        # Step 2: H2 residual at projected state.
        network.evaluate_CD(state, C, D)
        F_H2_0 = C[i_H2] - D[i_H2] * xH2_prev
        scale_H2 = np.maximum(
            np.maximum(np.abs(C[i_H2]),
                       np.abs(D[i_H2] * xH2_prev)),
            _TINY)
        H2_decoupled = scale_H2 < _TINY
        res_H2 = np.where(
            H2_decoupled, 0.0, np.abs(F_H2_0) / scale_H2)
        converged = res_H2 < tol
        if bool(np.all(converged)):
            break

        # Step 3: 1-D Newton step on xH2. FD perturbation re-projects
        # xHII at perturbed xH2 so the slope captures both direct and
        # via-xHII contributions to dF_H2/dxH2.
        dxH2 = eps_J * np.maximum(np.abs(xH2_prev), _TINY)
        state.x[i_H2] = xH2_prev + dxH2
        _project_xHII(state.x[i_H2])
        network.evaluate_CD(state, C, D)
        F_H2_p = C[i_H2] - D[i_H2] * state.x[i_H2]
        dF = (F_H2_p - F_H2_0) / dxH2

        # Restore xH2 + reproject at unperturbed state.
        state.x[i_H2] = xH2_prev
        _project_xHII(xH2_prev)

        # Newton step. Where the row is decoupled (hot-mask zeroes
        # C and D), leave xH2 untouched at the floor.
        safe = (~H2_decoupled) & (np.abs(dF) > _TINY)
        delta = np.where(safe, -F_H2_0 / np.where(safe, dF, 1.0), 0.0)
        # Trust-region clip on the per-iter log change so a stiff
        # cold cell can't jump xH2 by many decades and overshoot the
        # closure (xH2 < 0.5).
        xH2_new = xH2_prev + damping * delta
        # Clip to (floor, 0.5 - floor) and to a max log-change per iter.
        xH2_min = xH2_prev * np.exp(-log_step_clip)
        xH2_max = np.minimum(
            xH2_prev * np.exp(log_step_clip), 0.499999)
        xH2_new = np.clip(xH2_new,
                          np.maximum(xH2_min, floor),
                          np.maximum(xH2_max, floor))
        state.x[i_H2] = xH2_new

    # Final projection so the returned state is self-consistent.
    _project_xHII(state.x[i_H2])
    return EqSolverResult(
        n_iter=it + 1,
        converged=converged,
        residual_inf=_residual(network, state),
        f_xH2_inf=float(np.max(res_H2)),
        f_xHII_inf=0.0,  # xHII enforced at projection -> residual = 0
        method='picard_xH2',
    )
