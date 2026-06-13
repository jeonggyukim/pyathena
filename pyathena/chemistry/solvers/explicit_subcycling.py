"""Strip-first synchronous explicit subcycling solver.

`ExplicitSubcyclingSolver` integrates an NCR-style chemistry network on
a whole `(nspec, ncell)` strip in lockstep: every cell in the strip
walks the same substep length and the same substep count. This is the
synchronous semi-implicit Euler sweep pattern that the future C++
`SynchronousSemiImplicitSweep` solver also runs, so the Python solver
serves as the bit-stable regression target the C++ port can compare
against.

The substep contract mirrors `ExplicitSubcyclingSolver::DoOneSubstep`
in `tigris-ncr/src/photchem/ncr_solver.hpp:581-622` with one
difference: the C++ solver runs a per-cell `while (t_done < dt)` loop
inside `SolveCell`, taking a fresh per-cell `dt_sub` each iteration.
The Python solver collapses that to a strip-wide `while` over the
common remaining `dt_strip` and uses the strip-MIN over the per-cell
timescales for `dt_sub`. The net effect is that strips with a single
fast cell take more substeps overall than the per-cell adaptive
version; this matches the planned C++ Phase C sweep semantics, not the
current C++ ExplicitSubcyclingSolver semantics.

AthenaK porting note
--------------------

The pure-NumPy substep body in `_do_one_substep` is structurally
identical to a Kokkos `parallel_for` over the i-axis: every operation
is either a vectorised arithmetic primitive (`np.add`, `np.multiply`,
...) or a strip-wide `np.where`. None of it needs per-cell branching.
What does NOT translate one-for-one to AthenaK is

- The Python `while t_strip > 0` outer loop -- that is host-side
  control flow, expected on the AthenaK side too.
- The retry-on-rejection loop in `_attempt_substep` uses a Python
  exception-free protocol (`accepted = bool(...)` then halve `dt_sub`),
  which maps cleanly to `Kokkos::reduce` over the strip plus a
  host-side `if (!accepted) dt_sub *= 0.5`.
- The scratch dict (`state.scratch`) is keyed by string. AthenaK will
  resolve scratch by symbolic name at construction and hand the solver
  a tuple of Kokkos views; the per-strip refresh is the same.
- Floor / clamp operations use `np.maximum(..., out=...)`; the Kokkos
  analogue is `Kokkos::max`. No design change needed.

Everything inside `_do_one_substep` is branch-free over the strip
axis: the temperature gate above `temp_hot1` rides on `np.where` (no
Python `if`), and the rejection check is a strip-wide boolean reduction
(`bool(np.any(...))`).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..config import ChemistryConfig, register_solver
from ..networks.base import NetworkBase
from ..thermo.base import K_B_CGS, ThermoPolicy
from . import _substep_kernels as kern


# Default substep budget / retry parameters. The C++ side uses
# `nsub_max` from the input deck (no default); we keep a generous
# Python default so unit tests do not have to wire one in. The
# `nbad_dt_max` mirrors the C++ retry budget of 3.
_DEFAULT_NSUB_MAX: int = 1024
_NBAD_DT_MAX: int = 3
# Minimum substep length used by the rejection-and-halve protocol. The
# C++ side does not floor the substep length explicitly; here we floor
# at a numerically representable level so the retry loop cannot loop
# indefinitely on a pathological strip.
_DT_SUB_FLOOR: float = 1.0e-30


@register_solver('explicit_subcycling')
class ExplicitSubcyclingSolver:
    """Strip-first synchronous explicit subcycling solver.

    Parameters
    ----------
    config : ChemistryConfig
        Source of the substep CFL, the hot/cold transition temperature
        `temp_hot1`, and the substep budget `nsub_max`.
    network : NetworkBase
        The chemistry network whose `evaluate_CD` produces the
        substep (C, D) split. The solver expects an HI/HII/H2 network
        (e.g. `NCRNetwork3`) with `evolved == ('HI', 'HII', 'H2')`.
    thermo : ThermoPolicy
        Provides the mu / temperature couplings used by the
        semi-implicit T step.
    cooling : optional
        Phase 4 cooling policy. If absent, the solver leaves the
        temperature untouched and only advances the chemistry.

    Notes
    -----
    The solver keeps no per-cell `dt_remaining`; the strip-MIN
    convention amounts to a single scalar dt_remaining shared across
    cells. The C++ Phase C `SynchronousSemiImplicitSweep` will mirror
    this exactly.
    """

    __version__: str = '0.1'

    def __init__(
        self,
        config: ChemistryConfig,
        network: NetworkBase,
        thermo: ThermoPolicy,
        cooling: Any = None,
    ) -> None:
        self.config = config
        self.network = network
        self.thermo = thermo
        self.cooling = cooling
        # Knobs read out of config for the hot path. Stash as attributes
        # so the substep body does not chase the config object.
        params = config.solver.params if config.solver else {}
        self.cfl_cool_sub: float = float(
            params.get('cfl_cool_sub', config.cfl_cool_sub))
        self.nsub_max: int = int(
            params.get('nsub_max', max(config.nsub_max, _DEFAULT_NSUB_MAX)))
        self.temp_hot1: float = float(config.temp_hot1)
        # Pre-cache the species row indices the solver touches. These
        # are written by `allocate_scratch` once the state's species
        # set is known.
        self._i_HI: int = -1
        self._i_HII: int = -1
        self._i_H2: int = -1

    # ------------------------------------------------------------------
    # Scratch buffer wiring
    # ------------------------------------------------------------------
    def allocate_scratch(self, state: Any) -> None:
        """Register all hot-path scratch buffers on `state`.

        Called once at driver setup time. Subsequent calls re-register
        the same buffers (idempotent) so a driver that rebuilds the
        state can call this again without leaking memory.
        """
        ncell = state.nH.shape[0]
        nspec = state.x.shape[0]
        # Solver-owned scratch is namespaced under `solver:`.
        state.alloc_scratch('solver:C', (nspec, ncell))
        state.alloc_scratch('solver:D', (nspec, ncell))
        state.alloc_scratch('solver:tmp_ncell', (ncell,))
        state.alloc_scratch('solver:tmp_ncell_b', (ncell,))
        state.alloc_scratch('solver:T_old', (ncell,))
        state.alloc_scratch('solver:T_new', (ncell,))
        state.alloc_scratch('solver:net_cool', (ncell,))
        state.alloc_scratch('solver:d_net_cool_dT', (ncell,))
        state.alloc_scratch('solver:inv_heat_capacity', (ncell,))
        # Single-row scratch for the implicit-Euler chemistry update.
        # Three buffers cover HI, HII, H2 substep updates without
        # cross-row aliasing.
        state.alloc_scratch('solver:x_new_HI', (ncell,))
        state.alloc_scratch('solver:x_new_HII', (ncell,))
        state.alloc_scratch('solver:x_new_H2', (ncell,))
        # Boolean scratch for the temperature-update rejection mask
        # and the H2 hot/cold gate. Two slots so the two consumers can
        # use them independently without aliasing.
        state.alloc_scratch('solver:reject_mask', (ncell,), dtype=np.bool_)
        state.alloc_scratch('solver:mask_b1', (ncell,), dtype=np.bool_)
        state.alloc_scratch('solver:mask_b2', (ncell,), dtype=np.bool_)
        # Pre-resolve evolved-row indices. NCRNetwork3 uses
        # `('HI', 'HII', 'H2')` at indices (0, 1, 2) in the
        # ncr3_with_ghosts layout.
        idx = state.species.idx
        self._i_HI = idx['HI']
        self._i_HII = idx['HII']
        self._i_H2 = idx['H2']

    # ------------------------------------------------------------------
    # Outer driver loop
    # ------------------------------------------------------------------
    def step(self, dt: float, state: Any) -> int:
        """Advance the strip over `[0, dt]`.

        Returns the total number of substeps used. The implementation
        keeps the substep count strip-wide (one count per strip, not
        per cell) -- every cell walks the same nsub.
        """
        t_remaining = float(dt)
        nsub_used = 0
        while t_remaining > 0.0 and nsub_used < self.nsub_max:
            dt_sub = self._do_one_substep(state, t_remaining)
            t_remaining -= dt_sub
            nsub_used += 1
            state.diag.n_substeps_total += 1
            if nsub_used > state.diag.nsub_max_seen:
                state.diag.nsub_max_seen = nsub_used
            if dt_sub <= _DT_SUB_FLOOR:
                # Pathological strip; bail to avoid spinning forever.
                break
        if nsub_used >= self.nsub_max and t_remaining > 0.0:
            state.diag.n_strips_capped += 1
        return nsub_used

    # ------------------------------------------------------------------
    # Per-substep core
    # ------------------------------------------------------------------
    def _do_one_substep(self, state: Any, dt_remaining: float) -> float:
        """One synchronous semi-implicit Euler substep over the strip.

        Returns the substep length actually taken. The substep length
        is initially `cfl_cool_sub * min_strip(t_cool, t_chem_HII,
        t_chem_H2)` capped at `dt_remaining`; if the implied
        temperature step is rejected for any cell, the substep is
        halved on the whole strip and retried up to `_NBAD_DT_MAX`
        times.
        """
        C = state.get_scratch('solver:C')
        D = state.get_scratch('solver:D')
        T_old = state.get_scratch('solver:T_old')
        tmp_ncell = state.get_scratch('solver:tmp_ncell')

        # Snapshot T at substep entry so the rejection-and-halve loop
        # restores it on retry.
        np.copyto(T_old, state.T)

        # Step 1: evaluate (C, D) at the entry state.
        self.network.evaluate_CD(state, C, D)
        # Step 2: compute cooling rates / derivative at the entry state.
        self._evaluate_cooling(state)
        # Step 3: pick the strip-MIN dt_sub.
        dt_sub = self._estimate_dt_sub(state, C, D, dt_remaining,
                                       tmp_ncell)

        # Step 4: try the substep; halve on rejection until accepted or
        # the budget is exhausted.
        for _retry in range(_NBAD_DT_MAX + 1):
            accepted = self._attempt_T_step(state, T_old, dt_sub)
            state.diag.n_thermal_solves += 1
            if accepted:
                break
            # Restore T to the entry value before retrying.
            np.copyto(state.T, T_old)
            dt_sub *= 0.5
            if dt_sub < _DT_SUB_FLOOR:
                break
        else:
            state.diag.n_nan_traps += 1

        # Apply the implicit-Euler chemistry update + closure at the
        # accepted dt_sub. Rates are recomputed at the post-T state.
        self.network.evaluate_CD(state, C, D)
        self._update_chemistry(state, C, D, dt_sub)
        self.network.closure(state)
        return dt_sub

    # ------------------------------------------------------------------
    # Substep ingredients
    # ------------------------------------------------------------------
    def _attempt_T_step(
        self,
        state: Any,
        T_old: np.ndarray,
        dt_sub: float,
    ) -> bool:
        """Try the semi-implicit T step at `dt_sub`.

        Returns True if every cell accepts (T_new > 0, finite, within
        a factor of `1 + 2 * cfl_cool_sub` of T_old). On rejection the
        caller restores `state.T` and halves `dt_sub`.
        """
        net_cool = state.get_scratch('solver:net_cool')
        d_net_cool_dT = state.get_scratch('solver:d_net_cool_dT')
        inv_heat_capacity = state.get_scratch('solver:inv_heat_capacity')
        tmp_ncell = state.get_scratch('solver:tmp_ncell')
        T_new = state.get_scratch('solver:T_new')

        kern.semi_implicit_T_update(
            T_old, net_cool, d_net_cool_dT, inv_heat_capacity, dt_sub,
            out=T_new, tmp=tmp_ncell,
        )

        # Physical-range check, branch-free over the strip. The
        # rejection mask is stored in pre-allocated scratch.
        reject_mask = state.get_scratch('solver:reject_mask')
        viol_mask = state.get_scratch('solver:mask_b1')
        cfl = self.cfl_cool_sub
        # reject = ~isfinite(T_new) | (T_new <= 0)
        #        | (T_new > (1 + 2 cfl) * T_old)
        #        | (T_new < (1 - 2 cfl) * T_old)
        np.isfinite(T_new, out=reject_mask)
        np.logical_not(reject_mask, out=reject_mask)
        # viol_mask = T_new <= 0
        np.less_equal(T_new, 0.0, out=viol_mask)
        np.logical_or(reject_mask, viol_mask, out=reject_mask)
        # tmp_ncell = (1 + 2 cfl) * T_old; viol_mask = T_new > tmp_ncell
        np.multiply(T_old, 1.0 + 2.0 * cfl, out=tmp_ncell)
        np.greater(T_new, tmp_ncell, out=viol_mask)
        np.logical_or(reject_mask, viol_mask, out=reject_mask)
        # tmp_ncell = (1 - 2 cfl) * T_old; viol_mask = T_new < tmp_ncell
        np.multiply(T_old, 1.0 - 2.0 * cfl, out=tmp_ncell)
        np.less(T_new, tmp_ncell, out=viol_mask)
        np.logical_or(reject_mask, viol_mask, out=reject_mask)
        if bool(np.any(reject_mask)):
            return False
        # Accepted: write T_new back into state.T.
        np.copyto(state.T, T_new)
        return True

    def _estimate_dt_sub(
        self,
        state: Any,
        C: np.ndarray,
        D: np.ndarray,
        dt_remaining: float,
        scratch: np.ndarray,
    ) -> float:
        """Strip-MIN substep length.

        Three timescales contribute: the cooling timescale
        `e / |cool - heat|`, the HII chemistry timescale
        `1 / |xHI * C_HII_per_xHI - xHII * D_HII| = 1 / |C[HII] - xHII * D_HII|`
        (because the network already premultiplies C_HII_per_xHI by
        xHI), and (when T < temp_hot1) the same for H2. The strip
        takes the MIN over all cells and channels, then multiplies by
        `cfl_cool_sub`.
        """
        i_HII = self._i_HII
        i_H2 = self._i_H2
        tiny = 1.0e-300

        xHII = state.x[i_HII]
        xH2 = state.x[i_H2]
        # |C[HII] - xHII * D[HII]|
        np.multiply(xHII, D[i_HII], out=scratch)
        np.subtract(C[i_HII], scratch, out=scratch)
        np.fabs(scratch, out=scratch)
        inv_t_chem_hii_max = float(np.max(scratch)) + tiny

        # |C[H2] - xH2 * D[H2]| in the cold gas. Above temp_hot1 the
        # network writes zeros into both C and D for H2, so the
        # difference is zero; explicit gating below is belt-and-braces.
        np.multiply(xH2, D[i_H2], out=scratch)
        np.subtract(C[i_H2], scratch, out=scratch)
        np.fabs(scratch, out=scratch)
        # Gate above temp_hot1: write 0 in place using a 0/1 mask
        # because np.where does not have an `out=` overload. The
        # boolean mask broadcasts to 0/1 inside np.multiply.
        gate = state.get_scratch('solver:mask_b2')
        np.less(state.T, self.temp_hot1, out=gate)
        np.multiply(scratch, gate, out=scratch)
        inv_t_chem_h2_max = float(np.max(scratch))

        # Cooling timescale.
        net_cool = state.get_scratch('solver:net_cool')
        inv_heat_capacity = state.get_scratch('solver:inv_heat_capacity')
        # 1 / t_cool = inv_heat_capacity * |net_cool| / T
        np.multiply(inv_heat_capacity, net_cool, out=scratch)
        np.divide(scratch, state.T, out=scratch)
        np.fabs(scratch, out=scratch)
        inv_t_cool_max = float(np.max(scratch))

        inv_t_max = max(inv_t_chem_hii_max,
                        inv_t_chem_h2_max,
                        inv_t_cool_max,
                        tiny)
        dt_sub = self.cfl_cool_sub / inv_t_max
        if dt_sub > dt_remaining:
            dt_sub = dt_remaining
        return dt_sub

    def _evaluate_cooling(self, state: Any) -> None:
        """Populate `net_cool`, `d_net_cool_dT`, `inv_heat_capacity`.

        The Phase 3 driver does not yet have a cooling policy plumbed
        in (Phase 4 wires that up). Without one, `net_cool` and
        `d_net_cool_dT` are zero -- the temperature stays put and only
        the chemistry advances. `inv_heat_capacity` is still computed
        because the dt_sub estimator divides by T through it.

        When `self.cooling` is supplied, the policy is expected to
        write `solver:net_cool` and `solver:d_net_cool_dT` in place
        on `state.scratch`. The Phase 4 implementation follows this
        contract.
        """
        net_cool = state.get_scratch('solver:net_cool')
        d_net_cool_dT = state.get_scratch('solver:d_net_cool_dT')
        inv_heat_capacity = state.get_scratch('solver:inv_heat_capacity')

        if self.cooling is None:
            net_cool[:] = 0.0
            d_net_cool_dT[:] = 0.0
        else:
            self.cooling.update(state)

        # inv_heat_capacity = (gamma - 1) / (n_tot * k_B) per cell.
        # n_tot = n_H * (1 + A_He - x_H2 + x_e); recover it from mu
        # via mu = mu_hyd / (1 + A_He - x_H2 + x_e). So
        # n_tot / n_H = mu_hyd / mu, hence inv_heat_capacity
        #             = (gamma - 1) * mu / (n_H * mu_hyd * k_B).
        self.thermo.mu(state, inv_heat_capacity)
        mu_hyd = self.thermo.mu_hyd
        gamma_minus_one = self.thermo.gamma - 1.0
        np.multiply(inv_heat_capacity, gamma_minus_one,
                    out=inv_heat_capacity)
        np.divide(inv_heat_capacity, state.nH, out=inv_heat_capacity)
        np.divide(inv_heat_capacity, mu_hyd * K_B_CGS,
                  out=inv_heat_capacity)

    def _update_chemistry(
        self,
        state: Any,
        C: np.ndarray,
        D: np.ndarray,
        dt_sub: float,
    ) -> None:
        """Apply the implicit-Euler chemistry update to evolved rows.

        Writes `state.x[evolved_idx]` in place using per-row scratch
        buffers so the update is fancy-index-free (which keeps the
        hot path under `assert_no_alloc`).
        """
        i_HI = self._i_HI
        i_HII = self._i_HII
        i_H2 = self._i_H2

        tmp_ncell = state.get_scratch('solver:tmp_ncell')
        x_new_HI = state.get_scratch('solver:x_new_HI')
        x_new_HII = state.get_scratch('solver:x_new_HII')
        x_new_H2 = state.get_scratch('solver:x_new_H2')

        # HI row
        kern.implicit_euler_update(
            state.x[i_HI], C[i_HI], D[i_HI], dt_sub,
            out=x_new_HI, tmp=tmp_ncell,
        )
        # HII row
        kern.implicit_euler_update(
            state.x[i_HII], C[i_HII], D[i_HII], dt_sub,
            out=x_new_HII, tmp=tmp_ncell,
        )
        # H2 row
        kern.implicit_euler_update(
            state.x[i_H2], C[i_H2], D[i_H2], dt_sub,
            out=x_new_H2, tmp=tmp_ncell,
        )

        # Write back. Single-row assignment is a view-targeted write
        # via numpy's `__setitem__`, which does not allocate.
        np.copyto(state.x[i_HI], x_new_HI)
        np.copyto(state.x[i_HII], x_new_HII)
        np.copyto(state.x[i_H2], x_new_H2)
