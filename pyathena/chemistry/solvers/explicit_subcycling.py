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
# Minimum substep length [seconds]. The substep retry loop halves
# dt_sub on rejection up to `_NBAD_DT_MAX` times; the floor bails out
# of the outer `step()` loop if dt_sub falls below it, so the strip
# cannot spin forever on a pathological cell. Set well below any
# physical chemistry timescale (the fastest collisional rate at
# n = 1e20 cm^-3 is ~1e14 s^-1, dt ~ 1e-14 s; we floor a decade below
# that).
_DT_SUB_FLOOR: float = 1.0e-15
# Species fraction floor. Matches `X_FLOOR` in `networks/ncr3.py` and
# `TINY_NUMBER` on the C++ side (`src/photchem/defs.hpp`).
_TINY_NUMBER: float = 1.0e-20


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
        # Post-step adaptive-control thresholds (Zier+ 2024 §4.1.1).
        # `f_chem_cap`: reject the substep if observed max fractional
        # change in T or species exceeds this.
        # `f_chem_target`: forward controller targets this as the
        # average per-step fractional change via
        # `dt_next = dt * min(2, f_chem_target / f_observed)`.
        self.f_chem_cap: float = float(
            params.get('f_chem_cap', config.f_chem_cap))
        self.f_chem_target: float = float(
            params.get('f_chem_target', config.f_chem_target))
        # Forward-controller carry-over between substeps and between
        # successive `step()` calls. Initial value `inf` lets the first
        # substep fall back to the pre-step `_estimate_dt_sub` formula.
        self._dt_sub_next: float = float('inf')
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
        # Substep thermodynamic state. The conserved variable across the
        # substep is `temp_mu = T / mu`; see kernels::semi_implicit_temp_mu_update.
        # `solver:mu_at_entry` holds mu evaluated at substep entry so the
        # post-chemistry rescale state.T = mu_new * temp_mu_new can keep
        # the substep-invariant T/mu correct when species evolve.
        state.alloc_scratch('solver:temp_mu_old', (ncell,))
        state.alloc_scratch('solver:temp_mu_new', (ncell,))
        state.alloc_scratch('solver:mu_at_entry', (ncell,))
        state.alloc_scratch('solver:net_cool', (ncell,))
        state.alloc_scratch('solver:d_net_cool_d_temp_mu', (ncell,))
        state.alloc_scratch('solver:inv_heat_cap_per_temp_mu', (ncell,))
        # Single-row scratch for the implicit-Euler chemistry update.
        # Three buffers cover HI, HII, H2 substep updates without
        # cross-row aliasing.
        state.alloc_scratch('solver:x_new_HI', (ncell,))
        state.alloc_scratch('solver:x_new_HII', (ncell,))
        state.alloc_scratch('solver:x_new_H2', (ncell,))
        # Substep-entry snapshot for the post-step f-rule reject branch
        # (Zier+ 2024 §4.1.1). On x-cap rejection the chemistry rows
        # are restored from these buffers and the substep retried at a
        # smaller dt. `solver:f_denom` holds the floored denominator
        # `max(x_pre, x_floor)` used to compute the relative change.
        state.alloc_scratch('solver:xHI_pre', (ncell,))
        state.alloc_scratch('solver:xHII_pre', (ncell,))
        state.alloc_scratch('solver:xH2_pre', (ncell,))
        state.alloc_scratch('solver:f_denom', (ncell,))
        # 2x2 Cramer scratch for the joint (x_H2, x_HII) BE solve.
        # Ports `ncr_solver.hpp::UpdateChemistry`: lines 521-528 build
        # six scalars (a, b, c, d, e, f); we hold the strip-wide arrays
        # plus the determinant.
        state.alloc_scratch('solver:cramer_a', (ncell,))
        state.alloc_scratch('solver:cramer_b', (ncell,))
        state.alloc_scratch('solver:cramer_c', (ncell,))
        state.alloc_scratch('solver:cramer_d', (ncell,))
        state.alloc_scratch('solver:cramer_e', (ncell,))
        state.alloc_scratch('solver:cramer_f', (ncell,))
        state.alloc_scratch('solver:cramer_det', (ncell,))
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

        Substep length comes from either the pre-step formula
        `cfl_cool_sub / max |C - D x|` or the forward controller
        carry-over `self._dt_sub_next`, whichever is smaller, capped
        at `dt_remaining`. After the T + chemistry update, the post-
        step max fractional change `f = max(|delta T|/T,
        |delta x_i|/x_i)` is checked: if `f > f_chem_cap` the substep
        is rejected, the species and T are restored from the entry
        snapshot, dt is halved, and the substep retried up to
        `_NBAD_DT_MAX` times. On accept the forward controller stores
        `dt_sub * min(2, f_chem_target / f)` for the next substep.
        """
        i_HI = self._i_HI
        i_HII = self._i_HII
        i_H2 = self._i_H2

        C = state.get_scratch('solver:C')
        D = state.get_scratch('solver:D')
        T_entry = state.get_scratch('solver:tmp_ncell_b')
        temp_mu_old = state.get_scratch('solver:temp_mu_old')
        temp_mu_new = state.get_scratch('solver:temp_mu_new')
        mu_at_entry = state.get_scratch('solver:mu_at_entry')
        tmp_ncell = state.get_scratch('solver:tmp_ncell')
        xHI_pre = state.get_scratch('solver:xHI_pre')
        xHII_pre = state.get_scratch('solver:xHII_pre')
        xH2_pre = state.get_scratch('solver:xH2_pre')

        # Substep-entry snapshots. `T_entry` and `xX_pre` are
        # restored on rejection. `mu_at_entry` is captured here once
        # because the cooling step (re-)computes `temp_mu_old` from
        # `T_entry / mu_at_entry` on every retry.
        np.copyto(T_entry, state.T)
        self.thermo.mu(state, mu_at_entry)
        np.divide(T_entry, mu_at_entry, out=temp_mu_old)
        np.copyto(xHI_pre, state.x[i_HI])
        np.copyto(xHII_pre, state.x[i_HII])
        np.copyto(xH2_pre, state.x[i_H2])

        # (C, D) and cooling at the entry state for the pre-step dt
        # estimate; recomputed at the post-T state inside the retry
        # loop for the chemistry solve.
        self.network.evaluate_CD(state, C, D)
        self._evaluate_cooling(state)
        dt_sub_pre = self._estimate_dt_sub(state, C, D, dt_remaining,
                                           tmp_ncell)
        # Forward-controller cap: never exceed
        # `self._dt_sub_next` carried over from the previous substep.
        # Also re-cap at `dt_remaining` because the controller does
        # not know about the per-call remainder.
        dt_sub = min(dt_sub_pre, self._dt_sub_next, dt_remaining)

        f_observed = 0.0
        accepted = False
        for _retry in range(_NBAD_DT_MAX + 1):
            # T step.
            t_step_ok = self._attempt_temp_mu_step(
                state, temp_mu_old, mu_at_entry, dt_sub,
            )
            state.diag.n_thermal_solves += 1
            if not t_step_ok:
                # T-cap inside the kernel rejected; halve and retry.
                np.copyto(state.T, T_entry)
                dt_sub *= 0.5
                if dt_sub < _DT_SUB_FLOOR:
                    break
                continue
            # Chemistry step at the post-T rate coefficients.
            self.network.evaluate_CD(state, C, D)
            self._update_chemistry(state, C, D, dt_sub)
            self.network.closure(state)

            # Post-step fractional change `f`.
            f_observed = self._max_frac_change(
                state, temp_mu_old, temp_mu_new,
                xHI_pre, xHII_pre, xH2_pre,
            )
            if f_observed <= self.f_chem_cap:
                accepted = True
                break
            # x-cap rejection. Restore both T and chemistry from
            # snapshot, halve dt, retry.
            np.copyto(state.T, T_entry)
            np.copyto(state.x[i_HI], xHI_pre)
            np.copyto(state.x[i_HII], xHII_pre)
            np.copyto(state.x[i_H2], xH2_pre)
            dt_sub *= 0.5
            if dt_sub < _DT_SUB_FLOOR:
                break
        if not accepted:
            state.diag.n_nan_traps += 1

        # Forward controller: target `f_chem_target` average change,
        # capped at 2x growth per substep (matches the AREPO-RT
        # default in Zier+ 2024).
        eps = 1.0e-30
        factor = self.f_chem_target / max(f_observed, eps)
        factor = min(factor, 2.0)
        self._dt_sub_next = dt_sub * factor

        # Post-chemistry rescale: temp_mu is the substep-invariant
        # variable. Cooling fixed it at temp_mu_new; chemistry then
        # changed mu without touching temp_mu. The hydro-facing T must
        # therefore become mu_new * temp_mu_new so that pressure (=
        # den * temp_mu_new / temp_mu_cgs in code units) is consistent
        # with the operator-splitting energy accounting. Without this
        # rescale, T would silently track mu_at_entry and the gas would
        # gain or lose internal energy whenever species ionised /
        # recombined inside a single substep. tigris-ncr
        # (ncr_solver.hpp::UpdateTemperature) and mini-RAMSES
        # (cooling_module::cool_step) follow the same convention.
        mu_new = state.get_scratch('solver:tmp_ncell')
        self.thermo.mu(state, mu_new)
        np.multiply(mu_new, temp_mu_new, out=state.T)
        return dt_sub

    def _max_frac_change(
        self,
        state: Any,
        temp_mu_old: np.ndarray,
        temp_mu_new: np.ndarray,
        xHI_pre: np.ndarray,
        xHII_pre: np.ndarray,
        xH2_pre: np.ndarray,
    ) -> float:
        """Max fractional change in the substep.

        For hydrogen species the metric is the ABSOLUTE change
        `max(|delta x_HI|, |delta x_HII|, |delta(2 x_H2)|)`. Since
        `x_HI + x_HII + 2 x_H2 = 1` exactly, each term is naturally
        in `[0, 1]` and an absolute cap of 0.1 means "no more than
        10 percent of the total H budget moves between species per
        substep". Relative-change metrics would over-constrain dt
        for trace species (e.g. x_HII at 1e-12 in cold molecular
        gas).

        For thermodynamics the relative change is measured on the
        substep-invariant `temp_mu = T / mu`, not on `T` directly:
        chemistry inside one substep changes `mu` (e.g. ionising H
        bumps mu downward by ~ x_HII) without doing any work on the
        gas, so `delta T` at fixed `temp_mu` is a bookkeeping
        artefact, not a real thermodynamic change. `temp_mu` is the
        variable the cooling kernel evolves, and it isolates the
        actual energy / cooling step.
        """
        tmp = state.get_scratch('solver:tmp_ncell')

        # f_T = max |temp_mu_new - temp_mu_old| / temp_mu_old.
        np.subtract(temp_mu_new, temp_mu_old, out=tmp)
        np.fabs(tmp, out=tmp)
        np.divide(tmp, temp_mu_old, out=tmp)
        f = float(np.max(tmp))

        # f_HI = max |delta x_HI|.
        np.subtract(state.x[self._i_HI], xHI_pre, out=tmp)
        np.fabs(tmp, out=tmp)
        f = max(f, float(np.max(tmp)))

        # f_HII = max |delta x_HII|.
        np.subtract(state.x[self._i_HII], xHII_pre, out=tmp)
        np.fabs(tmp, out=tmp)
        f = max(f, float(np.max(tmp)))

        # f_H2 = max |delta (2 x_H2)|.
        np.subtract(state.x[self._i_H2], xH2_pre, out=tmp)
        np.fabs(tmp, out=tmp)
        np.multiply(tmp, 2.0, out=tmp)
        f = max(f, float(np.max(tmp)))
        return f

    # ------------------------------------------------------------------
    # Substep ingredients
    # ------------------------------------------------------------------
    def _attempt_temp_mu_step(
        self,
        state: Any,
        temp_mu_old: np.ndarray,
        mu_at_entry: np.ndarray,
        dt_sub: float,
    ) -> bool:
        """Try the semi-implicit temp_mu step at `dt_sub`.

        Returns True if every cell accepts (temp_mu_new > 0, finite,
        within a factor of `1 + 2 * cfl_cool_sub` of temp_mu_old). On
        rejection the caller restores `state.T` from the substep-entry
        snapshot and halves `dt_sub`. The conserved variable is
        `temp_mu = T / mu`; with `mu` held fixed across the cooling
        sub-step, the bound on `temp_mu_new / temp_mu_old` is the same
        as a bound on `T_new / T_old` (the ratio `mu` cancels). On
        accept the kernel result is written through to `state.T` as
        `mu_at_entry * temp_mu_new` so the chemistry sub-step sees the
        correct post-cooling temperature for its rate coefficients.
        """
        net_cool = state.get_scratch('solver:net_cool')
        d_net_cool_d_temp_mu = state.get_scratch(
            'solver:d_net_cool_d_temp_mu')
        inv_heat_cap_per_temp_mu = state.get_scratch(
            'solver:inv_heat_cap_per_temp_mu')
        tmp_ncell = state.get_scratch('solver:tmp_ncell')
        temp_mu_new = state.get_scratch('solver:temp_mu_new')

        kern.semi_implicit_temp_mu_update(
            temp_mu_old, net_cool, d_net_cool_d_temp_mu,
            inv_heat_cap_per_temp_mu, dt_sub,
            out=temp_mu_new, tmp=tmp_ncell,
        )

        # Physical-range check, branch-free over the strip. The
        # rejection mask is stored in pre-allocated scratch.
        reject_mask = state.get_scratch('solver:reject_mask')
        viol_mask = state.get_scratch('solver:mask_b1')
        cfl = self.cfl_cool_sub
        # reject = ~isfinite(temp_mu_new) | (temp_mu_new <= 0)
        #        | (temp_mu_new > (1 + 2 cfl) * temp_mu_old)
        #        | (temp_mu_new < (1 - 2 cfl) * temp_mu_old)
        np.isfinite(temp_mu_new, out=reject_mask)
        np.logical_not(reject_mask, out=reject_mask)
        np.less_equal(temp_mu_new, 0.0, out=viol_mask)
        np.logical_or(reject_mask, viol_mask, out=reject_mask)
        np.multiply(temp_mu_old, 1.0 + 2.0 * cfl, out=tmp_ncell)
        np.greater(temp_mu_new, tmp_ncell, out=viol_mask)
        np.logical_or(reject_mask, viol_mask, out=reject_mask)
        np.multiply(temp_mu_old, 1.0 - 2.0 * cfl, out=tmp_ncell)
        np.less(temp_mu_new, tmp_ncell, out=viol_mask)
        np.logical_or(reject_mask, viol_mask, out=reject_mask)
        if bool(np.any(reject_mask)):
            return False
        # Accepted: write T_new = mu_at_entry * temp_mu_new back into
        # state.T. mu has not changed yet (chemistry runs next).
        np.multiply(mu_at_entry, temp_mu_new, out=state.T)
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

        # Cooling timescale. The natural form on the substep-invariant
        # variable temp_mu = T / mu is
        #     1 / t_cool = inv_heat_cap_per_temp_mu * |net_cool| / temp_mu
        # which is mu-independent (both factors are). It is identical
        # to the T-space form |net_cool| / u_int because both reduce to
        # |net_cool| * (gamma - 1) / (n_H * mu_hyd * k_B * temp_mu).
        net_cool = state.get_scratch('solver:net_cool')
        inv_heat_cap_per_temp_mu = state.get_scratch(
            'solver:inv_heat_cap_per_temp_mu')
        temp_mu_old = state.get_scratch('solver:temp_mu_old')
        np.multiply(inv_heat_cap_per_temp_mu, net_cool, out=scratch)
        np.divide(scratch, temp_mu_old, out=scratch)
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
        """Populate `net_cool`, `d_net_cool_d_temp_mu`,
        `inv_heat_cap_per_temp_mu`.

        The Phase 3 driver does not yet have a cooling policy plumbed
        in (Phase 4 wires that up). Without one, `net_cool` and
        `d_net_cool_d_temp_mu` are zero -- temp_mu stays put and only
        the chemistry advances. `inv_heat_cap_per_temp_mu` is still
        computed because the dt_sub estimator reads it.

        Cooling policy contract (Phase 4): the policy writes
        `solver:net_cool` (= cool - heat per cell, erg / s / cm^3) and
        `solver:d_net_cool_d_temp_mu` (= d(cool - heat) / d(T/mu) per
        cell). The temp_mu-space derivative is `mu * d(net_cool) / dT`
        because temp_mu = T / mu; concrete policies that only know
        `d(net_cool) / dT` should scale by mu before writing.
        """
        net_cool = state.get_scratch('solver:net_cool')
        d_net_cool_d_temp_mu = state.get_scratch(
            'solver:d_net_cool_d_temp_mu')
        inv_heat_cap_per_temp_mu = state.get_scratch(
            'solver:inv_heat_cap_per_temp_mu')

        if self.cooling is None:
            net_cool[:] = 0.0
            d_net_cool_d_temp_mu[:] = 0.0
        else:
            self.cooling.update(state)

        # inv_heat_cap_per_temp_mu = (gamma - 1) / (n_H * mu_hyd * k_B)
        # per cell. This is mu-INDEPENDENT: it converts an erg / s / cm^3
        # net cooling rate into a K / s rate of temp_mu = T / mu.
        # Derivation: the gas internal energy density u_int satisfies
        # u_int = n_tot * k_B * T / (gamma - 1) and n_tot = n_H * (1 +
        # A_He - x_H2 + x_e) = n_H * mu_hyd / mu. Hence
        # d(temp_mu)/dt = d(T/mu)/dt = (1/mu) * dT/dt
        #               = (gamma - 1) / (n_H * mu_hyd * k_B) * (heat -
        #                 cool)
        # so the mu factor cancels: the only species inputs that survive
        # are n_H and mu_hyd, both constant on the substep timescale.
        # This matches tigris-ncr `ncr_solver.hpp` (it_heat / it_cool
        # computed against `igm1 * press` rather than `n_tot * k_B`,
        # which is the same expression with the press = n_tot k_B T
        # substitution undone).
        mu_hyd = self.thermo.mu_hyd
        gamma_minus_one = self.thermo.gamma - 1.0
        np.divide(gamma_minus_one, state.nH, out=inv_heat_cap_per_temp_mu)
        np.divide(inv_heat_cap_per_temp_mu, mu_hyd * K_B_CGS,
                  out=inv_heat_cap_per_temp_mu)

    def _update_chemistry(
        self,
        state: Any,
        C: np.ndarray,
        D: np.ndarray,
        dt_sub: float,
    ) -> None:
        """Apply the implicit-Euler chemistry update to evolved rows.

        Ports `UpdateChemistry` from
        `tigris-ncr/src/photchem/ncr_solver.hpp:513-571`: solves a
        2x2 BE system for `(x_H2, x_HII)` with hydrogen conservation
        `x_HI = 1 - 2 x_H2 - x_HII` substituted into the source
        terms, then derives `x_HI` from closure with RAMSES-style
        clipping. Three separate semi-implicit row updates on
        `(x_HI, x_HII, x_H2)` would NOT conserve hydrogen because the
        HII / H2 rows see a stale `x_HI` at substep entry and the HI
        row sees a stale `x_HII / x_H2`; the joint Cramer solve here
        is the conservation-respecting form.

        NCR3-specific: the kernel assumes the evolved set is
        `(x_HI, x_HII, x_H2)` with `x_HI` the closure species.
        Multi-ion networks will dispatch to a per-network
        `network.closure_substep(state, C, D, dt_sub)` hook.
        """
        i_HI = self._i_HI
        i_HII = self._i_HII
        i_H2 = self._i_H2

        xHI = state.x[i_HI]
        xH2 = state.x[i_H2]
        xHII = state.x[i_HII]

        # Recover the per-x_HI source rates. The network premultiplies
        # `C[i_H2]  = c_h2  * x_HI` and `C[i_HII] = c_hii * x_HI`
        # (see `NCRNetwork3.evaluate_CD`, lines 371, 375). Recover the
        # per-x_HI factors by dividing; the floor on x_HI guards the
        # HII-region case where x_HI -> 1e-4 and C also -> 0.
        tmp_ncell = state.get_scratch('solver:tmp_ncell')
        np.maximum(xHI, _TINY_NUMBER, out=tmp_ncell)
        c_h2 = state.get_scratch('solver:cramer_e')   # reuse: e overwrites c_h2
        c_hii = state.get_scratch('solver:cramer_f')  # reuse: f overwrites c_hii
        np.divide(C[i_H2],  tmp_ncell, out=c_h2)
        np.divide(C[i_HII], tmp_ncell, out=c_hii)

        a = state.get_scratch('solver:cramer_a')
        b = state.get_scratch('solver:cramer_b')
        cc = state.get_scratch('solver:cramer_c')
        dd = state.get_scratch('solver:cramer_d')
        e = state.get_scratch('solver:cramer_e')
        f = state.get_scratch('solver:cramer_f')
        det = state.get_scratch('solver:cramer_det')

        # a = 1 + (2 c_h2 + D_H2) * dt
        np.multiply(c_h2, 2.0, out=a)
        np.add(a, D[i_H2], out=a)
        np.multiply(a, dt_sub, out=a)
        np.add(a, 1.0, out=a)

        # b = c_h2 * dt
        np.multiply(c_h2, dt_sub, out=b)

        # cc = 2 c_hii * dt
        np.multiply(c_hii, 2.0 * dt_sub, out=cc)

        # dd = 1 + (c_hii + D_HII) * dt
        np.add(c_hii, D[i_HII], out=dd)
        np.multiply(dd, dt_sub, out=dd)
        np.add(dd, 1.0, out=dd)

        # e = x_H2 + c_h2 * dt   (overwrites c_h2)
        np.multiply(c_h2, dt_sub, out=e)
        np.add(e, xH2, out=e)

        # f = x_HII + c_hii * dt  (overwrites c_hii)
        np.multiply(c_hii, dt_sub, out=f)
        np.add(f, xHII, out=f)

        # det = a*d - b*c   (>= 1 + ... > 0 always)
        np.multiply(a, dd, out=det)
        np.multiply(b, cc, out=tmp_ncell)
        np.subtract(det, tmp_ncell, out=det)

        # x_H2_new = (d*e - b*f) / det
        x_new_H2 = state.get_scratch('solver:x_new_H2')
        np.multiply(dd, e, out=x_new_H2)
        np.multiply(b, f, out=tmp_ncell)
        np.subtract(x_new_H2, tmp_ncell, out=x_new_H2)
        np.divide(x_new_H2, det, out=x_new_H2)

        # x_HII_new = (a*f - c*e) / det
        x_new_HII = state.get_scratch('solver:x_new_HII')
        np.multiply(a, f, out=x_new_HII)
        np.multiply(cc, e, out=tmp_ncell)
        np.subtract(x_new_HII, tmp_ncell, out=x_new_HII)
        np.divide(x_new_HII, det, out=x_new_HII)

        # Hot-gas gate: x_H2 = 0 where T >= temp_hot1
        # (ncr_solver.hpp:532-536).
        hot_gate = state.get_scratch('solver:mask_b1')
        np.less(state.T, self.temp_hot1, out=hot_gate)
        np.multiply(x_new_H2, hot_gate, out=x_new_H2)

        # RAMSES-style closure (ncr_solver.hpp:539-555).
        # 1) Re-normalise if x_HII + 2 x_H2 > 1: divide by
        #    max(x_sum, 1.0) so the no-op case leaves x untouched.
        np.multiply(x_new_H2, 2.0, out=a)              # a := 2 x_H2_new
        np.add(a, x_new_HII, out=a)                    # a := x_sum
        np.maximum(a, 1.0, out=a)                      # a := max(x_sum, 1)
        np.divide(x_new_H2, a, out=x_new_H2)
        np.divide(x_new_HII, a, out=x_new_HII)

        # 2) Branch on x_H2 < 0.25. Compute both branches and select.
        #    branch_low  (x_H2 < 0.25):
        #        x_H2  = clip(x_H2,  TINY, 0.5)
        #        x_HII = clip(x_HII, TINY, 1 - 2 x_H2)
        #    branch_high (x_H2 >= 0.25):
        #        x_HII = clip(x_HII, TINY, 1)
        #        x_H2  = clip(x_H2,  TINY, 1 - x_HII)
        low_mask = state.get_scratch('solver:mask_b2')
        np.less(x_new_H2, 0.25, out=low_mask)

        # Branch low: x_H2_lo -> b, x_HII_lo -> cc
        np.maximum(x_new_H2, _TINY_NUMBER, out=b)
        np.minimum(b, 0.5, out=b)                      # x_H2_lo
        np.multiply(b, 2.0, out=tmp_ncell)
        np.subtract(1.0, tmp_ncell, out=tmp_ncell)     # 1 - 2 x_H2_lo
        np.maximum(x_new_HII, _TINY_NUMBER, out=cc)
        np.minimum(cc, tmp_ncell, out=cc)              # x_HII_lo

        # Branch high: x_HII_hi -> dd, x_H2_hi -> e
        np.maximum(x_new_HII, _TINY_NUMBER, out=dd)
        np.minimum(dd, 1.0, out=dd)                    # x_HII_hi
        np.subtract(1.0, dd, out=tmp_ncell)            # 1 - x_HII_hi
        np.maximum(x_new_H2, _TINY_NUMBER, out=e)
        np.minimum(e, tmp_ncell, out=e)                # x_H2_hi

        # Select: low_mask True -> low branch (b, cc); False -> high
        # branch (e, dd). `np.where` does not accept `out=`; copy the
        # high-branch result first, then overwrite the cells where
        # `low_mask` is True with the low-branch result. Two `copyto`
        # calls per output array; both are alloc-free.
        np.copyto(x_new_H2,  e)
        np.copyto(x_new_H2,  b,  where=low_mask)
        np.copyto(x_new_HII, dd)
        np.copyto(x_new_HII, cc, where=low_mask)

        # x_HI_new = max(1 - x_HII_new - 2 x_H2_new, TINY)
        x_new_HI = state.get_scratch('solver:x_new_HI')
        np.multiply(x_new_H2, 2.0, out=x_new_HI)
        np.add(x_new_HI, x_new_HII, out=x_new_HI)
        np.subtract(1.0, x_new_HI, out=x_new_HI)
        np.maximum(x_new_HI, _TINY_NUMBER, out=x_new_HI)

        # Write back.
        np.copyto(state.x[i_HI], x_new_HI)
        np.copyto(state.x[i_HII], x_new_HII)
        np.copyto(state.x[i_H2], x_new_H2)
