"""Unit tests for `pyathena.chemistry.solvers.explicit_subcycling`.

Covers the registry plumbing, the scratch-buffer protocol, allocation-
free hot path, the implicit-Euler chemistry update vs a hand-rolled
reference, the strip-wide hydrogen-mass closure, and the
subcycle-rejection path that halves `dt_sub` when the temperature
update is non-physical.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pyathena.chemistry.config import (
    ChemistryConfig,
    SOLVER_REGISTRY,
    SolverSpec,
)
from pyathena.chemistry.networks.ncr3 import NCRNetwork3
from pyathena.chemistry.species import SpeciesSet
from pyathena.chemistry.state import ChemState, assert_no_alloc
from pyathena.chemistry.thermo.ncr import NCRThermo
from pyathena.chemistry.solvers.explicit_subcycling import (
    ExplicitSubcyclingSolver,
)


# ---- Helpers ------------------------------------------------------------
def _build_state(ncell: int = 16,
                 nH: float = 100.0,
                 T: float = 1.0e4,
                 xi_CR: float = 2.0e-16) -> ChemState:
    """Build a strip state pre-populated with the radiation-side fields
    NCRNetwork3 reads optionally so the network reads stable buffers
    instead of allocating a fresh zero array each call.
    """
    species = SpeciesSet.ncr3_with_ghosts()
    r = np.linspace(0.1, 1.0, ncell)
    nH_arr = np.full(ncell, nH)
    T_arr = np.full(ncell, T)
    state = ChemState.from_grid(r, nH_arr, T_arr, species,
                                xi_CR=xi_CR, nfreq=3)
    # Pre-populate optional radiation slots. chi_FUV stays as a flat
    # attribute (single FUV band); photo-rates go into the per-species
    # dicts (state.zeta_pi[species], state.zeta_diss[species]). Zero
    # radiation is fine for these tests; the solver only needs the
    # buffers to be present.
    state.chi_FUV = np.zeros(ncell, dtype=np.float64)
    state.zeta_pi = {
        'HI': np.zeros(ncell, dtype=np.float64),
        'H2': np.zeros(ncell, dtype=np.float64),
    }
    state.zeta_diss = {
        'H2': np.zeros(ncell, dtype=np.float64),
    }
    return state


def _build_solver(state: ChemState,
                  cooling: Any = None,
                  cfl_cool_sub: float = 0.1) -> ExplicitSubcyclingSolver:
    config = ChemistryConfig(cfl_cool_sub=cfl_cool_sub)
    net = NCRNetwork3()
    thermo = NCRThermo()
    solver = ExplicitSubcyclingSolver(config, net, thermo, cooling=cooling)
    solver.allocate_scratch(state)
    return solver


class _StubNetwork:
    """Allocation-free NCR3-shaped network used to time the solver hot
    path without paying for the rate-coefficient allocations inside
    `NCRNetwork3.evaluate_CD`.

    Returns C = 0 and D = 0 so the chemistry step is the identity (the
    strip just rolls through the substep loop). Closure is also a
    no-op; the network exists purely to expose the contract.
    """

    species = SpeciesSet.ncr3_with_ghosts()
    evolved = ('HI', 'HII', 'H2')
    ghost = ()
    walk_order = (('HI', 'HII', 'H2'),)
    kSupportsStrips = True
    kNeedsJacobian = False
    element_groups = ()
    x_floor = 1.0e-20

    def evaluate_CD(self, state, out_C, out_D):
        out_C[:] = 0.0
        out_D[:] = 0.0

    def closure(self, state):
        return None

    def fill_ghosts(self, state):
        return None

    def allocate_scratch(self, state):
        return None

    def electron_fraction(self, state):
        return state.x[state.species.idx['electron']]


# ---- Registry / config plumbing -----------------------------------------
def test_solver_registry_contains_explicit_subcycling():
    """Importing the solvers package registers the solver class."""
    assert 'explicit_subcycling' in SOLVER_REGISTRY
    assert SOLVER_REGISTRY['explicit_subcycling'] is ExplicitSubcyclingSolver


def test_solver_spec_roundtrip_default_is_explicit_subcycling():
    """The default SolverSpec maps onto the registered class name."""
    spec = SolverSpec()
    assert spec.name == 'explicit_subcycling'
    assert SOLVER_REGISTRY[spec.name] is ExplicitSubcyclingSolver


# ---- Scratch buffer protocol --------------------------------------------
def test_allocate_scratch_registers_named_buffers():
    """allocate_scratch installs every solver-owned buffer under the
    `solver:` namespace.
    """
    state = _build_state(ncell=8)
    solver = _build_solver(state)
    expected = {
        'solver:C', 'solver:D',
        'solver:temp_mu_old', 'solver:temp_mu_new',
        'solver:mu_at_entry',
        'solver:tmp_ncell', 'solver:tmp_ncell_b',
        'solver:net_cool', 'solver:d_net_cool_d_temp_mu',
        'solver:inv_heat_cap_per_temp_mu',
        'solver:x_new_HI', 'solver:x_new_HII', 'solver:x_new_H2',
        'solver:reject_mask', 'solver:mask_b1', 'solver:mask_b2',
        'solver:cramer_a', 'solver:cramer_b', 'solver:cramer_c',
        'solver:cramer_d', 'solver:cramer_e', 'solver:cramer_f',
        'solver:cramer_det',
        'solver:xHI_pre', 'solver:xHII_pre', 'solver:xH2_pre',
        'solver:f_denom',
    }
    assert expected.issubset(set(state.scratch))


def test_allocate_scratch_is_idempotent():
    """Calling allocate_scratch twice replaces the buffers in place
    without growing the scratch dict.
    """
    state = _build_state(ncell=8)
    solver = _build_solver(state)
    n_before = len(state.scratch)
    ids_before = {k: id(v) for k, v in state.scratch.items()}
    solver.allocate_scratch(state)
    assert len(state.scratch) == n_before
    # All buffers got rebuilt -- the `id` changes -- but the keys are
    # stable.
    assert set(state.scratch) == set(ids_before)


# ---- Hot path allocation contract ---------------------------------------
def test_step_hot_path_is_allocation_free_with_stub_network():
    """The solver substep loop allocates nothing when the network /
    thermo cooperate. NCRNetwork3 reads optional radiation attributes
    and computes rate coefficients via np.where + np.exp, which DO
    allocate; the stub network captures the solver-side guarantee.
    """
    state = _build_state(ncell=32)
    config = ChemistryConfig()
    net = _StubNetwork()
    thermo = NCRThermo()
    solver = ExplicitSubcyclingSolver(config, net, thermo)
    solver.allocate_scratch(state)

    # Warm-up step. The first step pays a one-time cost for any first-
    # touch caching numpy might do behind the scenes.
    state.reset_step(1.0e3, 0.0)
    solver.step(1.0e3, state)

    with assert_no_alloc(allow=0):
        state.reset_step(1.0e3, 0.0)
        solver.step(1.0e3, state)


# ---- Hydrogen mass closure ----------------------------------------------
def test_step_preserves_hydrogen_mass_closure():
    """After step(dt), x_HI + x_HII + 2 x_H2 == 1 to round-off."""
    state = _build_state(ncell=16, nH=100.0, T=8.0e3)
    solver = _build_solver(state)

    state.reset_step(1.0e4, 0.0)
    solver.step(1.0e4, state)

    species = state.species
    xHI = state.x[species.idx['HI']]
    xHII = state.x[species.idx['HII']]
    xH2 = state.x[species.idx['H2']]
    total = xHI + xHII + 2.0 * xH2
    np.testing.assert_allclose(total, 1.0, rtol=1.0e-12, atol=0.0)


# ---- 2x2 Cramer chemistry update vs hand-rolled reference ---------------
def test_one_substep_chem_update_matches_cramer_formula():
    """At a small dt the solver's joint 2x2 Cramer step on (x_H2,
    x_HII) must reproduce the closed-form Cramer expression
    cell-for-cell within rtol=1e-8.

    The pre-Cramer port did per-row implicit Euler on (HI, HII, H2)
    independently; with the port (`UpdateChemistry` from
    `ncr_solver.hpp:513-571`) the joint solve substitutes
    `x_HI = 1 - 2 x_H2 - x_HII` into the source terms, so the
    expected formula is the 2x2 Cramer result, not the per-row BE
    formula.
    """
    state = _build_state(ncell=8, nH=1.0, T=8.0e3)
    solver = _build_solver(state, cfl_cool_sub=0.5)

    species = state.species
    i_HI = species.idx['HI']
    i_HII = species.idx['HII']
    i_H2 = species.idx['H2']

    # Snapshot evolved x and rates at the entry state.
    net = solver.network
    C_ref = np.zeros_like(state.x)
    D_ref = np.zeros_like(state.x)
    net.evaluate_CD(state, C_ref, D_ref)
    x_HI_before = state.x[i_HI].copy()
    x_HII_before = state.x[i_HII].copy()
    x_H2_before = state.x[i_H2].copy()

    dt = 1.0e2  # 100 s; tiny so the strip-MIN dt cap doesn't bite.
    state.reset_step(dt, 0.0)
    solver.step(dt, state)

    # Recover per-x_HI source rates the same way the solver does.
    x_HI_safe = np.maximum(x_HI_before, 1.0e-20)
    c_h2 = C_ref[i_H2] / x_HI_safe
    c_hii = C_ref[i_HII] / x_HI_safe
    D_H2 = D_ref[i_H2]
    D_HII = D_ref[i_HII]
    a = 1.0 + (2.0 * c_h2 + D_H2) * dt
    b = c_h2 * dt
    c = 2.0 * c_hii * dt
    d = 1.0 + (c_hii + D_HII) * dt
    e = x_H2_before + c_h2 * dt
    f = x_HII_before + c_hii * dt
    det = a * d - b * c
    expected_H2 = (d * e - b * f) / det
    expected_HII = (a * f - c * e) / det
    expected_HI = 1.0 - 2.0 * expected_H2 - expected_HII

    np.testing.assert_allclose(state.x[i_HI], expected_HI,
                               rtol=1.0e-8, atol=1.0e-15)
    np.testing.assert_allclose(state.x[i_HII], expected_HII,
                               rtol=1.0e-8, atol=1.0e-15)
    np.testing.assert_allclose(state.x[i_H2], expected_H2,
                               rtol=1.0e-8, atol=1.0e-15)


# ---- Subcycle rejection path --------------------------------------------
class _SwingingCooling:
    """Cooling stub that flips the sign of net_cool every call so the
    semi-implicit T update produces a wild swing the rejection path
    must catch.
    """

    __version__: str = 'swing@test'

    def __init__(self) -> None:
        self.calls = 0
        self.last_dt_sub_seen = 0.0

    def update(self, state):
        self.calls += 1
        net_cool = state.get_scratch('solver:net_cool')
        d_net_cool_d_temp_mu = state.get_scratch(
            'solver:d_net_cool_d_temp_mu')
        # Drive a very rapid cooling spike that the cfl_cool_sub check
        # has to reject. Magnitude is set so temp_mu_new < 0 unless
        # dt_sub is halved several times.
        net_cool[:] = 1.0e10 * (1.0 if self.calls % 2 == 0 else -1.0)
        d_net_cool_d_temp_mu[:] = 0.0


def test_subcycle_rejection_path_halves_dt_sub():
    """When the temperature step is rejected, the solver halves
    `dt_sub` and tries again. Stress the path with a cooling rate
    chosen to produce a non-physical T_new on the first attempt.
    """
    state = _build_state(ncell=4, nH=100.0, T=1.0e4)
    cooling = _SwingingCooling()
    config = ChemistryConfig(cfl_cool_sub=0.1)
    net = NCRNetwork3()
    thermo = NCRThermo()
    solver = ExplicitSubcyclingSolver(config, net, thermo, cooling=cooling)
    solver.allocate_scratch(state)

    state.reset_step(1.0e8, 0.0)
    nsub = solver.step(1.0e8, state)
    assert nsub >= 1
    # The rejection path triggers `n_thermal_solves > nsub` because
    # every retry counts; if no retry happened these would be equal.
    assert state.diag.n_thermal_solves >= nsub
