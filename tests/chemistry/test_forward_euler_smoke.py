"""Forward-Euler smoke driver for the Phase 2 chemistry stack.

This is not a parity test. It exercises the integration between
`ChemState`, `NCRNetwork3`, and `NCRThermo` over a small number of
steps and checks that the network's (C, D) split produces physically
plausible behaviour: H mass closure holds, every fraction stays
non-negative, and at least one cell ionises some H.

The C++ analogue is the explicit subcycling loop in
`tigris-ncr/src/photchem/photchem_ncr.cpp::CoolingExplicitSubcycling`,
but the Python smoke test does not attempt to match coefficients or
substep cadence -- only the algebraic shape `dx/dt = C - D x`.
"""
from __future__ import annotations

import numpy as np
import pytest

from pyathena.chemistry.networks.ncr3 import NCRNetwork3
from pyathena.chemistry.species import SpeciesSet
from pyathena.chemistry.state import ChemState
from pyathena.chemistry.thermo.ncr import NCRThermo


def test_forward_euler_smoke_10cell_isobaric():
    """Run 100 small forward-Euler steps on a 10-cell isobaric strip.

    Setup: T = 8000 K, nH = 1 cm^-3, x = (HI=1, HII=0, H2=0, e=0).
    Drive with a non-zero CR ionization rate so HII grows; check
    closure + non-negativity at the end. NCRThermo is exercised as a
    sanity wiring check (mu / pressure both must return finite arrays
    of the right shape).
    """
    species = SpeciesSet.minimal_HI_HII_H2()
    ncell = 10
    r = np.linspace(0.1, 1.0, ncell)
    nH = np.full(ncell, 1.0)
    T = np.full(ncell, 8000.0)
    state = ChemState.from_grid(r, nH, T, species, xi_CR=2.0e-16)

    net = NCRNetwork3()
    therm = NCRThermo()

    # Pre-allocate hot-path buffers.
    C = np.zeros((species.nspec, ncell))
    D = np.zeros((species.nspec, ncell))
    out_scalar = np.zeros(ncell)

    nsteps = 100
    dt = 1.0e8  # ~3 yr, small relative to t_rec at nH=1.

    for _ in range(nsteps):
        net.evaluate_CD(state, C, D)
        # Forward Euler on rows 0,1,2 (HI/HII/H2). The electron row is
        # set by closure() after the step.
        state.x[:3, :] += dt * (C[:3, :] - D[:3, :] * state.x[:3, :])
        net.closure(state)

    # ---- Closure: x_HI + x_HII + 2 x_H2 = 1 to floating-point ----
    x_HI = state.x[species.idx['HI']]
    x_HII = state.x[species.idx['HII']]
    x_H2 = state.x[species.idx['H2']]
    x_e = state.x[species.idx['electron']]
    np.testing.assert_allclose(x_HI + x_HII + 2.0 * x_H2, 1.0, atol=1e-10)

    # ---- Non-negativity ----
    assert np.all(state.x >= 0.0)

    # ---- Some ionization happened ----
    assert np.any(x_HII > 0.0), 'No ionization at all -- driver wiring broken'
    # x_e mirrors x_HII in this network.
    np.testing.assert_allclose(x_e, x_HII)

    # ---- Thermo sanity ----
    therm.mu(state, out_scalar)
    assert np.all(np.isfinite(out_scalar))
    assert np.all(out_scalar > 0.0)
    # For mostly-neutral H + He at A_He=0.0955, mu is near 1.27 amu.
    assert out_scalar.mean() == pytest.approx(1.27, abs=0.1)

    therm.pressure(state, out_scalar)
    assert np.all(np.isfinite(out_scalar))
    assert np.all(out_scalar > 0.0)
