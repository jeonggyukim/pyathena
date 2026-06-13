"""End-to-end smoke test for `ChemistryDriver`.

Builds a `ChemState` for a 1024-cell strip at HII-region conditions,
wires NCRNetwork3 + ExplicitSubcyclingSolver + NCRThermo with the
radiation / opacity / cooling stub policies, and steps for ~1000
years. The test does not assert any specific abundance or temperature
target; it confirms the wiring runs end to end without crashing,
preserves H mass closure, and records substep statistics on the
diagnostics.
"""
from __future__ import annotations

import numpy as np
import pytest

from pyathena.chemistry.config import ChemistryConfig
from pyathena.chemistry.driver import ChemistryDriver
from pyathena.chemistry.networks.ncr3 import NCRNetwork3
from pyathena.chemistry.species import SpeciesSet
from pyathena.chemistry.state import ChemState
from pyathena.chemistry.thermo.ncr import NCRThermo
from pyathena.chemistry.solvers._stubs import (
    CoolingStub,
    OpacityStub,
    RadiationStub,
)


YEAR_SEC = 3.155e7


def test_driver_step_smoke_1024_cell_HII_region():
    """Run one driver step on a 1024-cell strip at HII-region
    conditions for 1000 years and check the basic invariants.
    """
    species = SpeciesSet.ncr3_with_ghosts()
    ncell = 1024
    r = np.linspace(0.1, 1.0, ncell)
    nH = np.full(ncell, 100.0)
    T = np.full(ncell, 1.0e4)
    state = ChemState.from_grid(r, nH, T, species, xi_CR=2.0e-16,
                                nfreq=3)
    # Seed a small ionised fraction so HII chemistry has something to
    # work with.
    species_idx = species.idx
    state.x[species_idx['HI']] = 0.99
    state.x[species_idx['HII']] = 0.01

    config = ChemistryConfig()
    network = NCRNetwork3()
    thermo = NCRThermo()
    radiation = RadiationStub({
        'chi_FUV': 1.0,
        'xi_ph_HI': 0.0,
        'xi_ph_H2': 0.0,
        'xi_diss_H2': 0.0,
    })
    driver = ChemistryDriver(
        config, network, thermo,
        radiation=radiation,
        opacity=OpacityStub(),
        cooling=CoolingStub(),
    )
    driver.setup(state)

    dt = 1.0e3 * YEAR_SEC
    state.reset_step(dt, 0.0)
    nsub = driver.step(dt, state)

    # ---- Basic sanity ----
    assert nsub >= 1
    assert state.diag.n_substeps_total >= 1
    assert np.all(np.isfinite(state.x))
    assert np.all(np.isfinite(state.T))
    assert np.all(state.T > 0.0)
    # H closure to round-off.
    xHI = state.x[species_idx['HI']]
    xHII = state.x[species_idx['HII']]
    xH2 = state.x[species_idx['H2']]
    np.testing.assert_allclose(xHI + xHII + 2.0 * xH2, 1.0,
                               rtol=1.0e-12, atol=0.0)


def test_driver_diagnostics_snapshot_is_dict():
    """`ChemistryDriver.diagnostics(state)` round-trips the SolverDiag."""
    species = SpeciesSet.ncr3_with_ghosts()
    ncell = 8
    r = np.linspace(0.1, 1.0, ncell)
    nH = np.full(ncell, 1.0)
    T = np.full(ncell, 8000.0)
    state = ChemState.from_grid(r, nH, T, species, xi_CR=2.0e-16,
                                nfreq=3)

    config = ChemistryConfig()
    driver = ChemistryDriver(config, NCRNetwork3(), NCRThermo())
    driver.setup(state)

    state.reset_step(1.0e3, 0.0)
    driver.step(1.0e3, state)

    snap = ChemistryDriver.diagnostics(state)
    assert isinstance(snap, dict)
    assert snap['n_substeps_total'] >= 1


def test_driver_resolves_solver_from_config_when_solver_kw_absent():
    """When the caller omits `solver=...`, the driver constructs one
    from `config.solver.name` via the registry.
    """
    species = SpeciesSet.ncr3_with_ghosts()
    ncell = 4
    r = np.linspace(0.1, 1.0, ncell)
    nH = np.full(ncell, 1.0)
    T = np.full(ncell, 8000.0)
    state = ChemState.from_grid(r, nH, T, species, nfreq=3)

    config = ChemistryConfig()
    driver = ChemistryDriver(config, NCRNetwork3(), NCRThermo())
    # solver attribute is populated via the registry lookup.
    assert driver.solver is not None
    assert type(driver.solver).__name__ == 'ExplicitSubcyclingSolver'
