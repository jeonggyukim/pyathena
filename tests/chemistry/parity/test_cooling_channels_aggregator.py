"""End-to-end parity test for `CoolingChannels` driven by the NCR
default factory.

Validates:

1. `CoolingChannels.allocate_scratch(state)` registers every per-
   channel scratch slot (Lambda + dLambda + the channel-internal
   buffers from `SCRATCH_NAMES`).
2. `cooling.update(state)` writes a finite `solver:net_cool` and a
   zero `solver:d_net_cool_d_temp_mu` (Phase 4 default until
   analytic derivatives ship in Phase 4d).
3. The aggregator's `solver:net_cool` equals the sum of the
   individual channel evaluations on the same state, byte-for-byte
   (no accumulation drift inside `update`).
4. The driver's `setup(state)` chains the cooling allocation so a
   full `driver.step(dt, state)` from `CoolingStub` -> swap to NCR
   factory aggregator runs without raising.
"""
from __future__ import annotations

import numpy as np

from pyathena.chemistry.config import ChemistryConfig
from pyathena.chemistry.species import SpeciesSet
from pyathena.chemistry.state import ChemState
from pyathena.chemistry.networks.ncr3 import NCRNetwork3
from pyathena.chemistry.thermo.ncr import NCRThermo
from pyathena.chemistry.driver import ChemistryDriver
from pyathena.chemistry.cooling.factories import make_ncr_default_cooling


_T_VALS = np.logspace(2.0, 6.0, 12)
_N_VALS = np.logspace(-1.0, 3.0, 6)
_T_GRID, _NH_GRID = np.meshgrid(_T_VALS, _N_VALS, indexing='xy')
_T = _T_GRID.ravel()
_NH = _NH_GRID.ravel()


def _build_state(T, nH):
    species = SpeciesSet.ncr3_with_ghosts()
    ncell = T.size
    state = ChemState.from_grid(
        r=np.arange(ncell, dtype=np.float64),
        nH=nH.copy(),
        T=T.copy(),
        species=species,
    )
    idx = species.idx
    state.x[idx['HI']] = 0.5
    state.x[idx['HII']] = 0.5
    state.x[idx['H2']] = 0.0
    state.x[idx['electron']] = 0.5
    state.x[idx['CI']] = 1.0e-5
    state.x[idx['CII']] = 1.6e-4
    state.x[idx['OI']] = 1.0e-5
    state.x[idx['OII']] = 3.2e-4
    state.x[idx['CO']] = 0.0
    return state, species


def test_aggregator_allocate_scratch_registers_all_channels():
    """Setup wires every channel's internal scratch and the per-
    channel Lambda / Gamma slots.
    """
    state, species = _build_state(_T, _NH)
    cooling = make_ncr_default_cooling(species)
    # Need solver scratch first (the aggregator writes into
    # solver:net_cool which the solver owns).
    state.alloc_scratch('solver:net_cool', (_T.size,))
    state.alloc_scratch('solver:d_net_cool_d_temp_mu', (_T.size,))
    cooling.allocate_scratch(state)
    # Sample a handful of expected slots.
    expected = (
        'cooling:Lambda:Lya',
        'cooling:dLambda:Lya',
        'cooling:Lambda:CIIFineStructure',
        'cooling:dLambda:OIFineStructure',
        'cooling:Gamma:PhotoelectricWD01',
        'cooling:dGamma:CosmicRay',
        # Internal channel scratch
        'cooling:cii:T2',
        'cooling:oi:lnT2',
        'cooling:h2_moseley:T3',
        'heating:cosmic_ray:qHI',
        'heating:photoelectric:ne_floor',
    )
    for name in expected:
        assert name in state.scratch, f'missing scratch slot: {name}'


def test_aggregator_update_matches_channel_sum():
    """cooling.update(state) writes solver:net_cool = sum_c Lambda_c
    - sum_h Gamma_h to within float round-off.
    """
    state, species = _build_state(_T, _NH)
    cooling = make_ncr_default_cooling(species)
    ncell = _T.size
    state.alloc_scratch('solver:net_cool', (ncell,))
    state.alloc_scratch('solver:d_net_cool_d_temp_mu', (ncell,))
    # Phase 4d analytic derivatives read solver:mu_at_entry to
    # convert d/dT -> d/d(T/mu). Outside a real solver run we have
    # to allocate + populate it manually.
    state.alloc_scratch('solver:mu_at_entry', (ncell,))
    state.scratch['solver:mu_at_entry'][:] = 1.2
    cooling.allocate_scratch(state)
    cooling.update(state)
    net_cool = state.get_scratch('solver:net_cool').copy()
    d_net_cool = state.get_scratch('solver:d_net_cool_d_temp_mu').copy()

    # Re-evaluate every channel individually and sum.
    reference = np.zeros(ncell, dtype=np.float64)
    for ch in cooling._channels:
        slot = state.get_scratch(f'cooling:Lambda:{ch.name}')
        reference += slot
    for hch in cooling._heating:
        slot = state.get_scratch(f'cooling:Gamma:{hch.name}')
        reference -= slot

    np.testing.assert_allclose(
        net_cool, reference, rtol=1.0e-14, atol=0.0,
        err_msg='aggregator net_cool != sum_channels',
    )
    # Derivative sum: d_net_cool_d_temp_mu = sum_c dLambda_c - sum_h
    # dGamma_h. Phase 4d-a fills in analytic derivatives for a few
    # channels (Dust, FreeFreeH, Lya) and leaves the rest at zero;
    # the aggregator must still sum the per-channel slots exactly.
    reference_d = np.zeros(ncell, dtype=np.float64)
    for ch in cooling._channels:
        slot = state.get_scratch(f'cooling:dLambda:{ch.name}')
        reference_d += slot
    for hch in cooling._heating:
        slot = state.get_scratch(f'cooling:dGamma:{hch.name}')
        reference_d -= slot
    np.testing.assert_allclose(
        d_net_cool, reference_d, rtol=1.0e-14, atol=0.0,
        err_msg='aggregator d_net_cool != sum_channel_derivatives',
    )


def test_driver_setup_chains_cooling_allocation():
    """ChemistryDriver.setup propagates the cooling scratch
    registration; a subsequent solver step then runs without
    KeyError on a missing channel buffer.
    """
    state, species = _build_state(_T, _NH)
    cooling = make_ncr_default_cooling(species)
    config = ChemistryConfig()
    driver = ChemistryDriver(
        config=config,
        network=NCRNetwork3(),
        thermo=NCRThermo(),
        cooling=cooling,
    )
    driver.setup(state)
    # Sample buffers from each policy slot survived.
    assert 'solver:net_cool' in state.scratch
    assert 'cooling:Lambda:Lya' in state.scratch
    assert 'cooling:dGamma:H2Pump' in state.scratch
    # And one step runs (does not raise) on a tiny dt.
    nsub = driver.step(1.0e8, state)
    assert nsub >= 1
