"""Unit tests for `pyathena.chemistry.networks.ncr3.NCRNetwork3`.

Covers the (C, D) rate split, the H-mass + charge closure, electron-
fraction extraction, the class-level species metadata, and a small
forward-Euler driver that should drive the network toward something
close to collisional-ionisation equilibrium at T = 8000 K. The driver
test is a sanity check, not a parity test against
`pyathena.microphysics.photchem.PhotChem` — Phase 6 is where strict
1e-12 parity returns with `NCRNetwork3PlusIons16`.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import numpy as np
import pytest

from pyathena.chemistry.networks.ncr3 import NCRNetwork3
from pyathena.chemistry.species import SpeciesSet


# ---- Lightweight state stand-in ----------------------------------------
# The Phase 0 ChemState requires factory wiring that is not yet
# available; we build a minimal struct that satisfies what
# NCRNetwork3.evaluate_CD / closure / electron_fraction actually read.
def _make_state(
    *,
    x: np.ndarray,
    nH: float = 1.0,
    T: float = 1.0e4,
    Z_d: float = 1.0,
    Z_g: float = 1.0,
    xi_CR: float = 0.0,
    chi_FUV: Optional[float] = None,
    xi_ph_HI: Optional[float] = None,
    xi_ph_H2: Optional[float] = None,
    xi_diss_H2: Optional[float] = None,
    ncell: int = 1,
    species=None,
) -> SimpleNamespace:
    """Build a SimpleNamespace mimicking the fields of ChemState that
    NCRNetwork3 reads. Scalars are broadcast to (ncell,).
    """
    def _bcast(val):
        if val is None:
            return None
        return np.full((ncell,), float(val), dtype=np.float64)

    if species is None:
        species = NCRNetwork3.species

    return SimpleNamespace(
        species=species,
        x=np.ascontiguousarray(x, dtype=np.float64),
        nH=_bcast(nH),
        T=_bcast(T),
        Z_d=_bcast(Z_d),
        Z_g=_bcast(Z_g),
        xi_CR=_bcast(xi_CR),
        chi_FUV=_bcast(chi_FUV),
        xi_ph_HI=_bcast(xi_ph_HI),
        xi_ph_H2=_bcast(xi_ph_H2),
        xi_diss_H2=_bcast(xi_diss_H2),
    )


def _zeros_x(species, ncell: int = 1) -> np.ndarray:
    """Allocate a zero-initialised abundance array sized to `species`."""
    return np.zeros((species.nspec, ncell), dtype=np.float64)


def _seed_x(species, *, xHI: float, xHII: float, xH2: float,
            ncell: int = 1, x_floor: float = 1.0e-20) -> np.ndarray:
    """Build an abundance strip with the evolved rows seeded to the
    given (xHI, xHII, xH2) and the ghost rows initialised to the floor.
    """
    x = np.full((species.nspec, ncell), x_floor, dtype=np.float64)
    x[species.idx['HI']] = xHI
    x[species.idx['HII']] = xHII
    x[species.idx['H2']] = xH2
    return x


# ---- Tests --------------------------------------------------------------
def test_species_metadata_carries_evolved_ghost_partition():
    """NCRNetwork3.species is the 9-row evolved+ghost layout."""
    net = NCRNetwork3()
    species = NCRNetwork3.species
    assert isinstance(species, SpeciesSet)
    assert species.names == (
        'HI', 'HII', 'H2',
        'electron', 'CI', 'CII', 'CO', 'OI', 'OII',
    )
    assert species.nspec == 9
    # Identity-check the class-level constant matches the instance's
    # view so the solver can do `state.species is net.species`.
    assert net.species is species
    assert NCRNetwork3.walk_order == (('HI', 'HII', 'H2'),)
    assert NCRNetwork3.kSupportsStrips is True
    assert NCRNetwork3.kNeedsJacobian is False
    assert NCRNetwork3.evolved == ('HI', 'HII', 'H2')
    assert NCRNetwork3.ghost == (
        'electron', 'CI', 'CII', 'CO', 'OI', 'OII',
    )
    # SpeciesSet should agree with the network on the partition.
    assert species.evolved_names == NCRNetwork3.evolved
    assert species.ghost_names == NCRNetwork3.ghost
    assert species.n_evolved == 3
    assert species.n_ghost == 6


def test_evaluate_CD_shape_and_signs_dark_CR_only():
    """At T=1e4 K, nH=1, evaluate_CD should produce non-negative
    rates over the full 9-row strip. The ghost rows are left
    untouched by evaluate_CD (they are filled by fill_ghosts).
    """
    net = NCRNetwork3()
    species = NCRNetwork3.species
    x = _seed_x(species, xHI=0.5, xHII=0.5, xH2=0.0,
                x_floor=net.x_floor)
    # Seed the electron ghost row to match the diagnostic input.
    x[species.idx['electron']] = 0.5
    state = _make_state(x=x, T=1.0e4, nH=1.0, xi_CR=0.0)

    C = np.zeros_like(x)
    D = np.zeros_like(x)
    net.evaluate_CD(state, C, D)

    assert C.shape == (species.nspec, 1)
    assert D.shape == (species.nspec, 1)
    # All rate magnitudes finite.
    assert np.all(np.isfinite(C))
    assert np.all(np.isfinite(D))
    # Non-negativity: C is a positive creation rate per unit source
    # fraction; D is a positive destruction frequency.
    assert np.all(C >= 0.0)
    assert np.all(D >= 0.0)
    # Electron row is not integrated by this network: both 0.
    i_e = species.idx['electron']
    assert C[i_e, 0] == 0.0
    assert D[i_e, 0] == 0.0


def test_evaluate_CD_radiation_increases_HII_creation():
    """With photoionization on, HII creation rate strictly exceeds
    the dark, no-CR baseline. Verifies the radiation field actually
    enters the rate split.
    """
    net = NCRNetwork3()
    species = NCRNetwork3.species
    x = _seed_x(species, xHI=1.0, xHII=1.0e-10, xH2=0.0,
                x_floor=net.x_floor)
    x[species.idx['electron']] = 1.0e-10

    C_dark = np.zeros_like(x)
    D_dark = np.zeros_like(x)
    state_dark = _make_state(x=x.copy(), T=8000.0, nH=1.0,
                             xi_ph_HI=0.0)
    net.evaluate_CD(state_dark, C_dark, D_dark)

    C_phot = np.zeros_like(x)
    D_phot = np.zeros_like(x)
    state_phot = _make_state(x=x.copy(), T=8000.0, nH=1.0,
                             xi_ph_HI=1.0e-12)
    net.evaluate_CD(state_phot, C_phot, D_phot)

    i_HII = species.idx['HII']
    assert C_phot[i_HII, 0] > C_dark[i_HII, 0]


def test_evaluate_CD_hot_gas_zeroes_H2_chemistry():
    """Above TEMP_HOT1 = 3.5e4 K, the C++ reference zeroes H2 rates."""
    net = NCRNetwork3()
    species = NCRNetwork3.species
    x = _seed_x(species, xHI=0.5, xHII=0.5, xH2=1.0e-10,
                x_floor=net.x_floor)
    x[species.idx['electron']] = 0.5
    state = _make_state(x=x, T=5.0e4, nH=1.0, xi_CR=1.0e-16)

    C = np.zeros_like(x)
    D = np.zeros_like(x)
    net.evaluate_CD(state, C, D)

    i_H2 = species.idx['H2']
    assert C[i_H2, 0] == 0.0
    assert D[i_H2, 0] == 0.0


def test_closure_renormalises_to_hydrogen_mass_conservation():
    """closure() must enforce x_HI + x_HII + 2 x_H2 = 1 and refill
    the ghost rows from the post-conservation evolved values.
    """
    net = NCRNetwork3()
    species = NCRNetwork3.species
    # Deliberately violate the closure: sum is 0.4 + 0.4 + 2*0.05 = 0.9.
    x = _seed_x(species, xHI=0.4, xHII=0.4, xH2=0.05,
                x_floor=net.x_floor)
    state = _make_state(x=x, T=1.0e4)

    net.closure(state)

    i_HI = species.idx['HI']
    i_HII = species.idx['HII']
    i_H2 = species.idx['H2']
    i_e = species.idx['electron']
    i_CII = species.idx['CII']
    i_OII = species.idx['OII']

    total = state.x[i_HI, 0] + state.x[i_HII, 0] + 2.0 * state.x[i_H2, 0]
    assert total == pytest.approx(1.0, rel=1e-12)
    # Charge neutrality with metal ghosts: x_e = x_HII + x_CII + x_OII.
    assert state.x[i_e, 0] == pytest.approx(
        state.x[i_HII, 0] + state.x[i_CII, 0] + state.x[i_OII, 0],
        rel=1e-12,
    )


def test_closure_floors_negative_input():
    """Negative or tiny values get clamped to x_floor (1e-20)."""
    net = NCRNetwork3()
    species = NCRNetwork3.species
    x = _seed_x(species, xHI=0.999, xHII=-1.0e-12, xH2=-1.0e-12,
                x_floor=net.x_floor)
    state = _make_state(x=x, T=1.0e4)
    net.closure(state)

    i_HII = species.idx['HII']
    i_H2 = species.idx['H2']
    assert state.x[i_HII, 0] >= net.x_floor
    assert state.x[i_H2, 0] >= net.x_floor


def test_closure_renormalises_overflow():
    """If x_HII + 2 x_H2 > 1, closure() must rescale them to fit."""
    net = NCRNetwork3()
    species = NCRNetwork3.species
    # x_HII + 2 x_H2 = 0.7 + 2*0.4 = 1.5 — needs rescaling.
    x = _seed_x(species, xHI=0.0, xHII=0.7, xH2=0.4,
                x_floor=net.x_floor)
    state = _make_state(x=x, T=1.0e4)
    net.closure(state)

    i_HI = species.idx['HI']
    i_HII = species.idx['HII']
    i_H2 = species.idx['H2']
    total = state.x[i_HI, 0] + state.x[i_HII, 0] + 2.0 * state.x[i_H2, 0]
    assert total == pytest.approx(1.0, rel=1e-12)
    # Ratio between HII and 2 H2 should be preserved by the rescale.
    np.testing.assert_allclose(
        state.x[i_HII, 0] / (2.0 * state.x[i_H2, 0]),
        0.7 / (2.0 * 0.4),
        rtol=1e-10,
    )


def test_electron_fraction_returns_explicit_electron_row():
    """electron_fraction() reads the explicit electron row when the
    strip carries one.
    """
    net = NCRNetwork3()
    species = NCRNetwork3.species
    x = _seed_x(species, xHI=0.4, xHII=0.3, xH2=0.15,
                x_floor=net.x_floor)
    x[species.idx['electron']] = 0.99
    state = _make_state(x=x, T=1.0e4)
    xe = net.electron_fraction(state)
    np.testing.assert_allclose(xe, [0.99])
    # Read-only: state.x untouched.
    assert state.x[species.idx['HII'], 0] == 0.3


def test_forward_euler_drives_to_CIE_like_equilibrium():
    """Hand-rolled forward-Euler driver: start at pure H I, T=8000 K,
    with CR-only ionisation, and check the system relaxes toward
    Saha-like steady state.

    The expected balance: dx_HII/dt = 0  ==>
        C_HII_eff * x_HI = D_HII * x_HII
    with the same C/D the C++ EvaluateChemistryCD would write.
    For T=8000 K, nH=1, xi_CR=1e-16 s^-1, the CR-only ionisation
    balance against radiative + grain recombination yields a small
    but non-trivial x_HII. We only check qualitative behavior:
    H2 stays floor-level (no FUV shielding to grow it), x_HII climbs
    above the floor but stays << 1, and x_HI is close to 1.
    """
    net = NCRNetwork3()
    species = NCRNetwork3.species
    nH = 1.0
    T_val = 8.0e3
    xi_CR_val = 1.0e-16
    ncell = 1

    x = _seed_x(species, xHI=1.0, xHII=net.x_floor, xH2=net.x_floor,
                x_floor=net.x_floor)
    state = _make_state(
        x=x, nH=nH, T=T_val, Z_d=1.0, xi_CR=xi_CR_val,
        ncell=ncell,
    )

    # Initial fill_ghosts so the first evaluate_CD reads sensible ne.
    net.fill_ghosts(state)

    C = np.zeros_like(x)
    D = np.zeros_like(x)

    # Pick a small implicit-Euler dt and run 100 steps.
    dt = 3.0e10   # ~1000 yr in seconds
    n_steps = 200
    for _ in range(n_steps):
        net.evaluate_CD(state, C, D)
        # x_new = (x + C dt) / (1 + D dt) per-species, then closure.
        state.x[:] = (state.x + C * dt) / (1.0 + D * dt)
        net.closure(state)

    i_HI = species.idx['HI']
    i_HII = species.idx['HII']
    i_H2 = species.idx['H2']

    xHI = state.x[i_HI, 0]
    xHII = state.x[i_HII, 0]
    xH2 = state.x[i_H2, 0]

    # H mass closure holds.
    assert xHI + xHII + 2.0 * xH2 == pytest.approx(1.0, rel=1e-10)
    # H2 stayed small: kgr grows H2 slowly at 8000 K, no FUV shielding
    # but also no collisional dissociation either (T > 700 K threshold).
    # We accept up to a few percent at this temperature.
    assert xH2 < 0.1
    # x_HII climbed above floor but stayed << 1 (CR-only ionisation).
    assert xHII > net.x_floor
    assert xHII < 0.5
    # x_HI dominant.
    assert xHI > 0.5


def test_forward_euler_strip_of_8_cells():
    """Strip support: same forward-Euler driver on ncell=8 with
    identical inputs should produce identical outputs across cells.
    Exercises the broadcast contract over the strip axis.
    """
    net = NCRNetwork3()
    species = NCRNetwork3.species
    ncell = 8
    x = _seed_x(species, xHI=1.0, xHII=net.x_floor, xH2=net.x_floor,
                ncell=ncell, x_floor=net.x_floor)

    state = _make_state(
        x=x, nH=1.0, T=8.0e3, Z_d=1.0, xi_CR=1.0e-16, ncell=ncell,
    )
    net.fill_ghosts(state)

    C = np.zeros_like(x)
    D = np.zeros_like(x)
    dt = 3.0e10
    for _ in range(50):
        net.evaluate_CD(state, C, D)
        state.x[:] = (state.x + C * dt) / (1.0 + D * dt)
        net.closure(state)

    # Every cell got the same answer.
    for row in range(species.nspec):
        np.testing.assert_allclose(
            state.x[row, :], state.x[row, 0], rtol=1e-12, atol=0.0,
        )


def test_jacobian_raises_by_default():
    """NCRNetwork3 does not implement jacobian; default NetworkBase
    behavior is NotImplementedError, which the solver layer relies on.
    """
    net = NCRNetwork3()
    species = NCRNetwork3.species
    x = _seed_x(species, xHI=0.5, xHII=0.5, xH2=0.0,
                x_floor=net.x_floor)
    x[species.idx['electron']] = 0.5
    state = _make_state(x=x, T=1.0e4)
    out_J = np.zeros((species.nspec, species.nspec, 1), dtype=np.float64)
    with pytest.raises(NotImplementedError):
        net.jacobian(state, out_J)


def test_fill_ghosts_preserves_element_conservation():
    """After fill_ghosts on x = [0.5, 0.5, 0, ...] at T = 1e4 K and
    Z_g = 1, the C and O elemental budgets close to round-off:

        x_CI + x_CII + x_CO == xCstd * Z_g
        x_OI + x_OII + x_CO == xOstd * Z_g

    and the electron count matches the singly-ionised positive ions
    plus HII.
    """
    net = NCRNetwork3()
    species = NCRNetwork3.species
    x = _seed_x(species, xHI=0.5, xHII=0.5, xH2=0.0,
                x_floor=net.x_floor)
    state = _make_state(x=x, T=1.0e4, Z_g=1.0)
    net.fill_ghosts(state)

    i_CI = species.idx['CI']
    i_CII = species.idx['CII']
    i_CO = species.idx['CO']
    i_OI = species.idx['OI']
    i_OII = species.idx['OII']
    i_HII = species.idx['HII']
    i_e = species.idx['electron']

    Z_g = 1.0
    xCstd = NCRNetwork3.x_C_std
    xOstd = NCRNetwork3.x_O_std

    # Carbon budget: x_CI + x_CII + x_CO == xCstd * Z_g (to round-off).
    np.testing.assert_allclose(
        state.x[i_CI, 0] + state.x[i_CII, 0] + state.x[i_CO, 0],
        xCstd * Z_g, rtol=1e-12, atol=0.0,
    )
    # Oxygen budget: x_OI + x_OII + x_CO == xOstd * Z_g.
    np.testing.assert_allclose(
        state.x[i_OI, 0] + state.x[i_OII, 0] + state.x[i_CO, 0],
        xOstd * Z_g, rtol=1e-12, atol=0.0,
    )
    # Electron from charge balance.
    np.testing.assert_allclose(
        state.x[i_e, 0],
        state.x[i_HII, 0] + state.x[i_CII, 0] + state.x[i_OII, 0],
        rtol=1e-12, atol=0.0,
    )


def test_fill_ghosts_is_idempotent():
    """Calling fill_ghosts twice must leave state.x bit-for-bit
    unchanged on the second call.
    """
    net = NCRNetwork3()
    species = NCRNetwork3.species
    x = _seed_x(species, xHI=0.6, xHII=0.4, xH2=0.0,
                x_floor=net.x_floor)
    state = _make_state(x=x, T=1.0e4, Z_g=1.0)
    net.fill_ghosts(state)
    snapshot = state.x.copy()
    net.fill_ghosts(state)
    np.testing.assert_array_equal(state.x, snapshot)


def test_fill_ghosts_does_not_touch_evolved_rows():
    """fill_ghosts must mutate only the ghost rows; evolved rows
    survive unchanged.
    """
    net = NCRNetwork3()
    species = NCRNetwork3.species
    x = _seed_x(species, xHI=0.42, xHII=0.27, xH2=0.155,
                x_floor=net.x_floor)
    state = _make_state(x=x, T=1.0e4, Z_g=0.5)
    evolved_before = state.x[species.evolved_idx].copy()
    net.fill_ghosts(state)
    np.testing.assert_array_equal(
        state.x[species.evolved_idx], evolved_before,
    )
