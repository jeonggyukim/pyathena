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
    xi_CR: float = 0.0,
    chi_FUV: Optional[float] = None,
    xi_ph_HI: Optional[float] = None,
    xi_ph_H2: Optional[float] = None,
    xi_diss_H2: Optional[float] = None,
    ncell: int = 1,
) -> SimpleNamespace:
    """Build a SimpleNamespace mimicking the fields of ChemState that
    NCRNetwork3 reads. Scalars are broadcast to (ncell,).
    """
    def _bcast(val):
        if val is None:
            return None
        return np.full((ncell,), float(val), dtype=np.float64)

    return SimpleNamespace(
        species=NCRNetwork3.species,
        x=np.ascontiguousarray(x, dtype=np.float64),
        nH=_bcast(nH),
        T=_bcast(T),
        Z_d=_bcast(Z_d),
        xi_CR=_bcast(xi_CR),
        chi_FUV=_bcast(chi_FUV),
        xi_ph_HI=_bcast(xi_ph_HI),
        xi_ph_H2=_bcast(xi_ph_H2),
        xi_diss_H2=_bcast(xi_diss_H2),
    )


# ---- Tests --------------------------------------------------------------
def test_species_metadata_is_minimal_4_species():
    """NCRNetwork3.species must be the canonical minimal HI/HII/H2/e."""
    net = NCRNetwork3()
    assert isinstance(NCRNetwork3.species, SpeciesSet)
    assert NCRNetwork3.species.names == ('HI', 'HII', 'H2', 'electron')
    assert NCRNetwork3.species.nspec == 4
    # Identity-check the class-level constant matches the instance's
    # view so the solver can do `state.species is net.species`.
    assert net.species is NCRNetwork3.species
    assert NCRNetwork3.walk_order == (('HI', 'HII', 'H2'),)
    assert NCRNetwork3.kSupportsStrips is True
    assert NCRNetwork3.kNeedsJacobian is False


def test_evaluate_CD_shape_and_signs_dark_CR_only():
    """At T=1e4 K, nH=1, x=[0.5, 0.5, 0, 0.5], no radiation, no CR,
    evaluate_CD should produce (4,1)-shaped C and D buffers with
    non-negative entries.

    Note: the input violates H-mass closure (HI + HII = 1.0, with
    H2 = 0), and the survey notes ask for x[3] = 0.5 in this
    diagnostic call. closure() would clean it up; evaluate_CD just
    has to return finite, non-negative rates.
    """
    net = NCRNetwork3()
    x = np.array([[0.5], [0.5], [0.0], [0.5]], dtype=np.float64)
    state = _make_state(x=x, T=1.0e4, nH=1.0, xi_CR=0.0)

    C = np.zeros_like(x)
    D = np.zeros_like(x)
    net.evaluate_CD(state, C, D)

    assert C.shape == (4, 1)
    assert D.shape == (4, 1)
    # All rate magnitudes finite.
    assert np.all(np.isfinite(C))
    assert np.all(np.isfinite(D))
    # Non-negativity: C is a positive creation rate per unit source
    # fraction; D is a positive destruction frequency.
    assert np.all(C >= 0.0)
    assert np.all(D >= 0.0)
    # Electron row is not integrated by this network: both 0.
    i_e = net.species.idx['electron']
    assert C[i_e, 0] == 0.0
    assert D[i_e, 0] == 0.0


def test_evaluate_CD_radiation_increases_HII_creation():
    """With photoionization on, HII creation rate strictly exceeds
    the dark, no-CR baseline. Verifies the radiation field actually
    enters the rate split.
    """
    net = NCRNetwork3()
    x = np.array([[1.0], [1.0e-10], [0.0], [1.0e-10]], dtype=np.float64)

    C_dark = np.zeros_like(x)
    D_dark = np.zeros_like(x)
    state_dark = _make_state(x=x, T=8000.0, nH=1.0, xi_ph_HI=0.0)
    net.evaluate_CD(state_dark, C_dark, D_dark)

    C_phot = np.zeros_like(x)
    D_phot = np.zeros_like(x)
    state_phot = _make_state(x=x, T=8000.0, nH=1.0, xi_ph_HI=1.0e-12)
    net.evaluate_CD(state_phot, C_phot, D_phot)

    i_HII = net.species.idx['HII']
    assert C_phot[i_HII, 0] > C_dark[i_HII, 0]


def test_evaluate_CD_hot_gas_zeroes_H2_chemistry():
    """Above TEMP_HOT1 = 3.5e4 K, the C++ reference zeroes H2 rates."""
    net = NCRNetwork3()
    x = np.array([[0.5], [0.5], [1.0e-10], [0.5]], dtype=np.float64)
    state = _make_state(x=x, T=5.0e4, nH=1.0, xi_CR=1.0e-16)

    C = np.zeros_like(x)
    D = np.zeros_like(x)
    net.evaluate_CD(state, C, D)

    i_H2 = net.species.idx['H2']
    assert C[i_H2, 0] == 0.0
    assert D[i_H2, 0] == 0.0


def test_closure_renormalises_to_hydrogen_mass_conservation():
    """closure() must enforce x_HI + x_HII + 2 x_H2 = 1 and
    x_e = x_HII in place.
    """
    net = NCRNetwork3()
    # Deliberately violate the closure: sum is 0.4 + 0.4 + 2*0.05 = 0.9.
    x = np.array([[0.4], [0.4], [0.05], [0.123]], dtype=np.float64)
    state = _make_state(x=x, T=1.0e4)

    net.closure(state)

    i_HI = net.species.idx['HI']
    i_HII = net.species.idx['HII']
    i_H2 = net.species.idx['H2']
    i_e = net.species.idx['electron']

    total = state.x[i_HI, 0] + state.x[i_HII, 0] + 2.0 * state.x[i_H2, 0]
    assert total == pytest.approx(1.0, rel=1e-12)
    # Charge neutrality: x_e = x_HII for the H-only network.
    assert state.x[i_e, 0] == pytest.approx(state.x[i_HII, 0], rel=1e-12)


def test_closure_floors_negative_input():
    """Negative or tiny values get clamped to x_floor (1e-20)."""
    net = NCRNetwork3()
    x = np.array([[0.999], [-1.0e-12], [-1.0e-12], [0.0]], dtype=np.float64)
    state = _make_state(x=x, T=1.0e4)
    net.closure(state)

    i_HII = net.species.idx['HII']
    i_H2 = net.species.idx['H2']
    assert state.x[i_HII, 0] >= net.x_floor
    assert state.x[i_H2, 0] >= net.x_floor


def test_closure_renormalises_overflow():
    """If x_HII + 2 x_H2 > 1, closure() must rescale them to fit."""
    net = NCRNetwork3()
    # x_HII + 2 x_H2 = 0.7 + 2*0.4 = 1.5 — needs rescaling.
    x = np.array([[0.0], [0.7], [0.4], [0.0]], dtype=np.float64)
    state = _make_state(x=x, T=1.0e4)
    net.closure(state)

    i_HI = net.species.idx['HI']
    i_HII = net.species.idx['HII']
    i_H2 = net.species.idx['H2']
    total = state.x[i_HI, 0] + state.x[i_HII, 0] + 2.0 * state.x[i_H2, 0]
    assert total == pytest.approx(1.0, rel=1e-12)
    # Ratio between HII and 2 H2 should be preserved by the rescale.
    np.testing.assert_allclose(
        state.x[i_HII, 0] / (2.0 * state.x[i_H2, 0]),
        0.7 / (2.0 * 0.4),
        rtol=1e-10,
    )


def test_electron_fraction_returns_x_HII():
    """electron_fraction() returns x_HII (charge neutrality with only
    H II contributing electrons).
    """
    net = NCRNetwork3()
    x = np.array([[0.4], [0.3], [0.15], [0.99]], dtype=np.float64)
    state = _make_state(x=x, T=1.0e4)
    xe = net.electron_fraction(state)
    np.testing.assert_allclose(xe, [0.3])
    # Read-only: state.x untouched.
    assert state.x[net.species.idx['HII'], 0] == 0.3


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
    nH = 1.0
    T_val = 8.0e3
    xi_CR_val = 1.0e-16
    ncell = 1

    x = np.array(
        [[1.0], [net.x_floor], [net.x_floor], [net.x_floor]],
        dtype=np.float64,
    )
    state = _make_state(
        x=x, nH=nH, T=T_val, Z_d=1.0, xi_CR=xi_CR_val,
        ncell=ncell,
    )

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

    i_HI = net.species.idx['HI']
    i_HII = net.species.idx['HII']
    i_H2 = net.species.idx['H2']

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
    ncell = 8
    x = np.zeros((4, ncell), dtype=np.float64)
    x[net.species.idx['HI'], :] = 1.0
    x[net.species.idx['HII'], :] = net.x_floor
    x[net.species.idx['H2'], :] = net.x_floor
    x[net.species.idx['electron'], :] = net.x_floor

    state = _make_state(
        x=x, nH=1.0, T=8.0e3, Z_d=1.0, xi_CR=1.0e-16, ncell=ncell,
    )

    C = np.zeros_like(x)
    D = np.zeros_like(x)
    dt = 3.0e10
    for _ in range(50):
        net.evaluate_CD(state, C, D)
        state.x[:] = (state.x + C * dt) / (1.0 + D * dt)
        net.closure(state)

    # Every cell got the same answer.
    for row in range(4):
        np.testing.assert_allclose(
            state.x[row, :], state.x[row, 0], rtol=1e-12, atol=0.0,
        )


def test_jacobian_raises_by_default():
    """NCRNetwork3 does not implement jacobian; default NetworkBase
    behavior is NotImplementedError, which the solver layer relies on.
    """
    net = NCRNetwork3()
    x = np.array([[0.5], [0.5], [0.0], [0.5]], dtype=np.float64)
    state = _make_state(x=x, T=1.0e4)
    out_J = np.zeros((4, 4, 1), dtype=np.float64)
    with pytest.raises(NotImplementedError):
        net.jacobian(state, out_J)


def test_fill_ghosts_is_noop():
    """The default fill_ghosts is a no-op; NCR3 has no derived
    species (GOW17's CHx etc. arrive in a different network).
    """
    net = NCRNetwork3()
    x = np.array([[0.5], [0.5], [0.0], [0.5]], dtype=np.float64)
    state = _make_state(x=x, T=1.0e4)
    x_before = state.x.copy()
    net.fill_ghosts(state)
    np.testing.assert_array_equal(state.x, x_before)
