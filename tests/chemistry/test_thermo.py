"""Unit tests for `pyathena.chemistry.thermo.NCRThermo`.

Covers:

- mu on canonical compositions (fully ionized H+He, fully neutral
  atomic H+He, fully molecular H2+He).
- T_to_e / e_to_T round-trip consistency.
- pressure agreement with a direct n_total * k_B * T evaluation.
- gamma class attribute matches the Tigris hydro choice (5/3).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pytest

from pyathena.chemistry.state import ChemState
from pyathena.chemistry.diagnostics import SolverDiag
from pyathena.chemistry.thermo.base import K_B_CGS, M_H_CGS, ThermoPolicy
from pyathena.chemistry.thermo.ncr import NCRThermo


# ---------------------------------------------------------------------------
# Helper: minimal SpeciesSet stand-in.
#
# A real `SpeciesSet` arrives in Phase 2. For these unit tests we only
# need two attributes the NCRThermo policy reads: `charges` (per-species
# integer charge) and `h2_index` (row index of H2 in `state.x`, or
# `None` if H2 is not tracked).
# ---------------------------------------------------------------------------
@dataclass
class _MiniSpecies:
    charges: np.ndarray
    h2_index: Optional[int]
    names: Tuple[str, ...]


def _make_state(species: _MiniSpecies,
                x: np.ndarray,
                nH: float = 1.0,
                T: float = 1.0e4) -> ChemState:
    """Build a minimal ChemState carrying just enough payload for the
    thermo policy. Other fields use zero-filled placeholders.
    """
    ncell = x.shape[1]
    return ChemState(
        species=species,
        policy_versions={},
        walk_order=(),
        x=x,
        nH=np.full(ncell, nH),
        T=np.full(ncell, T),
        T_dust=np.zeros(ncell),
        Z_g=np.ones(ncell),
        Z_d=np.ones(ncell),
        chi=np.zeros((1, ncell)),
        xi_CR=np.zeros(ncell),
        N_col=np.zeros((1, ncell)),
        dvdr=np.zeros(ncell),
        dt=0.0,
        dt_remaining=np.zeros(ncell),
        t=0.0,
        diag=SolverDiag(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

A_HE = 0.0955


def test_subclasses_thermopolicy():
    """NCRThermo is a ThermoPolicy."""
    assert issubclass(NCRThermo, ThermoPolicy)


def test_gamma_is_5_over_3():
    """Tigris hydro runs with gamma = 5/3; NCRThermo inherits that."""
    assert NCRThermo.gamma == pytest.approx(5.0 / 3.0)
    assert NCRThermo().gamma == pytest.approx(5.0 / 3.0)


def test_mu_fully_ionized_h_plus_he():
    """Fully ionised H+He plasma: x_HII = 1, He doubly ionised.

    The denominator of mu is `1 + A_He - x_H2 + x_e`, with
    x_e = 1 + 2 * A_He for fully ionised He.
    """
    # Species order: HI, HII, HeI, HeII, HeIII, H2.
    species = _MiniSpecies(
        charges=np.array([0, 1, 0, 1, 2, 0], dtype=float),
        h2_index=5,
        names=('HI', 'HII', 'HeI', 'HeII', 'HeIII', 'H2'),
    )
    x = np.array([
        [0.0],          # HI
        [1.0],          # HII
        [0.0],          # HeI
        [0.0],          # HeII
        [A_HE],         # HeIII (relative to nH)
        [0.0],          # H2
    ])
    state = _make_state(species, x)

    thermo = NCRThermo(A_He=A_HE)
    out = np.empty(state.ncell)
    thermo.mu(state, out)

    # Expected: mu = (1 + 4*A_He) / (1 + A_He - 0 + (1 + 2*A_He))
    mu_expected = (1.0 + 4.0 * A_HE) / (1.0 + A_HE + 1.0 + 2.0 * A_HE)
    assert out[0] == pytest.approx(mu_expected, rel=1e-12)
    # Sanity range for a hot ionised plasma: 0.6 to 0.63.
    assert 0.59 < out[0] < 0.64


def test_mu_fully_neutral_h_plus_he():
    """Fully neutral atomic H+He: x_HI = 1, He neutral, no electrons.

    Expected mu ~ 1.27 (canonical neutral ISM value).
    """
    species = _MiniSpecies(
        charges=np.array([0, 1, 0, 1, 2, 0], dtype=float),
        h2_index=5,
        names=('HI', 'HII', 'HeI', 'HeII', 'HeIII', 'H2'),
    )
    x = np.array([
        [1.0],          # HI
        [0.0],          # HII
        [A_HE],         # HeI
        [0.0],          # HeII
        [0.0],          # HeIII
        [0.0],          # H2
    ])
    state = _make_state(species, x)

    thermo = NCRThermo(A_He=A_HE)
    out = np.empty(state.ncell)
    thermo.mu(state, out)

    # Expected: mu = (1 + 4*A_He) / (1 + A_He - 0 + 0)
    mu_expected = (1.0 + 4.0 * A_HE) / (1.0 + A_HE)
    assert out[0] == pytest.approx(mu_expected, rel=1e-12)
    # Canonical "neutral atomic ISM" value is 1.27.
    assert out[0] == pytest.approx(1.27, rel=0.01)


def test_mu_fully_molecular_h2_plus_he():
    """Fully molecular gas: x_H2 = 0.5 (half occupancy of H sites),
    He neutral, no free electrons.

    Expected mu ~ 2.35.
    """
    species = _MiniSpecies(
        charges=np.array([0, 1, 0, 1, 2, 0], dtype=float),
        h2_index=5,
        names=('HI', 'HII', 'HeI', 'HeII', 'HeIII', 'H2'),
    )
    x = np.array([
        [0.0],          # HI
        [0.0],          # HII
        [A_HE],         # HeI
        [0.0],          # HeII
        [0.0],          # HeIII
        [0.5],          # H2 (every H atom is in H2 -> x_H2 = 0.5)
    ])
    state = _make_state(species, x)

    thermo = NCRThermo(A_He=A_HE)
    out = np.empty(state.ncell)
    thermo.mu(state, out)

    # Expected: mu = (1 + 4*A_He) / (1 + A_He - 0.5 + 0)
    mu_expected = (1.0 + 4.0 * A_HE) / (1.0 + A_HE - 0.5)
    assert out[0] == pytest.approx(mu_expected, rel=1e-12)
    # Canonical "fully molecular" value at x_He = 0.1 is 2.35; at
    # A_He = 0.0955 the prediction drops to ~ 2.32.
    assert out[0] == pytest.approx(2.32, rel=0.02)


def test_T_to_e_round_trip():
    """T -> e -> T agrees to machine precision."""
    # Pick a partially ionised mix so mu is non-trivial.
    species = _MiniSpecies(
        charges=np.array([0, 1, 0, 1, 2, 0], dtype=float),
        h2_index=5,
        names=('HI', 'HII', 'HeI', 'HeII', 'HeIII', 'H2'),
    )
    x = np.array([
        [0.3, 0.0, 1.0],   # HI
        [0.4, 1.0, 0.0],   # HII
        [A_HE, 0.0, A_HE], # HeI
        [0.0, 0.0, 0.0],   # HeII
        [0.0, A_HE, 0.0],  # HeIII
        [0.15, 0.0, 0.0],  # H2
    ])
    T_in = np.array([1.0e4, 1.0e6, 1.0e2])
    state = _make_state(species, x, T=1.0)
    state.T[:] = T_in

    thermo = NCRThermo(A_He=A_HE)
    e = np.empty(state.ncell)
    thermo.T_to_e(state, e)

    T_back = np.empty(state.ncell)
    thermo.e_to_T(state, e, T_back)
    np.testing.assert_allclose(T_back, T_in, rtol=1e-12, atol=0.0)


def test_T_to_e_value_matches_closed_form():
    """T_to_e returns e = k_B T / ((gamma - 1) * mu * m_H).

    Cross-checks against an explicit numpy evaluation that does not
    share any intermediate buffer with NCRThermo.
    """
    species = _MiniSpecies(
        charges=np.array([0, 1, 0, 1, 2, 0], dtype=float),
        h2_index=5,
        names=('HI', 'HII', 'HeI', 'HeII', 'HeIII', 'H2'),
    )
    x = np.array([
        [0.5],
        [0.4],
        [A_HE - 0.01],
        [0.005],
        [0.005],
        [0.05],
    ])
    state = _make_state(species, x, T=8.0e3)

    thermo = NCRThermo(A_He=A_HE)
    out = np.empty(state.ncell)
    thermo.T_to_e(state, out)

    # Closed form: mu = (1 + 4 A_He) / (1 + A_He - x_H2 + x_e)
    x_e_ref = (np.asarray(species.charges) @ x).item()
    x_h2_ref = x[species.h2_index, 0]
    mu_ref = (1.0 + 4.0 * A_HE) / (1.0 + A_HE - x_h2_ref + x_e_ref)
    gamma = NCRThermo.gamma
    e_ref = K_B_CGS * 8.0e3 / ((gamma - 1.0) * mu_ref * M_H_CGS)
    assert out[0] == pytest.approx(e_ref, rel=1e-12)


def test_pressure_matches_nH_kT_n_total():
    """`pressure(state)` equals nH * (1 + A_He - x_H2 + x_e) * k_B * T,
    evaluated independently.
    """
    species = _MiniSpecies(
        charges=np.array([0, 1, 0, 1, 2, 0], dtype=float),
        h2_index=5,
        names=('HI', 'HII', 'HeI', 'HeII', 'HeIII', 'H2'),
    )
    rng = np.random.default_rng(seed=12345)
    ncell = 8
    # Random non-negative abundances, no conservation constraint imposed
    # since we just check the formula, not physics.
    x = rng.uniform(0.0, 1.0, size=(6, ncell))
    nH_arr = rng.uniform(0.1, 1.0e3, size=ncell)
    T_arr = rng.uniform(50.0, 1.0e6, size=ncell)
    state = _make_state(species, x)
    state.nH[:] = nH_arr
    state.T[:] = T_arr

    thermo = NCRThermo(A_He=A_HE)
    out = np.empty(ncell)
    thermo.pressure(state, out)

    x_e_ref = species.charges @ x
    x_h2_ref = x[species.h2_index, :]
    p_ref = nH_arr * (1.0 + A_HE - x_h2_ref + x_e_ref) * K_B_CGS * T_arr
    np.testing.assert_allclose(out, p_ref, rtol=1e-12, atol=0.0)


def test_mu_with_no_h2_index_treats_x_h2_as_zero():
    """If `species.h2_index is None`, the policy should treat x_H2 = 0
    (i.e., no H2 chemistry tracked). Useful for ion-only networks.
    """
    species = _MiniSpecies(
        charges=np.array([0, 1, 0, 1, 2], dtype=float),
        h2_index=None,
        names=('HI', 'HII', 'HeI', 'HeII', 'HeIII'),
    )
    x = np.array([
        [0.0],
        [1.0],
        [0.0],
        [0.0],
        [A_HE],
    ])
    state = _make_state(species, x)
    thermo = NCRThermo(A_He=A_HE)
    out = np.empty(state.ncell)
    thermo.mu(state, out)
    mu_expected = (1.0 + 4.0 * A_HE) / (1.0 + A_HE + 1.0 + 2.0 * A_HE)
    assert out[0] == pytest.approx(mu_expected, rel=1e-12)


def test_a_he_constructor_argument_is_used():
    """Non-default A_He must propagate into both mu_hyd and the
    denominator.
    """
    species = _MiniSpecies(
        charges=np.array([0, 1, 0, 1, 2, 0], dtype=float),
        h2_index=5,
        names=('HI', 'HII', 'HeI', 'HeII', 'HeIII', 'H2'),
    )
    x = np.array([
        [1.0], [0.0], [0.2], [0.0], [0.0], [0.0]
    ])
    state = _make_state(species, x)
    A_test = 0.20
    thermo = NCRThermo(A_He=A_test)
    out = np.empty(state.ncell)
    thermo.mu(state, out)
    # x_He = 0.2 < A_test only enters via charges (none ionised here).
    expected = (1.0 + 4.0 * A_test) / (1.0 + A_test - 0.0 + 0.0)
    assert out[0] == pytest.approx(expected, rel=1e-12)
