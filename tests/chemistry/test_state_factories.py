"""Unit tests for `pyathena.chemistry.state.ChemState.from_grid` and
the `ne` property.

Covers:

- Strip allocation from a 1-D radial grid + SpeciesSet (HI/HII/H2/e).
- `ne` property: zero for a neutral state, `nH` for a fully ionised
  state.
- `validate()` rejects a state whose payload arrays do not match the
  declared `(nspec, ncell)` shape.
"""
from __future__ import annotations

import numpy as np
import pytest

from pyathena.chemistry.species import SpeciesSet
from pyathena.chemistry.state import ChemState


def _radial_grid(ncell: int = 10):
    """Return a (r, nH, T) triple matching ncell."""
    r = np.linspace(0.1, 1.0, ncell)
    nH = np.full(ncell, 1.0)
    T = np.full(ncell, 8000.0)
    return r, nH, T


def test_from_grid_allocates_strip_shape():
    """from_grid produces correctly shaped arrays."""
    species = SpeciesSet.minimal_HI_HII_H2()
    r, nH, T = _radial_grid(10)
    state = ChemState.from_grid(r, nH, T, species, nfreq=3, ncol=2)

    assert state.ncell == 10
    assert state.nspec == 4
    assert state.x.shape == (4, 10)
    assert state.nH.shape == (10,)
    assert state.T.shape == (10,)
    assert state.T_dust.shape == (10,)
    assert state.Z_g.shape == (10,)
    assert state.Z_d.shape == (10,)
    assert state.chi.shape == (3, 10)
    assert state.N_col.shape == (2, 10)
    assert state.xi_CR.shape == (10,)
    assert state.dvdr.shape == (10,)
    assert state.dt_remaining.shape == (10,)

    # The neutral initialiser puts unity on the HI row.
    np.testing.assert_array_equal(state.x[0, :], 1.0)
    np.testing.assert_array_equal(state.x[1:, :], 0.0)


def test_from_grid_policy_versions_sentinel():
    """Concrete drivers fill these; the factory seeds sentinels."""
    species = SpeciesSet.minimal_HI_HII_H2()
    r, nH, T = _radial_grid()
    state = ChemState.from_grid(r, nH, T, species)
    assert state.policy_versions == {
        'network': '__none__', 'thermo': '__none__',
    }
    assert state.walk_order == ()


def test_from_grid_metallicity_broadcast():
    species = SpeciesSet.minimal_HI_HII_H2()
    r, nH, T = _radial_grid()
    state = ChemState.from_grid(r, nH, T, species, Z_g=0.3, Z_d=0.5)
    np.testing.assert_array_equal(state.Z_g, 0.3)
    np.testing.assert_array_equal(state.Z_d, 0.5)


def test_from_grid_validates_immediately():
    """A mismatched nH shape should be rejected at construction."""
    species = SpeciesSet.minimal_HI_HII_H2()
    r = np.linspace(0.1, 1.0, 10)
    nH_bad = np.full(9, 1.0)  # wrong size
    T = np.full(10, 8000.0)
    with pytest.raises(ValueError, match='nH'):
        ChemState.from_grid(r, nH_bad, T, species)


def test_ne_zero_for_neutral_state():
    """Fully neutral (HI only) state -> ne = 0 everywhere."""
    species = SpeciesSet.minimal_HI_HII_H2()
    r, nH, T = _radial_grid()
    state = ChemState.from_grid(r, nH, T, species)
    # Default: x_HI = 1, x_HII = x_H2 = x_e = 0 -> ne = 0.
    np.testing.assert_array_equal(state.ne, 0.0)


def test_ne_matches_nH_for_fully_ionised_state():
    """x_HII = 1, x_e = 1 -> ne = nH per cell."""
    species = SpeciesSet.minimal_HI_HII_H2()
    r, nH, T = _radial_grid()
    state = ChemState.from_grid(r, nH, T, species)
    # Flip to fully ionised.
    state.x[0, :] = 0.0       # HI
    state.x[1, :] = 1.0       # HII
    state.x[2, :] = 0.0       # H2
    state.x[3, :] = 1.0       # electron
    np.testing.assert_allclose(state.ne, state.nH)


def test_ne_partial_ionisation():
    """Per-cell ne = nH * x_HII for the canonical SpeciesSet."""
    species = SpeciesSet.minimal_HI_HII_H2()
    r, nH, T = _radial_grid()
    nH[:] = 2.5
    state = ChemState.from_grid(r, nH, T, species)
    x_HII = np.linspace(0.0, 0.5, state.ncell)
    state.x[1, :] = x_HII
    state.x[0, :] = 1.0 - x_HII
    state.x[3, :] = x_HII
    np.testing.assert_allclose(state.ne, 2.5 * x_HII)


def test_validate_rejects_mismatched_x_ncell():
    """Mutating x to a different ncell than the rest of the strip
    should fail validate()."""
    species = SpeciesSet.minimal_HI_HII_H2()
    r, nH, T = _radial_grid()
    state = ChemState.from_grid(r, nH, T, species)
    # x agrees with itself (nspec = x.shape[0]), but disagrees with
    # nH/T/etc. which all carry ncell=10.
    state.x = np.zeros((4, 9))
    with pytest.raises(ValueError, match='x shape'):
        state.validate()


def test_validate_rejects_mismatched_T_shape():
    species = SpeciesSet.minimal_HI_HII_H2()
    r, nH, T = _radial_grid()
    state = ChemState.from_grid(r, nH, T, species)
    state.T = np.zeros(9)
    with pytest.raises(ValueError, match='T shape'):
        state.validate()


def test_validate_rejects_mismatched_chi_shape():
    species = SpeciesSet.minimal_HI_HII_H2()
    r, nH, T = _radial_grid()
    state = ChemState.from_grid(r, nH, T, species, nfreq=3)
    state.chi = np.zeros((3, 9))   # wrong ncell on last axis
    with pytest.raises(ValueError, match='chi'):
        state.validate()


def test_validate_rejects_non_finite_T():
    species = SpeciesSet.minimal_HI_HII_H2()
    r, nH, T = _radial_grid()
    state = ChemState.from_grid(r, nH, T, species)
    state.T[3] = np.nan
    with pytest.raises(ValueError, match='T contains'):
        state.validate()


def test_from_grid_carries_grid_coordinate():
    """`r` is forwarded as a non-schema attribute for plotting."""
    species = SpeciesSet.minimal_HI_HII_H2()
    r, nH, T = _radial_grid()
    state = ChemState.from_grid(r, nH, T, species)
    np.testing.assert_array_equal(state.r, r)
    assert state.A_He == pytest.approx(0.0955)
