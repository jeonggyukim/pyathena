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
from pyathena.chemistry.state import ChemState, DEFAULT_CHI_BANDS


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


# ---- chi_bands + chi_for ----
def test_chi_bands_default_is_3band_NCR_convention():
    """nfreq=3 picks up the canonical FUV / LW / EUV layout."""
    species = SpeciesSet.minimal_HI_HII_H2()
    r, nH, T = _radial_grid()
    state = ChemState.from_grid(r, nH, T, species, nfreq=3)
    assert state.chi_bands == DEFAULT_CHI_BANDS == ('FUV', 'LW', 'EUV')


def test_chi_for_returns_correct_band_view():
    """`chi_for('LW')` returns the row of chi corresponding to LW."""
    species = SpeciesSet.minimal_HI_HII_H2()
    r, nH, T = _radial_grid()
    state = ChemState.from_grid(r, nH, T, species, nfreq=3)
    # Write into the FUV row through the helper, observe via chi[0,:].
    state.chi_for('FUV')[:] = 1.7
    state.chi_for('LW')[:] = 0.5
    state.chi_for('EUV')[:] = 0.1
    np.testing.assert_allclose(state.chi[0, :], 1.7)
    np.testing.assert_allclose(state.chi[1, :], 0.5)
    np.testing.assert_allclose(state.chi[2, :], 0.1)


def test_chi_for_unknown_band_raises_keyerror():
    """An unknown band name raises KeyError, not ValueError."""
    species = SpeciesSet.minimal_HI_HII_H2()
    r, nH, T = _radial_grid()
    state = ChemState.from_grid(r, nH, T, species, nfreq=3)
    with pytest.raises(KeyError, match='not present'):
        state.chi_for('soft_xray')


def test_chi_bands_explicit_override():
    """A caller-supplied chi_bands tuple is taken verbatim."""
    species = SpeciesSet.minimal_HI_HII_H2()
    r, nH, T = _radial_grid()
    state = ChemState.from_grid(
        r, nH, T, species, nfreq=2,
        chi_bands=('soft_xray', 'hard_xray'),
    )
    assert state.chi_bands == ('soft_xray', 'hard_xray')
    assert state.chi.shape == (2, state.ncell)


def test_chi_bands_length_mismatch_rejected():
    """chi_bands must match nfreq exactly."""
    species = SpeciesSet.minimal_HI_HII_H2()
    r, nH, T = _radial_grid()
    with pytest.raises(ValueError, match='chi_bands'):
        ChemState.from_grid(
            r, nH, T, species, nfreq=3,
            chi_bands=('FUV', 'LW'),
        )


def test_chi_bands_falls_back_to_positional_for_non_default_nfreq():
    """Non-3 nfreq with no chi_bands kwarg gets `chi_0`, `chi_1`, ..."""
    species = SpeciesSet.minimal_HI_HII_H2()
    r, nH, T = _radial_grid()
    state = ChemState.from_grid(r, nH, T, species, nfreq=4)
    assert state.chi_bands == ('chi_0', 'chi_1', 'chi_2', 'chi_3')


# ---- Scratch dict ----
def test_scratch_dict_is_empty_by_default():
    species = SpeciesSet.minimal_HI_HII_H2()
    r, nH, T = _radial_grid()
    state = ChemState.from_grid(r, nH, T, species)
    assert state.scratch == {}


def test_alloc_scratch_and_get_scratch_roundtrip():
    """alloc_scratch returns the buffer and registers it under `name`."""
    species = SpeciesSet.minimal_HI_HII_H2()
    r, nH, T = _radial_grid()
    state = ChemState.from_grid(r, nH, T, species)
    buf = state.alloc_scratch('metal_CT', (4, state.ncell))
    assert buf.shape == (4, state.ncell)
    assert buf.dtype == np.float64
    # Read back via the keyed accessor and confirm identity (no copy).
    assert state.get_scratch('metal_CT') is buf


def test_get_scratch_unknown_raises_keyerror():
    species = SpeciesSet.minimal_HI_HII_H2()
    r, nH, T = _radial_grid()
    state = ChemState.from_grid(r, nH, T, species)
    with pytest.raises(KeyError, match='not allocated'):
        state.get_scratch('metal_CT')


def test_alloc_scratch_custom_dtype():
    species = SpeciesSet.minimal_HI_HII_H2()
    r, nH, T = _radial_grid()
    state = ChemState.from_grid(r, nH, T, species)
    buf = state.alloc_scratch('regime_tag', (state.ncell,), dtype=np.int8)
    assert buf.dtype == np.int8


def test_alloc_scratch_reallocate_overwrites():
    """Re-allocating an existing name replaces the buffer."""
    species = SpeciesSet.minimal_HI_HII_H2()
    r, nH, T = _radial_grid()
    state = ChemState.from_grid(r, nH, T, species)
    buf_a = state.alloc_scratch('foo', (3, state.ncell))
    buf_b = state.alloc_scratch('foo', (5, state.ncell))
    assert buf_a is not buf_b
    assert state.get_scratch('foo') is buf_b
    assert state.get_scratch('foo').shape == (5, state.ncell)


# ---- policy_versions stamping ----
def test_policy_versions_with_policies_records_qualname_and_version():
    """Supplying policy instances stamps `Class@version` strings."""
    species = SpeciesSet.minimal_HI_HII_H2()
    r, nH, T = _radial_grid()

    class _StubNetwork:
        __version__ = '2.0'
        walk_order = (('HI', 'HII'),)

        def allocate_scratch(self, state):
            state.alloc_scratch('foo', (state.ncell,))

    class _StubThermo:
        # No __version__ attribute -> falls back to sentinel.
        pass

    net = _StubNetwork()
    therm = _StubThermo()
    state = ChemState.from_grid(r, nH, T, species,
                                network=net, thermo=therm)

    # qualname includes the enclosing function for locally-defined
    # classes; we only check the class name and version are present.
    assert '_StubNetwork' in state.policy_versions['network']
    assert state.policy_versions['network'].endswith('@2.0')
    assert state.policy_versions['thermo'].endswith('@__none__')
    # walk_order populated from the supplied network.
    assert state.walk_order == (('HI', 'HII'),)
    # allocate_scratch hook fired.
    assert 'foo' in state.scratch


def test_policy_versions_optional_roles_not_added_when_none():
    """Without `cooling`/`opacity`/`radiation`, those keys do not appear."""
    species = SpeciesSet.minimal_HI_HII_H2()
    r, nH, T = _radial_grid()
    state = ChemState.from_grid(r, nH, T, species)
    assert set(state.policy_versions.keys()) == {'network', 'thermo'}
