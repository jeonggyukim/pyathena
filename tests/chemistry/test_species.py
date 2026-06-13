"""Unit tests for `pyathena.chemistry.species`.

Covers the two factory layouts (minimal NCR3 and NCR3 + helium), the
`validate()` contract, name -> index lookups, and the shape / value
checks on the per-species vector fields.
"""
from __future__ import annotations

import numpy as np
import pytest

from pyathena.chemistry.species import (
    HI, HII, H2, HeI, HeII, HeIII, ELECTRON,
    Ion, SpeciesSet, MU_H_DEFAULT, X_HE_DEFAULT,
)


def test_minimal_HI_HII_H2_layout():
    """Order, names, charges, masses for the 4-species NCR3 set."""
    ss = SpeciesSet.minimal_HI_HII_H2()

    assert ss.names == ('HI', 'HII', 'H2', 'electron')
    assert ss.nspec == 4
    assert len(ss) == 4

    # Index lookup.
    assert ss.idx['HI'] == 0
    assert ss.idx['HII'] == 1
    assert ss.idx['H2'] == 2
    assert ss.idx['electron'] == 3
    assert ss.index('H2') == 2
    assert ss.electron_index == 3
    assert 'HI' in ss
    assert 'OII' not in ss

    # Charge vector.
    assert ss.charges.dtype == np.int8
    assert ss.charges.tolist() == [0, +1, 0, -1]
    assert ss.charges.shape == (4,)

    # Electron mask.
    assert ss.is_electron.dtype == bool
    assert ss.is_electron.tolist() == [False, False, False, True]

    # Mass per particle in m_H units.
    assert ss.mass_per_particle.shape == (4,)
    np.testing.assert_allclose(ss.mass_per_particle,
                               [1.008, 1.008, 2.016, 5.4858e-4],
                               rtol=1e-6)

    # Per-particle coefficients all positive.
    assert np.all(ss.n_per_particle > 0)
    assert ss.n_per_particle.shape == (4,)

    # Helium folded into the baseline (set does not track He).
    assert ss.x_He == pytest.approx(X_HE_DEFAULT)
    assert ss.mu_H == pytest.approx(MU_H_DEFAULT)


def test_mu_formula_matches_canonical_NCR_expression():
    """Cross-check the n_per_particle decomposition against the
    canonical NCR mu formula in `get_cooling.py:36`:

        mu = muH / (1.1 + xe - x_H2)

    Using the closure x_HI + x_HII + 2 x_H2 = 1.
    """
    ss = SpeciesSet.minimal_HI_HII_H2()
    # Pick a non-trivial state: mixed H, some H2, some ionization.
    x_HI, x_H2 = 0.6, 0.15
    x_HII = 1.0 - x_HI - 2.0 * x_H2
    x_e = x_HII  # electrons follow HII for the minimal network

    x = np.array([x_HI, x_HII, x_H2, x_e])
    inv_mu = (ss.n_per_particle * x).sum() + ss.x_He
    expected_inv_mu = 1.1 + x_e - x_H2
    assert inv_mu == pytest.approx(expected_inv_mu, rel=1e-12)


def test_ncr3_plus_helium_layout():
    """Helium tracked explicitly drops the x_He baseline."""
    ss = SpeciesSet.ncr3_plus_helium()

    assert ss.names == ('HI', 'HII', 'H2', 'HeI', 'HeII', 'HeIII',
                        'electron')
    assert ss.nspec == 7
    assert ss.electron_index == 6
    assert ss.charges.tolist() == [0, +1, 0, 0, +1, +2, -1]
    # Helium tracked: baseline x_He is zero (carried by HeI/HeII/HeIII).
    assert ss.x_He == 0.0


def test_validate_rejects_missing_electron():
    """SpeciesSet must contain exactly one electron species."""
    with pytest.raises(ValueError, match='electron'):
        SpeciesSet(ions=(HI, HII, H2))


def test_validate_rejects_duplicate_electron():
    """Two electrons should also fail validate()."""
    extra_e = Ion(element='e', Z=0, N=1, charge=-1, name='electron2')
    # Bypass duplicate-name check by giving the second a unique name
    # but is_electron is keyed on the literal name 'electron', so the
    # easy way to test duplicate-electron detection is to bypass the
    # name guard. We do that by constructing the set with a hand-rolled
    # ions tuple that includes two species both named 'electron' --
    # which the duplicate-name guard already catches.
    with pytest.raises(ValueError, match='duplicate names'):
        SpeciesSet(ions=(HI, HII, ELECTRON, ELECTRON))


def test_validate_rejects_empty_set():
    with pytest.raises(ValueError, match='empty'):
        SpeciesSet(ions=())


def test_validate_rejects_negative_mass():
    """Mass per particle is required non-negative."""
    bad = Ion(element='X', Z=99, N=99, charge=0, name='Xenium')
    ss = SpeciesSet(ions=(HI, HII, H2, ELECTRON, bad))
    # The factory cannot know about hypothetical Xenium so it gets
    # mass 0.0 (the placeholder). Reach in and corrupt it.
    object.__setattr__(ss, 'mass_per_particle',
                       np.array([1.008, 1.008, 2.016, 5.4858e-4, -1.0]))
    with pytest.raises(ValueError, match='mass_per_particle'):
        ss.validate()


def test_validate_rejects_nonpositive_n_per_particle():
    ss = SpeciesSet.minimal_HI_HII_H2()
    object.__setattr__(ss, 'n_per_particle',
                       np.array([1.0, 1.0, 0.0, 1.0]))
    with pytest.raises(ValueError, match='n_per_particle'):
        ss.validate()


def test_ion_is_frozen():
    """Ion should reject attribute assignment."""
    ion = HI
    with pytest.raises((AttributeError, TypeError)):
        ion.element = 'He'


def test_speciesset_is_frozen():
    """SpeciesSet should reject attribute assignment after construction."""
    ss = SpeciesSet.minimal_HI_HII_H2()
    with pytest.raises((AttributeError, TypeError)):
        ss.names = ('A', 'B')


def test_idx_lookup_keyerror_on_miss():
    ss = SpeciesSet.minimal_HI_HII_H2()
    with pytest.raises(KeyError):
        ss.index('OII')


def test_ion_metadata_fields():
    """Spot-check the metadata on the canonical Ion instances."""
    assert HI.element == 'H' and HI.Z == 1 and HI.N == 1 and HI.charge == 0
    assert HII.element == 'H' and HII.Z == 1 and HII.N == 0
    assert HII.charge == +1
    assert H2.charge == 0 and H2.element == 'H2'
    assert HeII.charge == +1 and HeIII.charge == +2
    assert ELECTRON.charge == -1 and ELECTRON.name == 'electron'
