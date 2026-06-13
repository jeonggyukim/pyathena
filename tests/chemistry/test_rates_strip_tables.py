"""Strip-vectorised rate-table accessors.

Covers the audit Phase 3 additions:

  PhotX.get_sigma_table(species_set, E)
  RecRate.get_rec_rate_table(species_set, T)
  CollIonRate.get_ci_rate_table(species_set, T)

The current implementations are thin wrappers that loop over the
species set and stack per-ion `get_*` results. The tests pin shape,
ordering, and per-ion agreement with the scalar getters so the C++
port (Phase D) can swap in a precomputed strip-shape table without
changing the contract.

`ChargeTransferRate` does NOT get a strip table here: CT rates carry
ion-pair structure (reactant + H I/H II collider) that is awkward to
vectorise alongside the photo/coll-ion/recombination strip and is
deferred to Phase 6 when the multi-ion sweep needs it.

Also pins the `interp_mode` argument plumbed through each class:
only `InterpMode.kExact` is implemented today; the table-based
modes (`kLogLog`, `kNqt1`, `kNqt2`) raise `NotImplementedError`
until Phase 3.5 lands.
"""
from __future__ import annotations

import numpy as np
import pytest

from pyathena.chemistry import InterpMode
from pyathena.chemistry.species import SpeciesSet
from pyathena.chemistry.rates.ci_rate import CollIonRate
from pyathena.chemistry.rates.photx import PhotX
from pyathena.chemistry.rates.rec_rate import RecRate


# Three-species NCR set: HI, HII, H2, electron. Small enough to
# enumerate the expected zero / non-zero rows explicitly.
@pytest.fixture(scope='module')
def species_set():
    return SpeciesSet.minimal_HI_HII_H2()


@pytest.fixture(scope='module')
def E_grid():
    return np.logspace(0.0, 4.0, 32)


@pytest.fixture(scope='module')
def T_grid():
    return np.logspace(2.0, 8.0, 25)


# ---------------------------------------------------------------------
# interp_mode plumbing: only kExact is implemented today.
# ---------------------------------------------------------------------

@pytest.mark.parametrize('cls', [PhotX, RecRate, CollIonRate])
def test_interp_mode_default_is_kExact(cls):
    obj = cls()
    assert obj.interp_mode == InterpMode.kExact


@pytest.mark.parametrize('cls', [PhotX, RecRate, CollIonRate])
@pytest.mark.parametrize('mode', [
    InterpMode.kLogLog, InterpMode.kNqt1, InterpMode.kNqt2,
])
def test_interp_mode_table_modes_not_implemented(cls, mode):
    with pytest.raises(NotImplementedError):
        cls(interp_mode=mode)


# ---------------------------------------------------------------------
# PhotX.get_sigma_table
# ---------------------------------------------------------------------

def test_photx_sigma_table_shape(species_set, E_grid):
    px = PhotX()
    sigma = px.get_sigma_table(species_set, E_grid)
    assert sigma.shape == (species_set.nspec, E_grid.size)


def test_photx_sigma_table_matches_per_ion(species_set, E_grid):
    px = PhotX()
    sigma = px.get_sigma_table(species_set, E_grid)
    for i, ion in enumerate(species_set.ions):
        if ion.element in ('e', 'H2'):
            # Strip-table convention: bare electron / H2 row is zero;
            # `get_sigma` itself would not be well-defined here.
            np.testing.assert_array_equal(sigma[i, :], 0.0)
            continue
        key = (int(ion.Z), int(ion.N))
        if key not in px._ion_idx:
            # Fully-stripped ions (e.g. H II at Z=1, N=0) have no
            # electron to photoionize; the strip-table row is zero by
            # design.
            np.testing.assert_array_equal(sigma[i, :], 0.0)
            continue
        expected = px.get_sigma(ion.Z, ion.N, E_grid)
        np.testing.assert_allclose(sigma[i, :], expected,
                                   rtol=0.0, atol=0.0,
                                   err_msg=f'row {i} ({ion.name})')


# ---------------------------------------------------------------------
# RecRate.get_rec_rate_table
# ---------------------------------------------------------------------

def test_rec_rate_table_shape(species_set, T_grid):
    rc = RecRate(caseB=True)
    rates = rc.get_rec_rate_table(species_set, T_grid)
    assert rates.shape == (species_set.nspec, T_grid.size)


def test_rec_rate_table_matches_per_ion(species_set, T_grid):
    rc = RecRate(caseB=True)
    rates = rc.get_rec_rate_table(species_set, T_grid)
    for i, ion in enumerate(species_set.ions):
        # Strip-table convention: only ions with charge >= 1
        # recombine; the rest (neutral atoms, H2, electron) stay 0.
        if ion.element in ('e', 'H2') or ion.charge <= 0:
            np.testing.assert_array_equal(rates[i, :], 0.0,
                                          err_msg=f'row {i} ({ion.name})')
            continue
        expected = rc.get_rec_rate(ion.Z, ion.N, T_grid)
        np.testing.assert_allclose(rates[i, :], expected,
                                   rtol=0.0, atol=0.0,
                                   err_msg=f'row {i} ({ion.name})')


def test_rec_rate_table_caseA_branch(species_set, T_grid):
    """Same coverage with `caseB=False` so the Z=1 branch flip is
    exercised."""
    rc = RecRate(caseB=False)
    rates = rc.get_rec_rate_table(species_set, T_grid)
    for i, ion in enumerate(species_set.ions):
        if ion.element in ('e', 'H2') or ion.charge <= 0:
            np.testing.assert_array_equal(rates[i, :], 0.0)
            continue
        expected = rc.get_rec_rate(ion.Z, ion.N, T_grid)
        np.testing.assert_allclose(rates[i, :], expected,
                                   rtol=0.0, atol=0.0,
                                   err_msg=f'row {i} ({ion.name})')


# ---------------------------------------------------------------------
# CollIonRate.get_ci_rate_table
# ---------------------------------------------------------------------

def test_ci_rate_table_shape(species_set, T_grid):
    ci = CollIonRate()
    rates = ci.get_ci_rate_table(species_set, T_grid)
    assert rates.shape == (species_set.nspec, T_grid.size)


def test_ci_rate_table_matches_per_ion(species_set, T_grid):
    ci = CollIonRate()
    rates = ci.get_ci_rate_table(species_set, T_grid)
    for i, ion in enumerate(species_set.ions):
        if ion.element in ('e', 'H2'):
            np.testing.assert_array_equal(rates[i, :], 0.0)
            continue
        key = (int(ion.Z), int(ion.N))
        if key not in ci._ion_idx:
            np.testing.assert_array_equal(rates[i, :], 0.0)
            continue
        expected = ci.get_ci_rate(ion.Z, ion.N, T_grid)
        np.testing.assert_allclose(rates[i, :], expected,
                                   rtol=0.0, atol=0.0,
                                   err_msg=f'row {i} ({ion.name})')


# ---------------------------------------------------------------------
# Strip tables on the ncr3_with_ghosts (9-species) layout. The
# `evolved_names + ghost_names` traversal is the canonical order the
# Phase 3 driver hands to the rate layer.
# ---------------------------------------------------------------------

@pytest.fixture(scope='module')
def ncr3_ghosts_set():
    return SpeciesSet.ncr3_with_ghosts()


def test_photx_strip_honours_evolved_ghost_partition(
        ncr3_ghosts_set, E_grid):
    px = PhotX()
    sigma = px.get_sigma_table(ncr3_ghosts_set, E_grid)
    assert sigma.shape == (ncr3_ghosts_set.nspec, E_grid.size)
    # The traversal order is evolved_names + ghost_names, which for
    # ncr3_with_ghosts equals the storage order in this set. Confirm
    # the strip rows agree with the per-ion getter on the
    # `species_set.ions[i]` instances.
    for i, ion in enumerate(ncr3_ghosts_set.ions):
        if ion.element in ('e', 'H2'):
            np.testing.assert_array_equal(sigma[i, :], 0.0)
            continue
        key = (int(ion.Z), int(ion.N))
        if key not in px._ion_idx:
            np.testing.assert_array_equal(sigma[i, :], 0.0)
            continue
        expected = px.get_sigma(ion.Z, ion.N, E_grid)
        np.testing.assert_allclose(sigma[i, :], expected,
                                   rtol=0.0, atol=0.0,
                                   err_msg=f'row {i} ({ion.name})')


def test_rec_rate_strip_honours_evolved_ghost_partition(
        ncr3_ghosts_set, T_grid):
    rc = RecRate()
    rates = rc.get_rec_rate_table(ncr3_ghosts_set, T_grid)
    assert rates.shape == (ncr3_ghosts_set.nspec, T_grid.size)


def test_ci_rate_strip_honours_evolved_ghost_partition(
        ncr3_ghosts_set, T_grid):
    ci = CollIonRate()
    rates = ci.get_ci_rate_table(ncr3_ghosts_set, T_grid)
    assert rates.shape == (ncr3_ghosts_set.nspec, T_grid.size)
