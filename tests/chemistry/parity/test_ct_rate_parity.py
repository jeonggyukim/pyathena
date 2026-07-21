"""Parity test: `pyathena.chemistry.rates.ct_rate` matches the frozen
`pyathena.microphysics.ct_rate` reference, bit-for-bit at rtol=1e-12.

Phase 1 of the chemistry rewrite (see `tigris-notes/docs-claude/
pyathena/chemistry-rewrite-plan.md`) copies the leaf rate module
verbatim, adjusting only the data-file path lookup. The numerical
behavior must not change, so this test instantiates one of each and
compares every public rate method on a representative (Z, N, T_grid)
catalog covering the followed-element list (H, He, C, N, O, Ne, S).
"""
from __future__ import annotations

import numpy as np
import pytest

from pyathena.chemistry import _parity
from pyathena.chemistry.rates.ct_rate import (
    ChargeTransferRate as ChargeTransferRateNew,
)
from pyathena.microphysics.ct_rate import (
    ChargeTransferRate as ChargeTransferRateOld,
)


# Reactant-indexed (Z, N) catalogs. Covers the followed-element set
# H, He, C, N, O, Ne, S. Each entry exercises a different code path
# inside `get_ct_*_rate`.
#
# CT_ION_CATALOG: neutral or low-charge reactants. (8,8) hits the
# Draine 2011 OI exception, (12,12) and (14,14) hit the MgI/SiI
# special cases, the rest hit the Cloudy table path. (26,21) probes
# the q > 3 zero-return fallback.
CT_ION_CATALOG = [
    (1, 1, "H I"),
    (6, 6, "C I"),
    (7, 7, "N I"),
    (8, 8, "O I (Draine11 exception)"),
    (10, 10, "Ne I"),
    (16, 16, "S I (placeholder)"),
    (12, 12, "Mg I (special)"),
    (14, 14, "Si I (special)"),
    (8, 7, "O II"),
    (26, 21, "Fe^5+ (q>3 zero)"),
]

# CT_REC_CATALOG: ionized reactants. (8,7) hits the Draine 2011 OII
# exception, (26,21) probes the q>4 Dalgarno fallback, rest hit the
# Cloudy table path. (1,1) probes the q<1 zero-return guard.
CT_REC_CATALOG = [
    (1, 1, "H I (q<1 zero)"),
    (2, 1, "He II"),
    (6, 5, "C II"),
    (7, 6, "N II"),
    (8, 7, "O II (Draine11 exception)"),
    (10, 9, "Ne II"),
    (16, 15, "S II"),
    (26, 21, "Fe^5+ (Dalgarno)"),
]

# UGA charge-exchange recombination / ionization tables key off the
# H+ collider. Pick a few entries that exist in `ctr_hyd.dat` /
# `cti_hyd.dat`.
CT_UGA_REC_CATALOG = [
    (6, 5, "C II rec UGA"),
    (7, 6, "N II rec UGA"),
    (8, 7, "O II rec UGA"),
]

CT_UGA_ION_CATALOG = [
    (6, 6, "C I ion UGA"),
    (7, 7, "N I ion UGA"),
    (8, 8, "O I ion UGA"),
]


@pytest.fixture(scope="module")
def ct_old():
    return ChargeTransferRateOld()


@pytest.fixture(scope="module")
def ct_new():
    return ChargeTransferRateNew()


@pytest.fixture(scope="module")
def T_grid():
    # Cover the HII-region and warm-neutral ranges.
    return np.logspace(2.0, 6.0, 41)


# ---------------------------------------------------------------------
# Cloudy-table paths.
# ---------------------------------------------------------------------

@pytest.mark.parametrize("Z,N,label", CT_ION_CATALOG)
def test_parity_get_ct_ion_rate(ct_old, ct_new, T_grid, Z, N, label):
    out_old, out_new = _parity.run_both(
        ct_old.get_ct_ion_rate, ct_new.get_ct_ion_rate, Z, N, T_grid)
    _parity.assert_close(out_old, out_new, rtol=1e-12, atol=0.0,
                         label=f"get_ct_ion_rate {label}")


@pytest.mark.parametrize("Z,N,label", CT_REC_CATALOG)
def test_parity_get_ct_rec_rate(ct_old, ct_new, T_grid, Z, N, label):
    out_old, out_new = _parity.run_both(
        ct_old.get_ct_rec_rate, ct_new.get_ct_rec_rate, Z, N, T_grid)
    _parity.assert_close(out_old, out_new, rtol=1e-12, atol=0.0,
                         label=f"get_ct_rec_rate {label}")


# ---------------------------------------------------------------------
# UGA-table paths.
# ---------------------------------------------------------------------

@pytest.mark.parametrize("Z,N,label", CT_UGA_REC_CATALOG)
def test_parity_get_ct_rec_rate_uga(ct_old, ct_new, T_grid, Z, N, label):
    out_old, out_new = _parity.run_both(
        ct_old.get_ct_rec_rate_uga, ct_new.get_ct_rec_rate_uga, Z, N, T_grid)
    _parity.assert_close(out_old, out_new, rtol=1e-12, atol=0.0,
                         label=f"get_ct_rec_rate_uga {label}")


@pytest.mark.parametrize("Z,N,label", CT_UGA_ION_CATALOG)
def test_parity_get_ct_ion_rate_uga(ct_old, ct_new, T_grid, Z, N, label):
    out_old, out_new = _parity.run_both(
        ct_old.get_ct_ion_rate_uga, ct_new.get_ct_ion_rate_uga, Z, N, T_grid)
    _parity.assert_close(out_old, out_new, rtol=1e-12, atol=0.0,
                         label=f"get_ct_ion_rate_uga {label}")


# ---------------------------------------------------------------------
# Static helpers (Draine 2011 OI fits, KF96 OII fit, MgI/SiI specials).
# ---------------------------------------------------------------------

def test_parity_get_ct_rec_HI_OII(T_grid):
    out_old, out_new = _parity.run_both(
        ChargeTransferRateOld.get_ct_rec_HI_OII,
        ChargeTransferRateNew.get_ct_rec_HI_OII,
        T_grid)
    _parity.assert_close(out_old, out_new, rtol=1e-12, atol=0.0,
                         label="get_ct_rec_HI_OII")


def test_parity_get_ct_ion_HII_OI(T_grid):
    out_old, out_new = _parity.run_both(
        ChargeTransferRateOld.get_ct_ion_HII_OI,
        ChargeTransferRateNew.get_ct_ion_HII_OI,
        T_grid)
    _parity.assert_close(out_old, out_new, rtol=1e-12, atol=0.0,
                         label="get_ct_ion_HII_OI")


def test_parity_get_ct_rec_HI_OII_Draine11_sum(T_grid):
    out_old, out_new = _parity.run_both(
        ChargeTransferRateOld.get_ct_rec_HI_OII_Draine11,
        ChargeTransferRateNew.get_ct_rec_HI_OII_Draine11,
        T_grid)
    _parity.assert_close(out_old, out_new, rtol=1e-12, atol=0.0,
                         label="get_ct_rec_HI_OII_Draine11 sum")


def test_parity_get_ct_rec_HI_OII_Draine11_perJ(T_grid):
    out_old = ChargeTransferRateOld.get_ct_rec_HI_OII_Draine11(T_grid, sum=False)
    out_new = ChargeTransferRateNew.get_ct_rec_HI_OII_Draine11(T_grid, sum=False)
    _parity.assert_close(out_old, out_new, rtol=1e-12, atol=0.0,
                         label="get_ct_rec_HI_OII_Draine11 per-J")


def test_parity_get_ct_ion_HII_OI_Draine11_sum(T_grid):
    out_old, out_new = _parity.run_both(
        ChargeTransferRateOld.get_ct_ion_HII_OI_Draine11,
        ChargeTransferRateNew.get_ct_ion_HII_OI_Draine11,
        T_grid)
    _parity.assert_close(out_old, out_new, rtol=1e-12, atol=0.0,
                         label="get_ct_ion_HII_OI_Draine11 sum")


def test_parity_get_ct_ion_HII_OI_Draine11_perJ(T_grid):
    out_old = ChargeTransferRateOld.get_ct_ion_HII_OI_Draine11(T_grid, sum=False)
    out_new = ChargeTransferRateNew.get_ct_ion_HII_OI_Draine11(T_grid, sum=False)
    _parity.assert_close(out_old, out_new, rtol=1e-12, atol=0.0,
                         label="get_ct_ion_HII_OI_Draine11 per-J")


def test_parity_get_ct_ion_MgI_HII(T_grid):
    out_old, out_new = _parity.run_both(
        ChargeTransferRateOld.get_ct_ion_MgI_HII,
        ChargeTransferRateNew.get_ct_ion_MgI_HII,
        T_grid)
    _parity.assert_close(out_old, out_new, rtol=1e-12, atol=0.0,
                         label="get_ct_ion_MgI_HII")


def test_parity_get_ct_ion_SiI_HII(T_grid):
    out_old, out_new = _parity.run_both(
        ChargeTransferRateOld.get_ct_ion_SiI_HII,
        ChargeTransferRateNew.get_ct_ion_SiI_HII,
        T_grid)
    _parity.assert_close(out_old, out_new, rtol=1e-12, atol=0.0,
                         label="get_ct_ion_SiI_HII")


# ---------------------------------------------------------------------
# Loaded-table attributes: assert the data ports loaded the same
# numbers, so any future divergence in the table-load path shows up
# even when the rate-evaluator code path agrees by coincidence.
# ---------------------------------------------------------------------

def test_parity_cloudy_table_arrays(ct_old, ct_new):
    for attr in ("Z1", "Z2", "N1", "N2",
                 "a1", "b1", "c1", "d1", "Tmin1", "Tmax1",
                 "dE_rate1", "dE_heat1",
                 "a2", "b2", "c2", "d2", "Tmin2", "Tmax2", "dE_heat2"):
        old = getattr(ct_old, attr)
        new = getattr(ct_new, attr)
        np.testing.assert_array_equal(
            new, old,
            err_msg=f"cloudy table attribute {attr} differs")


def test_parity_uga_table_frames(ct_old, ct_new):
    # Compare the relevant numeric columns of the two pandas frames.
    for frame_attr in ("dfi", "dfr"):
        old = getattr(ct_old, frame_attr)
        new = getattr(ct_new, frame_attr)
        assert list(new.columns) == list(old.columns), (
            f"{frame_attr} columns differ: "
            f"old={list(old.columns)} new={list(new.columns)}")
        for col in old.columns:
            if old[col].dtype.kind in ("f", "i"):
                np.testing.assert_array_equal(
                    new[col].to_numpy(), old[col].to_numpy(),
                    err_msg=f"{frame_attr}[{col!r}] differs")
            else:
                assert list(new[col]) == list(old[col]), (
                    f"{frame_attr}[{col!r}] differs")
