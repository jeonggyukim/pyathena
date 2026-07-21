"""Parity test: `pyathena.chemistry.rates.rec_rate.RecRate` vs
`pyathena.microphysics.rec_rate.RecRate`.

Phase 1 strangler port. The new module is supposed to be a verbatim
copy of the old one with the data-file path lookup adjusted for the
new package location (the .dat files themselves are NOT moved).
This test exercises both `RecRate(caseB=True)` and
`RecRate(caseB=False)` across the followed-ion catalog (H, He, C, N,
O, Ne, S) and verifies bit-for-bit agreement at `rtol=1e-12`.

Failure here means either:
  (a) the path-lookup change is reading a different .dat file than
      the old module, OR
  (b) someone touched the numerics during the port.
"""
from __future__ import annotations

import numpy as np
import pytest

from pyathena.chemistry import _parity
from pyathena.chemistry.rates.rec_rate import RecRate as RecRateNew
from pyathena.microphysics.rec_rate import RecRate as RecRateOld


# (Z, N_initial, label) -- (Z, N) = ion BEFORE recombination.
# Covers the followed-element list H, He, C, N, O, Ne, S plus the
# fully-stripped ions (N=0) which exercise the "no DR" branch.
REC_CATALOG = [
    (1,  0,  "H II"),
    (2,  1,  "He II"),
    (2,  0,  "He III"),
    (6,  5,  "C II"),
    (6,  4,  "C III"),
    (6,  1,  "C VI"),
    (7,  6,  "N II"),
    (7,  5,  "N III"),
    (8,  7,  "O II"),
    (8,  6,  "O III"),
    (8,  5,  "O IV"),
    (10, 9,  "Ne II"),
    (10, 8,  "Ne III"),
    (16, 15, "S II"),
    (16, 14, "S III"),
]


@pytest.fixture(scope="module")
def rc_pair_caseB():
    return RecRateOld(caseB=True), RecRateNew(caseB=True)


@pytest.fixture(scope="module")
def rc_pair_caseA():
    return RecRateOld(caseB=False), RecRateNew(caseB=False)


@pytest.fixture(scope="module")
def T_grid():
    # Wide range: cold molecular gas through coronal.
    return np.logspace(2.0, 8.0, 25)


# ---------------------------------------------------------------------
# Raw data tables must match byte-for-byte after `_read_data` runs.
# ---------------------------------------------------------------------

@pytest.mark.parametrize("attr", [
    "Zd", "Nd", "Md", "Wd", "Cd", "Ed", "nd",
    "Zr", "Nr", "Mr", "Wr", "Ar", "Br", "T0r", "T1r", "Cr", "T2r", "modr",
])
def test_data_tables_identical(rc_pair_caseB, attr):
    """The new module is supposed to read the SAME .dat files as the
    old one. Verify directly: the parsed arrays must be identical.
    """
    rc_old, rc_new = rc_pair_caseB
    np.testing.assert_array_equal(
        getattr(rc_new, attr), getattr(rc_old, attr),
        err_msg=f"data-table attribute '{attr}' diverged after port")


# ---------------------------------------------------------------------
# Per-call parity: get_rr_rate / get_dr_rate / get_rec_rate.
# ---------------------------------------------------------------------

@pytest.mark.parametrize("Z,N,label", REC_CATALOG)
def test_parity_get_rr_rate(rc_pair_caseB, T_grid, Z, N, label):
    rc_old, rc_new = rc_pair_caseB
    out_old, out_new = _parity.run_both(
        rc_old.get_rr_rate, rc_new.get_rr_rate, Z, N, T_grid)
    _parity.assert_close(out_old, out_new, rtol=1e-12, atol=0.0,
                         label=f"rr[{label}]")


@pytest.mark.parametrize("Z,N,label", REC_CATALOG)
def test_parity_get_dr_rate(rc_pair_caseB, T_grid, Z, N, label):
    if Z == 1 or N == 0:
        pytest.skip("DR not defined for Z=1 or N=0")
    rc_old, rc_new = rc_pair_caseB
    out_old, out_new = _parity.run_both(
        rc_old.get_dr_rate, rc_new.get_dr_rate, Z, N, T_grid)
    _parity.assert_close(out_old, out_new, rtol=1e-12, atol=0.0,
                         label=f"dr[{label}]")


@pytest.mark.parametrize("Z,N,label", REC_CATALOG)
def test_parity_get_rec_rate_caseB(rc_pair_caseB, T_grid, Z, N, label):
    """Total recombination rate, caseB=True (the production default).
    Exercises both the Z=1 Draine Case B branch and the Z>1 RR+DR
    branch.
    """
    rc_old, rc_new = rc_pair_caseB
    out_old, out_new = _parity.run_both(
        rc_old.get_rec_rate, rc_new.get_rec_rate, Z, N, T_grid)
    _parity.assert_close(out_old, out_new, rtol=1e-12, atol=0.0,
                         label=f"rec_caseB[{label}]")


@pytest.mark.parametrize("Z,N,label", REC_CATALOG)
def test_parity_get_rec_rate_caseA(rc_pair_caseA, T_grid, Z, N, label):
    """Total recombination rate, caseB=False (Case A for H via
    Badnell RR). Catches accidental flips of the caseB attribute
    semantics during the port.
    """
    rc_old, rc_new = rc_pair_caseA
    out_old, out_new = _parity.run_both(
        rc_old.get_rec_rate, rc_new.get_rec_rate, Z, N, T_grid)
    _parity.assert_close(out_old, out_new, rtol=1e-12, atol=0.0,
                         label=f"rec_caseA[{label}]")


# ---------------------------------------------------------------------
# Static-method H formulas + grain-assisted recombination.
# These are pure functions; their behavior cannot change during a
# port unless the bodies were edited. Verify anyway.
# ---------------------------------------------------------------------

def test_parity_static_H_formulas(T_grid):
    """Draine 2011 Case A/B and Athena-TIGRESS Case B-fit formulas."""
    for name in ("get_rec_rate_H_caseA_Dr11",
                 "get_rec_rate_H_caseB_Dr11",
                 "get_rec_rate_H_caseB"):
        f_old = getattr(RecRateOld, name)
        f_new = getattr(RecRateNew, name)
        out_old, out_new = _parity.run_both(f_old, f_new, T_grid)
        _parity.assert_close(out_old, out_new, rtol=1e-12, atol=0.0,
                             label=f"static[{name}]")


@pytest.mark.parametrize("Z", [1, 2, 6, 12, 16, 20])
def test_parity_grain_assisted(Z, T_grid):
    """`get_rec_rate_grain(ne, G0, T, Z)` for each element with a
    Draine 2011 Eq. 14.37 fit entry.
    """
    ne = 0.1
    G0 = 1.0
    out_old, out_new = _parity.run_both(
        RecRateOld.get_rec_rate_grain, RecRateNew.get_rec_rate_grain,
        ne, G0, T_grid, Z)
    _parity.assert_close(out_old, out_new, rtol=1e-12, atol=0.0,
                         label=f"grain[Z={Z}]")
