"""Tests for charge-exchange rate sign convention and detailed balance.

Documents the expected behavior of `ChargeTransferRate.get_ct_rec_rate`
and `get_ct_ion_rate` and pins it down before downstream code starts
wiring CT into `evolve_one_species` (pyathena_ct_fixes_plan.md MVP
item 3 prerequisite).

API conventions verified here:
  * `get_ct_rec_rate(Z, N, T)` -> rate for X^(q+1) + H I -> X^q + H II.
    (Z, N) labels the PRODUCT (ion X^q with charge q = Z - N).
    The product side is the exothermic side; no Boltzmann factor in
    the rate.
  * `get_ct_ion_rate(Z, N, T)` -> rate for X^q + H II -> X^(q+1) + H I.
    (Z, N) labels the REACTANT (ion X^q with charge q = Z - N).
    The reactant side is the endothermic side for the ions we care
    about (O, N, S); the rate carries `exp(-dE/T)` with dE > 0.

Reference data:
  * Kingdon & Ferland 1996, ApJS 106 205, Table 1 (Cloudy
    ctiondata.dat / ctrecombdata.dat).
  * Draine 2011, ISM and IGM, Sec 14.5 (O-H near-resonant CT,
    per-J-level fits coded in `get_ct_rec_HI_OII_Draine11` /
    `get_ct_ion_HII_OI_Draine11`).
  * Pequignot 1996, A&A 313 1026 (resonance-line fits cross-check).

The tests do not yet wire CT into the abundance sweep -- that is the
follow-up commit. They only document the rate signs and magnitudes so
that any later change to `ct_rate.py` flags a regression here first.
"""

import numpy as np
import pytest

from pyathena.microphysics.ct_rate import ChargeTransferRate


# Ion catalog: (Z, N, label) for ions with documented CT data in
# Kingdon & Ferland 1996 (Table 1 entries used by Cloudy / pyathena).
# (Z, N) = (atomic number, electron number); charge q = Z - N.
CT_ION_CATALOG = [
    (8, 8, "O I"),     # O + HII -> O+ + HI; near-resonant, dE ~ 0.02 eV
    (7, 7, "N I"),     # N + HII -> N+ + HI; dE ~ 0.93 eV
    (16, 16, "S I"),   # S + HII -> S+ + HI; dE ~ 1.6 eV
]


@pytest.fixture(scope="module")
def ct():
    return ChargeTransferRate()


@pytest.fixture(scope="module")
def T_grid():
    return np.logspace(np.log10(5e3), np.log10(5e4), 11)


# ---------------------------------------------------------------------
# Smoke tests: rates are finite + positive in HII-region T range.
# ---------------------------------------------------------------------

@pytest.mark.parametrize("Z,N,label", CT_ION_CATALOG)
def test_ct_rec_rate_finite_positive(ct, T_grid, Z, N, label):
    """CT recombination rate is finite and positive for all T."""
    rate = ct.get_ct_rec_rate(Z, N, T_grid)
    assert np.all(np.isfinite(rate)), f"{label}: ct_rec rate non-finite"
    assert np.all(rate > 0), f"{label}: ct_rec rate <= 0 somewhere"


@pytest.mark.parametrize("Z,N,label", CT_ION_CATALOG)
def test_ct_ion_rate_finite_nonneg(ct, T_grid, Z, N, label):
    """CT ionization rate is finite and >= 0 for all T."""
    rate = ct.get_ct_ion_rate(Z, N, T_grid)
    assert np.all(np.isfinite(rate)), f"{label}: ct_ion rate non-finite"
    assert np.all(rate >= 0), f"{label}: ct_ion rate < 0 somewhere"


# ---------------------------------------------------------------------
# Sign-convention probe: for endothermic CT-ion (N I), the rate must
# grow with T faster than CT-rec. For the near-resonant O I case the
# growth is comparable to ct_rec because the Boltzmann factor is ~1
# across the HII-region T range (dE ~ 230 K). S I in the Cloudy data
# is a near-constant placeholder (a=1e-5, b=c=d=dE=0) -- the constant
# behavior is pinned in `test_ct_ion_rate_SI_is_placeholder` below.
# ---------------------------------------------------------------------

def test_ct_ion_rate_NI_grows_faster_than_rec(ct):
    """N I CT-ion has dE ~ 0.93 eV -- clearly endothermic.
    ct_ion(N I) should grow with T markedly faster than ct_rec.
    If the Boltzmann factor sign is wrong, this fails immediately.
    """
    T_lo = 5.0e3
    T_hi = 5.0e4
    ion_lo = ct.get_ct_ion_rate(7, 7, T_lo)
    ion_hi = ct.get_ct_ion_rate(7, 7, T_hi)
    rec_lo = ct.get_ct_rec_rate(7, 7, T_lo)
    rec_hi = ct.get_ct_rec_rate(7, 7, T_hi)

    assert ion_hi > ion_lo, (
        f"N I ct_ion decreasing with T -- sign convention may be "
        f"inverted. ion(T_lo)={ion_lo:.3e}, ion(T_hi)={ion_hi:.3e}"
    )

    # Empirical growth ratio at the current data: ion grows 3.7x, rec
    # drops to 0.5x across [5e3, 5e4] K -- ion grows ~7x faster than
    # rec. Threshold 5x: comfortable margin while catching sign flips.
    growth_ion = ion_hi / ion_lo
    growth_rec = rec_hi / rec_lo
    assert growth_ion >= 5.0 * growth_rec, (
        f"N I ct_ion growth {growth_ion:.2f}x vs ct_rec growth "
        f"{growth_rec:.2f}x. Expected >= 5x faster."
    )


def test_ct_OH_resonance_growth_similar(ct):
    """For near-resonant O+H (dE ~ 230 K) the Boltzmann factor is
    close to 1 across the HII T range. ct_ion and ct_rec should grow
    by a similar factor; their ratio at T_lo and T_hi should not
    differ by more than ~30%.
    """
    T_lo = 5.0e3
    T_hi = 5.0e4
    ion_lo = ct.get_ct_ion_rate(8, 8, T_lo)
    ion_hi = ct.get_ct_ion_rate(8, 8, T_hi)
    rec_lo = ct.get_ct_rec_rate(8, 8, T_lo)
    rec_hi = ct.get_ct_rec_rate(8, 8, T_hi)

    ratio_lo = ion_lo / rec_lo
    ratio_hi = ion_hi / rec_hi
    np.testing.assert_allclose(
        ratio_hi, ratio_lo, rtol=0.5,
        err_msg=(
            f"O+H near-resonance: ratio_lo={ratio_lo:.3f}, "
            f"ratio_hi={ratio_hi:.3f} differ by >50%, unexpected for "
            f"dE ~ 230 K resonance"
        ),
    )


def test_ct_ion_rate_SI_is_placeholder(ct):
    """S I CT-ion in Cloudy's `ctiondata.dat` is a placeholder
    (a=1e-5, b=c=d=dE=0) -> constant 1e-14 cm^3/s regardless of T.
    Documented here so that downstream code can detect when better
    rates are needed. Probable fix when this becomes important:
    upgrade to Pequignot 1996 / Stancil+99 fits or UGACXDB ctr_hyd
    data for S I (currently only used as a collider, not as a target
    ion). See pyathena_ct_fixes_plan.md item 9.
    """
    T_lo = 5.0e3
    T_hi = 5.0e4
    rate_lo = ct.get_ct_ion_rate(16, 16, T_lo)
    rate_hi = ct.get_ct_ion_rate(16, 16, T_hi)
    np.testing.assert_allclose(
        rate_lo, 1.0e-14, rtol=0.01,
        err_msg=f"S I ct_ion at T={T_lo} = {rate_lo:.3e}, expected "
                f"placeholder ~1e-14"
    )
    np.testing.assert_allclose(
        rate_hi, 1.0e-14, rtol=0.01,
        err_msg=f"S I ct_ion at T={T_hi} = {rate_hi:.3e}, expected "
                f"placeholder ~1e-14"
    )


# ---------------------------------------------------------------------
# Detailed-balance ratio probe: at high T (kT >> dE), the ratio
# ct_ion / ct_rec approaches the statistical-weight ratio. For the
# Draine 2011 O+H per-J-level fit (used inside get_ct_*_rate for
# Z=N=8), the total ratio at T -> infty is the sum-weighted g-ratio.
# ---------------------------------------------------------------------

def test_ct_OH_detailed_balance_Draine11():
    """O+H CT-ion / CT-rec ratio at high T matches Draine 2011 per-J
    statistical weights: k0i/k0r = 8/5, k1i/k1r = 8/3, k2i/k2r = 8/1.
    """
    T_hi = 1e6  # well above all energy defects (max ~ 229 K).
    k0r, k1r, k2r = ChargeTransferRate.get_ct_rec_HI_OII_Draine11(
        T_hi, sum=False
    )
    k0i, k1i, k2i = ChargeTransferRate.get_ct_ion_HII_OI_Draine11(
        T_hi, sum=False
    )

    # per-J-level g-ratios (g_OII = 4 for ^4S_3/2; g_OI per J = 5, 3, 1).
    np.testing.assert_allclose(k0i / k0r, 8.0 / 5.0, rtol=1e-3,
                                err_msg="O+H J=2 detailed balance failed")
    np.testing.assert_allclose(k1i / k1r, 8.0 / 3.0, rtol=1e-3,
                                err_msg="O+H J=1 detailed balance failed")
    np.testing.assert_allclose(k2i / k2r, 8.0 / 1.0, rtol=1e-3,
                                err_msg="O+H J=0 detailed balance failed")


def test_ct_OH_resonance_at_T10000():
    """At T = 1e4 K (typical HII region), O+H total CT-rec and CT-ion
    are comparable (within factor of a few) due to small energy
    defect. If either rate disappears, the O-H ionization-state
    coupling that Kim+23 Section 4.6.2 relies on is broken.
    """
    T = 1.0e4
    rec = ChargeTransferRate.get_ct_rec_HI_OII_Draine11(T)
    ion = ChargeTransferRate.get_ct_ion_HII_OI_Draine11(T)

    # Order of magnitude check: ratio in [0.1, 10].
    ratio = ion / rec
    assert 0.1 < ratio < 10.0, (
        f"O+H CT-ion/ct-rec ratio at T=1e4 = {ratio:.3f}, "
        f"expected O(1) due to near-resonance"
    )

    # Both rates should be ~1e-9 cm^3/s (Draine 2011 Sec 14.5).
    assert 1e-10 < rec < 1e-8, f"O+H ct_rec at T=1e4 = {rec:.3e}"
    assert 1e-10 < ion < 1e-8, f"O+H ct_ion at T=1e4 = {ion:.3e}"


# ---------------------------------------------------------------------
# Documentation tests: catch silent zero-returns when ion outside data
# range or charge q > 3 cap in get_ct_ion_rate.
# ---------------------------------------------------------------------

def test_ct_ion_rate_zero_for_high_q(ct):
    """`get_ct_ion_rate` returns zero for q > 3. Flagged in
    pyathena_ct_fixes_plan.md item 6 as a Phase 7 extension issue.
    The test pins the current behavior so any future change is
    explicit.
    """
    T = 1.0e4
    # Fe^5+: Z=26, N=21, q=5 -> outside Cloudy table coverage.
    rate = ct.get_ct_ion_rate(26, 21, T)
    assert rate == 0.0, "high-q ct_ion fallback should be zero (q > 3 cap)"


def test_ct_rec_rate_dalgarno_for_high_q(ct):
    """`get_ct_rec_rate` falls back to Dalgarno-like generic estimate
    `1.92e-9 * (q + 1)` for q > 3. Pin this for now.
    """
    T = 1.0e4
    q = 5
    Z, N = 26, 26 - q  # Fe^5+
    rate = ct.get_ct_rec_rate(Z, N, T)
    np.testing.assert_allclose(rate, 1.92e-9 * (q + 1), rtol=1e-6)
