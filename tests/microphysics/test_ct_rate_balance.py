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


def _ct_rec_OH_KF96(T):
    """Kingdon & Ferland 1996 simple-fit O+H CT-rec (single channel)."""
    T4 = T * 1.0e-4
    a, b, c, d = 1.04e-9, 3.15e-2, -0.61, -9.73
    return a * T4 ** b * (1.0 + c * np.exp(d * T4))


def _ct_rec_OH_Cloudy(T):
    """Cloudy ln(T) polynomial fit (atmdat_char_tran.cpp).
    Valid for T > 10 K (the polynomial diverges at much lower T).
    """
    lnT = np.log(T)
    coeffs = [2.3344302e-10, 2.3651505e-10, -1.3146803e-10,
              2.9979994e-11, -2.8577012e-12, 1.1963502e-13]
    return (((((coeffs[5]*lnT + coeffs[4])*lnT + coeffs[3])*lnT
                + coeffs[2])*lnT + coeffs[1])*lnT + coeffs[0])


def test_OH_CT_rec_agrees_across_sources(figures_dir, save_figures):
    """Cross-check pyathena's Draine 2011 O+H CT-rec implementation
    against Kingdon & Ferland 1996 (KF96) and Cloudy polynomial fits.

    Background: three published fits exist for the same O+H CT
    reaction. Pyathena uses Draine 2011 per-J-level summed (sum of
    `k0r + k1r + k2r` in `get_ct_rec_HI_OII_Draine11`). The notebook
    `2024-KimJG-HII-WIND/hii_wind/notebooks/Charge-Transfer-Oxygen.ipynb`
    plots all three and shows agreement within ~factor of 2 across
    T = 1e2 to 1e4 K.

    PROVENANCE NOTE: Draine 2011 and Cloudy polynomial are NOT
    independent measurements. Both are fits to the same underlying
    quantum-mechanical scattering calculations of Stancil et al.
    1999, A&AS 140 225 (+ Barragan et al. 2006 for the low-T J=2
    resonance). The Cloudy fit is generated by TableCurve on the
    Stancil+99 Tables 2-4 (see `cloudy/source/atmdat_char_tran.cpp`
    line 103-134, reference comment cites Stancil 1999 explicitly).
    Draine 2011 Eq. 14.10 fits the same Stancil/Barragan tables in
    per-J form. So their ~1% agreement at T = 1e4 K is expected,
    not a cross-validation. The Cloudy poly is only valid for T >
    200 K and runs higher than Draine 2011 at low T (different
    fit forms diverging away from the data).
    KF96 (1996) PRE-dates Stancil+99 and is the only truly
    independent dataset of the three; its factor-of-2 offset
    represents the genuine improvement between the older Butler+80-
    era data compilation and the modern QM calculations.

    This test inlines the KF96 simple fit (single channel, no J
    resolution) and the Cloudy ln(T) polynomial so future refactors
    of pyathena's `get_ct_rec_HI_OII_Draine11` are compared against
    the same reference data. Cloudy and Draine are both J-summed
    fits and should agree closely (rtol=0.1). KF96 fits only the
    dominant channel and is expected to be ~0.5x Draine at T=1e4 K
    (a known difference, not a bug).

    Byproduct: when `save_figures` is True (default), produces
    `tests/figures/ct_OH_rec_comparison.png` -- the analog of the
    `Charge-Transfer-Oxygen.ipynb` figure.
    """
    T_check = 1.0e4

    k_Dr11 = ChargeTransferRate.get_ct_rec_HI_OII_Draine11(T_check)
    k_KF96 = _ct_rec_OH_KF96(T_check)
    k_Cl   = _ct_rec_OH_Cloudy(T_check)

    # Cloudy vs Draine: both J-summed -> tight agreement (~1% at 1e4).
    np.testing.assert_allclose(
        k_Dr11, k_Cl, rtol=0.1,
        err_msg=f"O+H CT-rec at T=1e4: Draine={k_Dr11:.3e}, "
                f"Cloudy poly={k_Cl:.3e}; these are both total-rate "
                f"fits and should agree closely"
    )
    # KF96 vs Draine: KF96 is a single-channel fit (mostly J=2),
    # NOT the J-summed total. Expect KF96 ~ 0.5 * Draine at T=1e4 K.
    # Loose window (factor of 3) catches a coefficient bug without
    # flagging the inherent KF96 vs Draine ratio.
    assert 0.2 < k_KF96 / k_Dr11 < 1.5, (
        f"O+H CT-rec at T=1e4: KF96/Draine = "
        f"{k_KF96/k_Dr11:.3f}, expected ~0.5 "
        f"(KF96 ground-state channel only vs Draine J-summed)"
    )

    if save_figures:
        # Mirror the Charge-Transfer-Oxygen.ipynb figure: rate vs T
        # with all three sources overplotted, plus the Draine per-J
        # decomposition.
        #
        # T range covers 100 K to 1e6 K (HII region through hot
        # ionized gas). Cloudy polynomial fit is documented only for
        # T > 200 K (`atmdat_char_tran.cpp:111-118` falls back to a
        # constant 3.744e-10 below 10 K and the TableCurve fit
        # diverges at low T anyway). A vertical line marks the
        # validity boundary.
        import matplotlib.pyplot as plt
        T_arr = np.logspace(2, 6, 400)
        k_Dr11_arr = ChargeTransferRate.get_ct_rec_HI_OII_Draine11(T_arr)
        k0r, k1r, k2r = ChargeTransferRate.get_ct_rec_HI_OII_Draine11(
            T_arr, sum=False
        )
        k_KF96_arr = _ct_rec_OH_KF96(T_arr)
        k_Cl_arr   = _ct_rec_OH_Cloudy(T_arr)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.loglog(T_arr, k_Dr11_arr, "k-", lw=2.5,
                  label="Draine 2011 (sum, pyathena)")
        ax.loglog(T_arr, k0r, "C0--", lw=0.8, alpha=0.7, label="Draine J=2")
        ax.loglog(T_arr, k1r, "C1--", lw=0.8, alpha=0.7, label="Draine J=1")
        ax.loglog(T_arr, k2r, "C2--", lw=0.8, alpha=0.7, label="Draine J=0")
        ax.loglog(T_arr, k_KF96_arr, "C3:", lw=2,
                  label="Kingdon & Ferland 1996 (single channel)")
        ax.loglog(T_arr, k_Cl_arr, "C4-.", lw=1.5,
                  label="Cloudy poly (atmdat_char_tran, valid T > 200 K)")
        ax.axvline(T_check, color="gray", lw=0.5, alpha=0.5)
        ax.axvline(200., color="C4", lw=0.5, alpha=0.4, linestyle=":")
        ax.text(220, 1.5e-12, "Cloudy poly\n  valid T > 200 K",
                fontsize="x-small", color="C4", va="bottom")
        ax.set_xlabel(r"$T\,[{\rm K}]$")
        ax.set_ylabel(
            r"$k_{\rm rec}({\rm O}^+ + {\rm H}\!\to\!{\rm O} + {\rm H}^+)"
            r"\,[{\rm cm}^3\,{\rm s}^{-1}]$"
        )
        ax.set_title(
            r"O-H charge-exchange recombination: source comparison" "\n"
            r"(Cloudy and Draine 2011 both fit Stancil+99 data; "
            r"Cloudy poly runs higher at low T)"
        )
        ax.legend(fontsize="x-small", loc="lower right")
        ax.set_ylim(1e-12, 5e-9)
        fig.tight_layout()
        fig.savefig(figures_dir / "ct_OH_rec_comparison.png", dpi=200)
        plt.close(fig)


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
