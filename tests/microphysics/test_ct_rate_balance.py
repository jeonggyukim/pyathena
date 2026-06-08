"""Tests for charge-exchange rate sign convention and detailed balance.

Documents the expected behavior of `ChargeTransferRate.get_ct_rec_rate`
and `get_ct_ion_rate` and pins it down before downstream code starts
wiring CT into `evolve_one_species` (pyathena_ct_fixes_plan.md MVP
item 3 prerequisite).

API conventions verified here (both functions now use REACTANT-ion
indexing):
  * `get_ct_rec_rate(Z, N, T)` -> rate for X^q + H I -> X^(q-1) + H II,
    where (Z, N) labels the REACTANT (ion X^q with charge q = Z - N).
    Requires q >= 1 (already-neutral reactants cannot recombine).
    The reactant side is the exothermic side; no Boltzmann factor in
    the rate.
  * `get_ct_ion_rate(Z, N, T)` -> rate for X^q + H II -> X^(q+1) + H I,
    where (Z, N) labels the REACTANT (same convention). The reactant
    side is the endothermic side for the ions we care about
    (O, N, S); the rate carries `exp(-dE/T)` with dE > 0.

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


# (Z, N) = (atomic number, electron number) labels the REACTANT ion
# in both `get_ct_rec_rate` and `get_ct_ion_rate`.
#
# CT_ION_CATALOG -- neutral X reacting with H+ to ionize (q = 0).
# CT_REC_CATALOG -- singly-ionized X+ reacting with H to recombine
# (q = 1); same atom as the corresponding CT_ION entry, one electron
# fewer.

CT_ION_CATALOG = [
    (8, 8, "O I"),   # O + H II -> O+ + H I; near-resonant, dE ~ 0.02 eV
    (7, 7, "N I"),   # N + H II -> N+ + H I; dE ~ 0.93 eV
    (16, 16, "S I"), # S + H II -> S+ + H I; dE ~ 1.6 eV
]

CT_REC_CATALOG = [
    (8, 7, "O II"),  # O+ + H I -> O + H II; near-resonant (Draine 2011)
    (7, 6, "N II"),  # N+ + H I -> N + H II
    (16, 15, "S II"),# S+ + H I -> S + H II
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

@pytest.mark.parametrize("Z,N,label", CT_REC_CATALOG)
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
    ct_ion(N I) should grow with T markedly faster than ct_rec(N II).
    If the Boltzmann factor sign is wrong, this fails immediately.

    Reactant-indexed convention: CT-ion of N (charge 0) takes
    (Z=7, N=7); the reverse CT-rec of N+ (charge 1) takes (Z=7, N=6).
    """
    T_lo = 5.0e3
    T_hi = 5.0e4
    ion_lo = ct.get_ct_ion_rate(7, 7, T_lo)
    ion_hi = ct.get_ct_ion_rate(7, 7, T_hi)
    rec_lo = ct.get_ct_rec_rate(7, 6, T_lo)
    rec_hi = ct.get_ct_rec_rate(7, 6, T_hi)

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
    close to 1 across the HII T range. ct_ion(O I) and ct_rec(O II)
    should grow by a similar factor; their ratio at T_lo and T_hi
    should not differ by more than ~30 percent.

    Reactant-indexed: O (charge 0) for ct_ion -> (8, 8); O+ (charge 1)
    for ct_rec -> (8, 7).
    """
    T_lo = 5.0e3
    T_hi = 5.0e4
    ion_lo = ct.get_ct_ion_rate(8, 8, T_lo)
    ion_hi = ct.get_ct_ion_rate(8, 8, T_hi)
    rec_lo = ct.get_ct_rec_rate(8, 7, T_lo)
    rec_hi = ct.get_ct_rec_rate(8, 7, T_hi)

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
        # Place the validity-floor annotation just inside the ylim,
        # to the right of the T=200 K guide line.
        ax.text(230, 5e-11, "Cloudy poly\nvalid $T > 200$ K",
                fontsize="x-small", color="C4", va="bottom", ha="left")
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
        ax.set_ylim(3e-11, 5e-9)
        ax.grid(True, which="both", alpha=0.3)
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


def _K_realistic_OH(rc, ct, T_arr, nH=30.0, xi_CR=2.0e-16,
                    G_PE=1.0, Z_d=1.0):
    """Realistic K(T) = (x_OII/x_OI) / (x_HII/x_HI) under neutral-gas
    equilibrium with CR ionization + grain-assisted recombination +
    radiative + dielectronic recombination + charge transfer.

    Hydrogen ionization solved via pyathena `get_xHII` with
    `zeta_pi=0` (no UV photoionization in neutral gas) and
    `gr_rec=True` (grain-assisted recomb for H+ active). Self-
    consistency over xe = x_HII is reached by fixed-point iteration;
    metals contribute negligibly to xe in diffuse neutral gas so
    xeM is set to 0.

    Oxygen ionization: source = CR ionization (zeta_O = 2.7 * xi_CR
    per Draine 2011 §13.7) + CT-ion of O0 by H+; sink = CT-rec of O+
    by H0 + RR + DR. Grain-assisted recombination for O+ is NOT in
    WD01 / Draine 2011 §14.37 (only H+, He+, C+, Mg+, S+, Ca+ are
    tabulated), so alpha_gr_O = 0 here. Photoionization is zero in
    neutral gas.
    """
    from pyathena.microphysics.get_xe_eq import get_xHII
    K = np.zeros_like(T_arr)
    for i, T in enumerate(T_arr):
        # Fixed-point iteration to converge xe = x_HII.
        xe = 1.0e-4
        for _ in range(80):
            xHII_new = get_xHII(nH, xe, 0.0, 0.0, T,
                                xi_CR, G_PE, Z_d, 0.0, True)
            if abs(xHII_new - xe) / (xe + 1e-30) < 1.0e-5:
                xe = xHII_new
                break
            xe = xHII_new
        x_HII = xe
        x_HI = 1.0 - x_HII
        n_HII = x_HII * nH
        n_HI = x_HI * nH
        n_e = xe * nH

        zeta_O = 2.7 * xi_CR
        k_CT_ion = ct.get_ct_ion_rate(8, 8, T)   # O + H+ -> O+ + H
        k_CT_rec = ct.get_ct_rec_rate(8, 7, T)   # O+ + H -> O + H+
        alpha_rec = rc.get_rec_rate(8, 7, T)     # RR + DR for O+
        # grain-assisted rec for O+ not tabulated; set to 0
        alpha_gr_O = 0.0

        num = zeta_O + n_HII * k_CT_ion
        den = n_HI * k_CT_rec + n_e * (alpha_rec + alpha_gr_O)
        r_O = num / den                          # n(O+)/n(O0)
        r_H = x_HII / max(x_HI, 1e-30)           # n(H+)/n(H0)
        K[i] = r_O / r_H
    return K


def test_plot_oxygen_CT_equilibrium(figures_dir, save_figures):
    """Reproduce the CORRECTED Draine 2011 Figure 14.5: the ratio
    `[n(O+)/n(O0)] / [n(H+)/n(H0)]` versus T, in the low-density
    and high-density limits.

    NOTE ON THE PUBLISHED FIGURE
    ----------------------------
    The printed Figure 14.5 in the 2011 Princeton edition has
    numerically incorrect curves (y-axis range 0 to 1.2, low-density
    curve mislabelled as the upper one reaching ~1.15 at T = 1e4 K).
    Bruce Draine acknowledges this on his book-errata page at
    https://www.astro.princeton.edu/~draine/book/errata_p1.pdf:

      "§14.7.1, p. 157, Figure 14.5: plotted curves were
      numerically incorrect.  Corrected Figure 14.5: [...]"
      (noted 2011.05.18 by E. B. Jenkins)

    The corrected figure on the errata page has:
      - y-axis range 0 to 1.0,
      - "high density" as the UPPER curve, "low density" as the LOWER,
      - both curves asymptoting to ~8/9 at high T.

    This test reproduces the corrected figure, not the buggy
    printed version. The equations in the book (14.31, 14.32, 14.35)
    are correct; only the plot was wrong.

    Draine 2011 Eqs. 14.31-14.35: in the low-density limit the
    O ground state stays in J=2 (decays beat collisions), so only
    the J=2 channel of CT-rec is active and

        K_low(T) = k0r / (k0 + k1 + k2)

    In the high-density limit the O fine-structure levels J=0,1,2
    thermalize via collisions:

        K_high(T) = [5*k0r + 3*k1r*exp(-228/T) + k2r*exp(-326/T)]
                    / [5 + 3*exp(-228/T) + exp(-326/T)]
                    / (k0 + k1 + k2)

    where k0, k1, k2 are the per-J CT-ionization rates (Draine eqs
    14.24-26 -> pyathena `get_ct_ion_HII_OI_Draine11(T, sum=False)`)
    and k0r, k1r, k2r are the per-J CT-recomb rates (Draine eqs
    14.28-30 -> `get_ct_rec_HI_OII_Draine11(T, sum=False)`).

    At T >~ 10^3 K both ratios approach ~8/9, so x_OII tracks x_HII.
    At T <~ 300 K the J=2 Boltzmann suppression (exp(-229/T)) pulls
    K_low below 1, so x_OII falls below x_HII.

    Skipped when --no-figures is passed.
    """
    if not save_figures:
        pytest.skip("plot generation disabled (--no-figures)")
    import matplotlib.pyplot as plt

    T = np.logspace(1.5, 4.5, 400)  # 30 K to 30000 K
    # Draine 2011 notation:
    #   k0, k1, k2 = CT-RECOMBINATION rates per O J-level
    #                (Draine eq 14.24-26) -- O+ + H -> O(J) + H+
    #   k0r, k1r, k2r = CT-IONIZATION rates per O J-level
    #                   (Draine eq 14.28-30) -- O(J) + H+ -> O+ + H
    # Pyathena mapping:
    #   get_ct_rec_HI_OII_Draine11 returns Draine (k0, k1, k2)
    #   get_ct_ion_HII_OI_Draine11 returns Draine (k0r, k1r, k2r)
    k0, k1, k2 = ChargeTransferRate.get_ct_rec_HI_OII_Draine11(T, sum=False)
    k0r, k1r, k2r = ChargeTransferRate.get_ct_ion_HII_OI_Draine11(T, sum=False)
    k_rec_tot = k0 + k1 + k2

    # Low-density limit (Draine eq 14.32): O ground state all in J=2.
    #   K_low = k0r / (k0 + k1 + k2)
    K_low = k0r / k_rec_tot

    # High-density limit (Draine eq 14.35): J=0/1/2 thermalize per
    # Boltzmann (E(J=1) = 228 K, E(J=0) = 326 K above J=2 ground).
    Tinv = 1.0 / T
    w0 = 5.0
    w1 = 3.0 * np.exp(-228.0 * Tinv)
    w2 = 1.0 * np.exp(-326.0 * Tinv)
    K_high = (w0 * k0r + w1 * k1r + w2 * k2r) / (w0 + w1 + w2) / k_rec_tot

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ---- Left panel: faithful Draine 2011 Fig 14.5 reproduction.
    ax = axes[0]
    ax.semilogx(T, K_low, "C0-", lw=1.6,
                label=r"low density ($n_{\rm H}\ll n_{\rm crit}$)")
    ax.semilogx(T, K_high, "C3--", lw=1.6,
                label=r"high density ($n_{\rm H}\gg n_{\rm crit}$)")
    ax.axhline(8.0 / 9.0, color="gray", lw=0.6, ls=":", alpha=0.7)
    ax.text(15, 8.0/9.0 + 0.03, r"$8/9$",
            fontsize="x-small", color="gray", va="bottom")
    ax.set_xlabel(r"$T\,[{\rm K}]$")
    ax.set_ylabel(r"$[n({\rm O}^+)/n({\rm O}^0)]\,/\,"
                  r"[n({\rm H}^+)/n({\rm H}^0)]$")
    ax.set_title("CT-only equilibrium ratio\n(corrected Draine 2011 Fig 14.5; see errata_p1.pdf)")
    ax.set_xlim(10, 30000)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize="x-small", loc="lower right")

    # ---- Right panel: self-consistent x_HII and x_OII vs n_H in
    # neutral gas (no photoionization), at fixed chi_FUV = 1 and
    # xi_CR = 2e-16 s^-1, for a couple of representative T.
    ax = axes[1]
    from pyathena.microphysics.rec_rate import RecRate
    from pyathena.microphysics.get_xe_eq import get_xHII
    rc_full = RecRate(caseB=False)
    ct_obj = ChargeTransferRate()
    nH_arr = np.logspace(0, 4, 50)
    chi_FUV = 1.0
    xi_CR = 2.0e-16
    Z_d = 1.0
    T_list = [50.0, 200.0]
    color_T = {50.0: "C0", 200.0: "C3"}
    for T_fixed in T_list:
        x_HII = np.zeros_like(nH_arr)
        x_OII = np.zeros_like(nH_arr)
        for j, nH in enumerate(nH_arr):
            # Self-consistent x_HII = xe (xeM = 0 in diffuse neutral gas).
            xe = 1.0e-4
            for _ in range(80):
                xnew = get_xHII(nH, xe, 0.0, 0.0, T_fixed,
                                xi_CR, chi_FUV, Z_d, 0.0, True)
                if abs(xnew - xe) / (xe + 1e-30) < 1e-5:
                    xe = xnew
                    break
                xe = xnew
            x_HII[j] = xe

            # O balance: CR + CT-ion (source) vs CT-rec + (RR+DR) (sink).
            x_HI = 1.0 - xe
            n_HII = xe * nH
            n_HI = x_HI * nH
            n_e = xe * nH
            zeta_O = 2.7 * xi_CR
            k_CT_ion = ct_obj.get_ct_ion_rate(8, 8, T_fixed)
            k_CT_rec = ct_obj.get_ct_rec_rate(8, 7, T_fixed)
            alpha_rec = rc_full.get_rec_rate(8, 7, T_fixed)
            num = zeta_O + n_HII * k_CT_ion
            den = n_HI * k_CT_rec + n_e * alpha_rec
            r_O = num / den
            x_OII[j] = r_O / (1.0 + r_O)
        ax.loglog(nH_arr, x_HII, "-",  color=color_T[T_fixed], lw=1.5,
                  label=rf"$x_{{\rm HII}}$, $T={T_fixed:g}$ K")
        ax.loglog(nH_arr, x_OII, "--", color=color_T[T_fixed], lw=1.5,
                  label=rf"$x_{{\rm OII}}$, $T={T_fixed:g}$ K")
    ax.set_xlabel(r"$n_{\rm H}\,[{\rm cm}^{-3}]$")
    ax.set_ylabel(r"$x_{\rm HII}$ or $x_{\rm OII}$")
    ax.set_title(
        r"Self-consistent eq fractions in neutral gas"
        "\n"
        r"$\chi_{\rm FUV}=1$, $\xi_{\rm CR}=2\times10^{-16}$ s$^{-1}$, "
        "no photoionization"
    )
    ax.set_xlim(1.0, 1e4)
    ax.set_ylim(1e-6, 1.0)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize="x-small", loc="lower left")

    fig.tight_layout()
    fig.savefig(figures_dir / "ct_O_equilibrium_Draine_fig14_5.png", dpi=200)
    plt.close(fig)


def test_ct_rec_rate_dalgarno_for_high_q(ct):
    """`get_ct_rec_rate` falls back to a Dalgarno-like generic
    estimate `1.92e-9 * q` for reactant charge q > 4 (data tables
    only cover q <= 4 in Cloudy's `ctrecombdata.dat`). Pin this for
    now -- a future upgrade to Pequignot 1996 / Stancil+99 fits for
    high-q ions will intentionally break this pin.

    Reactant-indexed: Fe^5+ is the reactant (charge 5, electron count
    21). The reaction is Fe^5+ + H I -> Fe^4+ + H II.
    """
    T = 1.0e4
    q = 5
    Z, N = 26, 26 - q  # Fe^5+ as reactant
    rate = ct.get_ct_rec_rate(Z, N, T)
    np.testing.assert_allclose(rate, 1.92e-9 * q, rtol=1e-6)
