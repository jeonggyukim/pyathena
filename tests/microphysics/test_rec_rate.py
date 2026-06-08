"""Tests for radiative + dielectronic recombination rates.

What this file covers
---------------------
`pyathena/microphysics/rec_rate.py` exposes recombination rate
coefficients via the Badnell 2003-2023 analytic fits (radiative RR)
+ the Badnell 2006a/b dielectronic (DR) fits, both loaded from the
`badnell_rr_2023.dat`, `badnell_dr_C_2023.dat`, `badnell_dr_E_2023.dat`
data files. The `RecRate` class also has hand-coded special-case fits
for H (Draine 2011 Case A and Case B) that override the Badnell path
when `caseB=True`.

These rates feed the multi-ion sweep `evolve_one_species`. Bugs in
the dispatch -- wrong (Z, N) index, Case A vs B mixup, missing DR
contribution -- propagate directly into the equilibrium ionization
fractions, so a regression anchor here is essential before wiring CT
and the metal-line cooling on top.

API conventions verified
------------------------
* `get_rr_rate(Z, N, T, M=1)` -> radiative recombination coefficient
  [cm^3/s] for the ion (Z, N) BEFORE recombination (the ion that
  captures the electron). Examples:
    - H II  -> H I:  (Z=1, N=0)
    - He II -> He I: (Z=2, N=1)
    - O III -> O II: (Z=8, N=6)
  `M=1` selects the ground metastable level (default everywhere).
* `get_dr_rate(Z, N, T, M=1)` -> dielectronic recombination
  coefficient. Same (Z, N) = initial-ion convention. NOT defined for
  Z=1 (no DR for one-electron systems) or for N=0 (fully stripped);
  the dispatcher `get_rec_rate` skips DR in those cases.
* `get_rec_rate(Z, N, T, M=1, kind='badnell')` -> RR + DR total.
  Special case for Z=1: if `RecRate(caseB=True)` (the default), this
  returns the Draine 2011 Case B fit `get_rec_rate_H_caseB_Dr11(T)`;
  with `caseB=False`, it returns the Badnell RR rate (Case A for H,
  since RR alone covers recombination to all n levels).

Reference values used in spot-checks
------------------------------------
* H II Case B at T=1e4 K: alpha_B ~ 2.59e-13 cm^3/s
    (Draine 2011 Eq. 14.6; Hummer 1994).
* H II Case A at T=1e4 K: alpha_A ~ 4.18e-13 cm^3/s
    (Draine 2011 Eq. 14.5).
* O III DR peaks at T ~ 1e5 K (Badnell 2006a Fig. 1) -- used to
  verify the DR temperature shape.

Test strategy
-------------
* Smoke: RR / DR / total are finite + non-negative for an ion catalog
  covering H, He, C, N, O, S in the (Z, N) = initial-ion convention.
* Monotonicity: RR alone decreases with T (~T^-0.5 to T^-1.0) --
  follows from the Badnell RR functional form.
* H spot-check: alpha_B at T=1e4 K matches Draine; alpha_A > alpha_B.
* DR temperature shape: O III DR larger at T=1e5 than at T=1e3.
  Catches a sign-flip in the DR exponent.
* Consistency: for non-H ions, get_rec_rate == get_rr_rate +
  get_dr_rate exactly.
"""

import numpy as np
import pytest

from pyathena.microphysics.rec_rate import RecRate


# (Z, N_initial, label) -- (Z, N) = ion BEFORE recombination.
# E.g., HII rec -> HI is queried as (1, 0).
REC_CATALOG = [
    (1,  0, "H II"),
    (2,  1, "He II"),
    (2,  0, "He III"),
    (6,  5, "C II"),
    (6,  4, "C III"),
    (7,  6, "N II"),
    (7,  5, "N III"),
    (8,  7, "O II"),
    (8,  6, "O III"),
    (8,  5, "O IV"),
    (16, 15, "S II"),
    (16, 14, "S III"),
]


@pytest.fixture(scope="module")
def rc():
    return RecRate(caseB=True)


@pytest.fixture(scope="module")
def rc_caseA():
    return RecRate(caseB=False)


@pytest.fixture(scope="module")
def T_grid():
    return np.logspace(np.log10(5e3), np.log10(5e4), 11)


# ---------------------------------------------------------------------
# Smoke: rr / dr / total finite and nonneg across HII-region T.
# ---------------------------------------------------------------------

@pytest.mark.parametrize("Z,N,label", REC_CATALOG)
def test_rr_rate_finite_positive(rc, T_grid, Z, N, label):
    """Radiative recombination rate must be finite and strictly
    positive at every T in the HII-region grid [5e3, 5e4] K.

    Failure modes caught:
      - Wrong (Z, N) index -> KeyError or IndexError in the Badnell
        data lookup.
      - Missing entry in `badnell_rr_2023.dat` for one of the ions.
      - Negative rate from a bad fit parameter (would indicate a
        data corruption upstream).
    """
    rate = rc.get_rr_rate(Z, N, T_grid)
    assert np.all(np.isfinite(rate)), f"{label}: rr non-finite"
    assert np.all(rate > 0), f"{label}: rr <= 0"


@pytest.mark.parametrize("Z,N,label", REC_CATALOG)
def test_dr_rate_finite_nonneg(rc, T_grid, Z, N, label):
    """DR rate is nonneg; skipped for Z=1 or N=0 (no DR)."""
    if Z == 1 or N == 0:
        pytest.skip("DR not defined for Z=1 or fully stripped (N=0)")
    rate = rc.get_dr_rate(Z, N, T_grid)
    assert np.all(np.isfinite(rate)), f"{label}: dr non-finite"
    assert np.all(rate >= 0), f"{label}: dr < 0"


@pytest.mark.parametrize("Z,N,label", REC_CATALOG)
def test_total_rec_rate_finite_positive(rc, T_grid, Z, N, label):
    rate = rc.get_rec_rate(Z, N, T_grid)
    assert np.all(np.isfinite(rate)), f"{label}: total non-finite"
    assert np.all(rate > 0), f"{label}: total <= 0"


# ---------------------------------------------------------------------
# Monotonicity: RR alone decreases with T (T^-0.5 to T^-1 typical).
# Test on a wide T range so the slope is unambiguous.
# ---------------------------------------------------------------------

@pytest.mark.parametrize("Z,N,label", [
    (1, 0, "H II RR"),
    (2, 0, "He III RR"),
    (8, 6, "O III RR"),
])
def test_rr_rate_decreases_with_T(rc, Z, N, label):
    T_lo = 1.0e3
    T_hi = 1.0e6
    rate_lo = rc.get_rr_rate(Z, N, T_lo)
    rate_hi = rc.get_rr_rate(Z, N, T_hi)
    assert rate_lo > rate_hi, (
        f"{label}: rr should decrease with T. "
        f"rate(T=1e3)={rate_lo:.3e}, rate(T=1e6)={rate_hi:.3e}"
    )


# ---------------------------------------------------------------------
# Reference values: Case B H II at T=1e4, Draine 2011 Eq. 14.6.
# ---------------------------------------------------------------------

def test_HII_caseB_at_T10000(rc):
    """Case B Hydrogen recombination at T=1e4 K: alpha_B ~ 2.59e-13.

    Why this matters: alpha_B at T=1e4 is the single most-cited
    number in HII region physics. It sets the Stromgren radius
    `R_S = (3*Q/(4*pi*alpha_B*n_H^2))^(1/3)`, the Lyman-alpha
    emissivity normalization, and the H II equilibrium ionization
    fraction. If this value is wrong, every downstream calculation
    in `evolve_one_species` is wrong by the same factor.

    Reference: Draine 2011 Eq. 14.6; Hummer 1994 alpha_B(T=1e4 K) =
    2.59e-13 cm^3/s. Tolerance 5% covers the range of different
    Case B parameterizations (Hummer, Verner, Pequignot).
    """
    T = 1.0e4
    alpha = rc.get_rec_rate(1, 0, T)
    np.testing.assert_allclose(
        alpha, 2.59e-13, rtol=0.05,
        err_msg=f"H II alpha_B at T=1e4: got {alpha:.3e}, "
                f"expected ~2.59e-13 (Draine 2011 Eq. 14.6)"
    )


def test_HII_caseA_vs_caseB(rc_caseA, rc):
    """Case A > Case B (Case A includes recombinations to n=1).

    Physics: Case A = total recomb to all n levels.
             Case B = exclude n=1 recombinations (their Lyman-line
                      emission gets re-absorbed in optically-thick
                      HII regions and re-ionizes another H atom).
    `alpha_A = alpha_B + alpha_to_n_1`; in dense HII regions, the
    n=1 recombinations contribute ~1.6e-13, making Case A ~1.6x
    Case B at T=1e4 K.

    Why this matters: Case B is the right choice for HII region
    simulations (clouds are optically thick to Lyman lines). The
    `RecRate(caseB=True)` default and the Z=1 dispatch in
    `get_rec_rate` enforce this. If someone instantiates
    `RecRate(caseB=False)` by accident and the dispatcher silently
    falls through to Case A, equilibrium x_HII jumps by 1.6x. This
    test verifies both that Case A and Case B are distinguishable
    and that Case A > Case B.

    Reference: Draine 2011 Eq. 14.5 (Case A) and 14.6 (Case B);
    alpha_A(T=1e4) ~ 4.18e-13 cm^3/s, alpha_B(T=1e4) ~ 2.59e-13.
    Tolerance window [3.5e-13, 4.5e-13] accepts either the Badnell
    H RR fit (~4.06e-13) or the Draine 2011 Case A formula
    (~4.18e-13).
    Initial-ion convention: HII = (Z=1, N=0).
    """
    T = 1.0e4
    alpha_B = rc.get_rec_rate(1, 0, T)
    # Case-A object: Z=1 with kind='badnell' uses get_rr_rate (case A).
    alpha_A = rc_caseA.get_rec_rate(1, 0, T)
    assert alpha_A > alpha_B, (
        f"alpha_A={alpha_A:.3e} should exceed alpha_B={alpha_B:.3e}"
    )
    # Loose tolerance: Badnell H RR fit gives ~4.06e-13 at T=1e4; Draine
    # 2011 Case A formula gives ~4.18e-13. Accept either.
    assert 3.5e-13 < alpha_A < 4.5e-13, (
        f"H Case A at T=1e4: got {alpha_A:.3e}, expected ~4.0e-13"
    )


# ---------------------------------------------------------------------
# DR peak behavior: O III DR has a peak around T ~ 1e5 K (Badnell 2006a).
# At T = peak, DR > RR. At very low T (T << 1e4), DR << RR.
# ---------------------------------------------------------------------

def test_OIII_DR_peaks_in_warm_T(rc):
    """O III DR rate is small at T=1e3 K, large at T ~ 1e5 K.

    Physics: dielectronic recombination requires the captured
    electron's energy to match an autoionizing-state energy of the
    product ion. For O^++ -> O^+, the relevant autoionizing states
    sit at ~10 eV above the ground; the Maxwell-Boltzmann tail of
    electrons at T ~ 1e5 K populates those energies efficiently.
    At T=1e3 K, electrons are far too cold to access them, so DR
    is exponentially suppressed.

    Why this matters: O III is the dominant ion of O in warm HII
    gas (T ~ 1.5e4 K) and hotter, and DR contributes ~half of its
    recombination rate in that regime. A sign-flip in the DR
    exponent (`+E_m/T` instead of `-E_m/T`) would make DR diverge
    at low T and vanish at high T -- exactly opposite of the
    correct behavior.

    Reference: Badnell 2006a Fig. 1 shows O III DR peaking around
    T_e ~ 1e5 K. Test just checks monotonicity between T=1e3 (very
    cold, DR strongly suppressed) and T=1e5 (peak region).
    Initial-ion convention: O III = (Z=8, N=6).
    """
    T_cold = 1.0e3
    T_warm = 1.0e5
    dr_cold = rc.get_dr_rate(8, 6, T_cold)
    dr_warm = rc.get_dr_rate(8, 6, T_warm)
    assert dr_warm > dr_cold, (
        f"O III DR should be higher at warm T. "
        f"dr(T=1e3)={dr_cold:.3e}, dr(T=1e5)={dr_warm:.3e}"
    )


# ---------------------------------------------------------------------
# Total = RR + DR for non-H, non-fully-stripped ions (Z>1, N>0).
# ---------------------------------------------------------------------

def test_plot_rec_rate_overview(rc, figures_dir, save_figures, ion_colors):
    """Diagnostic plot (no assertion): RR and DR rate coefficients
    for a sample of the followed ions across T = [1e2, 1e8] K.

    Shows the classic decline of RR with T, the DR peaks at T~Tmax
    of each ion, and that the total (RR + DR) is non-monotonic.

    Per-ion color follows the `photchem.py` convention via the
    `ion_colors(Z, q)` fixture.

    Skipped when `--no-figures`.
    """
    if not save_figures:
        pytest.skip("plot generation disabled (--no-figures)")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as path_effects
    from pyathena.plt_tools.line_annotation import line_annotate
    T = np.logspace(2, 8, 400)
    # (Z, N_initial, label, x_annot_K). x_annot picks a per-ion T
    # where the line sits in an uncrowded region of the plot.
    PLOT_IONS = [
        (1, 0, "H II",   3e3),
        (2, 1, "He II",  3e3),
        (2, 0, "He III", 1e5),
        (6, 5, "C II",   1e4),
        (7, 6, "N II",   3e4),
        (8, 7, "O II",   1e5),
        (8, 6, "O III",  3e5),
    ]
    stroke = [path_effects.withStroke(linewidth=2.5, foreground="white")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for Z, N, label, x_annot in PLOT_IONS:
        q = Z - N           # initial-ion charge
        color = ion_colors(Z, q)
        rr = rc.get_rr_rate(Z, N, T)
        ln_rr, = axes[0].loglog(T, rr, color=color, lw=1.4)
        line_annotate(label, ln_rr, x=x_annot, fontsize="x-small",
                      color=color, path_effects=stroke)
        if Z > 1 and N > 0:
            dr = rc.get_dr_rate(Z, N, T)
            ln_dr, = axes[1].loglog(T, dr, color=color, lw=1.4)
            line_annotate(label, ln_dr, x=x_annot, fontsize="x-small",
                          color=color, path_effects=stroke)
    axes[0].set_title("Radiative recombination (Badnell)")
    axes[1].set_title("Dielectronic recombination (Badnell)")
    for ax in axes:
        ax.set_xlabel(r"$T\,[{\rm K}]$")
        ax.set_ylabel(r"$\alpha\,[{\rm cm}^3\,{\rm s}^{-1}]$")
        ax.set_ylim(1e-14, 1e-9)
        ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "rec_rate_overview.png", dpi=200)
    plt.close(fig)


@pytest.mark.parametrize("Z,N,label", [
    (6, 5, "C II"),
    (8, 6, "O III"),
    (7, 6, "N II"),
])
def test_total_equals_RR_plus_DR(rc, T_grid, Z, N, label):
    """For non-H ions, `get_rec_rate` should equal `rr + dr`."""
    rr = rc.get_rr_rate(Z, N, T_grid)
    dr = rc.get_dr_rate(Z, N, T_grid)
    total = rc.get_rec_rate(Z, N, T_grid)
    np.testing.assert_allclose(
        total, rr + dr, rtol=1e-6,
        err_msg=f"{label}: total != rr+dr"
    )
