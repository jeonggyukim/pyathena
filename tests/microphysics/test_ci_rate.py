"""Tests for collisional ionization rates (Voronov 1997 fits).

What this file covers
---------------------
`pyathena/microphysics/ci_rate.py` exposes the Voronov 1997 (At. Data
Nucl. Data Tables 65, 1) analytic fit for collisional ionization rate
coefficients beta_CI(T) [cm^3/s]. The reaction is

    X^q + e^-  -->  X^(q+1) + 2 e^-

where (Z, N) is the reactant (the ion before ionization, q = Z-N).
The fit form is

    beta = A * (1 + P*U^0.5) / (X + U) * U^K * exp(-U),
    U    = dE / (k_B * T),

with parameters (A, P, X, K) and threshold dE loaded from Cloudy's
`coll_ion.dat`. The Boltzmann factor `exp(-U)` makes the rate
strongly endothermic for small T (large U), and large for T >= dE/k_B.

These rates feed the multi-ion sweep:

    drate += n_e * beta_CI(Z, N, T)    # destruction of (Z, N)
    crate += n_e * beta_CI(Z, N+1, T) * n_{(Z,N+1)}
                                       # creation from one-less-ionized

Sign / magnitude / cutoff bugs here directly distort non-equilibrium
ionization in the warm photoionized + hot collisionally ionized
regime where collisional ionization competes with photoionization.

API conventions verified
------------------------
* `get_ci_rate(Z, N, T)` -> rate coefficient [cm^3/s].
* (Z, N) labels the REACTANT (the ion being ionized).
* Returns exactly 0 when U > 80 (T < dE_Kel / 80) -- the explicit
  cutoff in `ci_rate.py:36-38` that avoids `exp(-80)` underflow.

Reference values used in spot-checks
------------------------------------
* H I at T=1e4 K: U = 13.6 eV / k_B / 1e4 K = 15.8.
    beta_HI ~ 7e-16 cm^3/s -- strongly suppressed because U >> 1.
* H I at T=2e5 K: U = 0.79.
    beta_HI = 2.91e-8 / (0.232 + 0.79) * 0.79^0.39 * exp(-0.79)
            ~ 1.18e-8 cm^3/s -- enormous because U < 1 (the
    Boltzmann factor is exp(-0.79) ~ 0.45, not the suppressing tail).

Test strategy
-------------
* Smoke: finite + non-negative for the 10 followed coolant ions +
  H, He on a warm-to-hot T grid.
* Boltzmann sign: rate must GROW with T (endothermic). A sign-flip
  in `exp(-U)` would make the rate decrease with T.
* Cold cutoff: rate is exactly zero at T = 100 K (U >> 80 for any
  ion with dE > 1 eV).
* Spot-check: H I at T=2e5 K matches the Voronov fit evaluation to
  10%; H I at T=1e4 K is in [1e-17, 1e-14] (loose order-of-magnitude
  check, since the fit uncertainty at U ~ 16 is large).
* Isoelectronic ordering: at fixed T, beta_CI decreases with charge
  for the same element (higher q -> higher IP -> more endothermic).
"""

import numpy as np
import pytest

from pyathena.microphysics.ci_rate import CollIonRate


# (Z, N_reactant, label) -- (Z, N) = ion BEFORE ionization.
# H I: Z=1, N=1; HeI: Z=2, N=2; CI: Z=6, N=6; etc.
CI_CATALOG = [
    (1,  1, "H I"),
    (2,  2, "He I"),
    (2,  1, "He II"),
    (6,  6, "C I"),
    (6,  5, "C II"),
    (7,  7, "N I"),
    (7,  6, "N II"),
    (8,  8, "O I"),
    (8,  7, "O II"),
    (8,  6, "O III"),
    (16, 16, "S I"),
    (16, 15, "S II"),
    (16, 14, "S III"),
]


@pytest.fixture(scope="module")
def ci():
    return CollIonRate()


@pytest.fixture(scope="module")
def T_grid_warm():
    """T range where CI is non-negligible -- HII region and hotter."""
    return np.logspace(np.log10(1e4), np.log10(1e6), 11)


# ---------------------------------------------------------------------
# Smoke: finite + nonneg across warm T grid.
# ---------------------------------------------------------------------

@pytest.mark.parametrize("Z,N,label", CI_CATALOG)
def test_ci_rate_finite_nonneg(ci, T_grid_warm, Z, N, label):
    """Collisional ionization rate is finite and >= 0 on a grid that
    spans T = [1e4, 1e6] K (HII region through hot bubble interiors).

    >= 0 (not > 0): the U > 80 branch returns exactly 0 at low T,
    which is the intended behavior. Smoke test only catches missing
    data + NaN, not order-of-magnitude bugs.
    """
    rate = ci.get_ci_rate(Z, N, T_grid_warm)
    assert np.all(np.isfinite(rate)), f"{label}: ci non-finite"
    assert np.all(rate >= 0), f"{label}: ci < 0"


# ---------------------------------------------------------------------
# Boltzmann direction: CI is endothermic; rate must grow with T.
# At T << dE/k_B, rate is near zero (explicit U > 80 branch returns 0).
# At T >> dE/k_B, rate plateaus.
# ---------------------------------------------------------------------

@pytest.mark.parametrize("Z,N,label", [
    (1, 1, "H I"),     # dE = 13.6 eV / k_B ~ 1.58e5 K
    (2, 2, "He I"),    # dE = 24.6 eV
    (8, 8, "O I"),     # dE = 13.6 eV
])
def test_ci_rate_grows_with_T(ci, Z, N, label):
    """Endothermic CI rate must increase with T.

    Sign of the Boltzmann factor `exp(-U)` in the Voronov fit
    (`ci_rate.py:39`): U = dE / k_B / T, so U decreases as T grows,
    so exp(-U) -> 1, so the rate grows. A sign flip
    (`exp(+U)` by mistake) would diverge at low T and vanish at
    high T -- this test catches that immediately.

    Tested with H I, He I, O I where dE / k_B > T_lo so the rate is
    in the suppressed regime at T = 1e4 and the unsuppressed regime
    at T = 1e6.
    """
    T_lo = 1.0e4
    T_hi = 1.0e6
    rate_lo = ci.get_ci_rate(Z, N, T_lo)
    rate_hi = ci.get_ci_rate(Z, N, T_hi)
    assert rate_hi > rate_lo, (
        f"{label}: ci should grow with T. "
        f"rate(T=1e4)={rate_lo:.3e}, rate(T=1e6)={rate_hi:.3e}"
    )


# ---------------------------------------------------------------------
# Cold cutoff: at T << dE/k_B the U > 80 branch returns exactly zero.
# H I dE = 1.58e5 K, so at T = 1e3 K, U = 158 -> rate must be exactly 0.
# ---------------------------------------------------------------------

@pytest.mark.parametrize("Z,N,label", [
    (1, 1, "H I"),
    (2, 1, "He II"),   # dE = 54.4 eV -> 6.3e5 K threshold
    (8, 7, "O II"),    # dE = 35.1 eV
])
def test_ci_rate_zero_at_cold_T(ci, Z, N, label):
    """At T = 100 K, every ion in this test has U = dE_Kel/T > 80,
    triggering the cutoff branch `ci_rate.py:36-37` that returns
    exactly 0. Pins this behavior so any future refactor of the
    cutoff (e.g., switching to a `T -> 0` asymptote) is intentional.

    Why a cutoff at U=80: `exp(-80) ~ 1.8e-35`, which underflows to
    0 in double precision after the `(1+P*U^0.5)/(X+U)*U^K` prefactor
    multiplies in -- but the explicit branch is cleaner and avoids
    any platform-dependent denormal behavior.
    """
    T_cold = 1.0e2
    rate = ci.get_ci_rate(Z, N, T_cold)
    assert rate == 0.0, (
        f"{label}: ci at T=100 K should be exactly 0 "
        f"(U > 80 branch); got {rate:.3e}"
    )


# ---------------------------------------------------------------------
# Reference value: H I at T = 2e5 K. This is in the regime where
# U = 13.6 eV * 1.16e4 K/eV / 2e5 K = 0.79, so the rate is large.
# Voronov 1997 Table 1: at T = 2e5 K, beta_HI ~ 1.2e-9 cm^3/s.
# ---------------------------------------------------------------------

def test_HI_ci_rate_at_T200000(ci):
    """H I collisional ionization at T = 2e5 K.

    Why this T: U = 13.6 eV / k_B / 2e5 K = 0.79. The Voronov fit
    parameters for H I (from Cloudy `coll_ion.dat`) are A=2.91e-8,
    P=0, X=0.232, K=0.39. Direct evaluation:
      beta = 2.91e-8 * 1 / (0.232 + 0.79) * 0.79^0.39 * exp(-0.79)
           = 2.91e-8 * 0.978 * 0.911 * 0.454
           ~ 1.18e-8 cm^3/s

    This is the "knee" of the H ionization curve -- at T ~ 1-2e5 K,
    collisional ionization of H starts to dominate over photoionization
    in the absence of an external UV field (the standard CIE regime).
    Critical reference point for the multi-ion sweep.

    Tolerance 10%: covers the published Voronov fit + any in-code
    rounding. Anything beyond 10% indicates a coefficient swap or a
    sign error in the Boltzmann factor.
    """
    T = 2.0e5
    rate = ci.get_ci_rate(1, 1, T)
    np.testing.assert_allclose(
        rate, 1.18e-8, rtol=0.1,
        err_msg=f"H I CI at T=2e5: got {rate:.3e}, "
                f"expected ~1.18e-8 (Voronov 1997 fit, U=0.79)"
    )


def test_HI_ci_rate_at_T10000(ci):
    """H I collisional ionization at T = 1e4 K is in the strongly
    suppressed Boltzmann tail (U ~ 16).

    Loose order-of-magnitude check (factor of 30 window centered on
    7e-16 cm^3/s). The exact value is dominated by exp(-U) which is
    sensitive to U at ~10 ppm level, so a tighter tolerance would
    fail for trivial reasons. The point is to verify the rate is
    in the right decade -- a missing exp factor or wrong dE would
    push it well outside.
    """
    T = 1.0e4
    rate = ci.get_ci_rate(1, 1, T)
    # Loose tolerance: empirical Voronov fit at 1e4 K is ~6e-16 cm^3/s,
    # but exact value depends on the fit's A/X/K parameters.
    assert 1e-17 < rate < 1e-14, (
        f"H I CI at T=1e4: got {rate:.3e}, expected ~6e-16 +/- decade"
    )


# ---------------------------------------------------------------------
# Ionization-stage ordering: at fixed T, beta(X^q) should DECREASE
# with q for a given element (higher charge -> higher IP -> more
# endothermic -> smaller rate at the same T).
# ---------------------------------------------------------------------

def test_plot_ci_rate_overview(ci, figures_dir, save_figures, ion_colors):
    """Diagnostic plot (no assertion): collisional ionization rate
    coefficient beta_CI(T) for a sample of ions across
    T = [1e3, 1e8] K.

    Shows the sharp exponential rise above each ion's threshold T
    (~ dE/k_B) and the plateau / slow falloff at very high T. The
    isoelectronic ordering (higher charge -> shifted to higher T)
    is also visible.

    Per-ion color follows the `photchem.py` convention via the
    `ion_colors(Z, q)` fixture.

    Skipped when `--no-figures`.
    """
    if not save_figures:
        pytest.skip("plot generation disabled (--no-figures)")
    import matplotlib.pyplot as plt
    T = np.logspace(3, 8, 400)
    # (Z, N_reactant, label). Plot a representative set per element.
    PLOT_IONS = [
        (1, 1, "H I"),
        (2, 2, "He I"),
        (2, 1, "He II"),
        (6, 6, "C I"),
        (6, 5, "C II"),
        (7, 7, "N I"),
        (8, 8, "O I"),
        (8, 7, "O II"),
        (8, 6, "O III"),
    ]
    fig, ax = plt.subplots(figsize=(7, 5))
    for Z, N, label in PLOT_IONS:
        q = Z - N      # reactant ion charge
        beta = ci.get_ci_rate(Z, N, T)
        ax.loglog(T, np.where(beta > 0, beta, np.nan),
                  label=label, color=ion_colors(Z, q), lw=1.2)
    ax.set_xlabel(r"$T\,[{\rm K}]$")
    ax.set_ylabel(r"$\beta_{\rm CI}\,[{\rm cm}^3\,{\rm s}^{-1}]$")
    ax.set_title("Voronov 1997 collisional ionization rates")
    ax.legend(fontsize="x-small", ncol=2, loc="lower right")
    ax.set_ylim(1e-15, 1e-7)
    fig.tight_layout()
    fig.savefig(figures_dir / "ci_rate_overview.png", dpi=200)
    plt.close(fig)


def test_O_isoelectronic_ci_decreases_with_charge(ci):
    """O I -> O II -> O III ionization rates at fixed T = 1e5 K:
    higher charge ions need more energy, so rate decreases.

    Ionization potentials:
      O I  -> O II:  13.6 eV
      O II -> O III: 35.1 eV
      O III-> O IV:  54.9 eV

    At T = 1e5 K (k_B*T = 8.6 eV), U values are 1.58, 4.08, 6.38.
    The Boltzmann factor exp(-U) drops by ~50x from O I to O III at
    this T -- the dominant ordering effect. The A/X/K prefactors
    have weaker (Z-dependent) trends.

    Why this test: catches a (Z, N) -> (Z, N+1) index swap in the
    data file or in `_read_data`. Such a swap would put the O III
    rate at the O II energy, etc., breaking the monotonicity here.
    """
    T = 1.0e5
    rate_OI   = ci.get_ci_rate(8, 8, T)
    rate_OII  = ci.get_ci_rate(8, 7, T)
    rate_OIII = ci.get_ci_rate(8, 6, T)
    assert rate_OI > rate_OII > rate_OIII, (
        f"CI rate should decrease with charge: "
        f"O I={rate_OI:.3e}, O II={rate_OII:.3e}, O III={rate_OIII:.3e}"
    )
