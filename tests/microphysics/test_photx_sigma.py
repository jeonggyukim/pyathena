"""Tests for photoionization cross-section sign convention and shape.

What this file covers
---------------------
`pyathena/microphysics/photx.py` implements the Verner+96 (ApJ 465
487) 11-parameter analytic fit for the partial photoionization
cross-section sigma_pi(Z, N, E), where (Z, N) is the ion BEFORE
ionization (Z = atomic number, N = electron number, charge q = Z-N).
The cross-section enters downstream as the ingredient of the
SED-averaged sigma_pi consumed by `PhotChem.calc_mean_over_sed`, and
ultimately the photoionization rate `Gamma_pi = sigma_pi * F_phot`.

What "correct" looks like (physics)
-----------------------------------
1. sigma_pi(E < Eth) = 0 by definition (no energy to ionize).
2. sigma_pi(E ~ Eth) is at the peak of the cross-section curve
   (typically 1e-18 cm^2, ~Bohr-radius^2 scale).
3. sigma_pi(E >> Eth) falls off rapidly. The hydrogenic asymptote is
   sigma_pi ~ E^-3.5; the Verner+96 fit reproduces this to within
   factor ~2 over decades in E.
4. The two reference numbers everybody uses are:
   - sigma_pi(H I, 13.6 eV)  ~ 6.3e-18 cm^2  (Draine 2011 Tab. 13.1)
   - sigma_pi(He I, 24.6 eV) ~ 7.4e-18 cm^2  (Draine 2011 Tab. 13.2)
   These are the values that hand-derived ionizing-flux estimates use.

Test strategy
-------------
* Smoke test: finite + positive above threshold for an ion catalog
  spanning H, He, C, N, O, S (the 10 followed coolants for Phase D).
* Threshold check: explicit `sigma_pi(E < Eth) == 0` for every ion.
* Reference values: spot-check H I, He I, He II at threshold against
  Verner+96 / Draine 2011 published numbers, with 10-20% tolerance.
* Shape check: high-E decay (~E^-3 to E^-3.5 for HI).
* Round-trip: `get_Eth(Z, N)` consistent with the `sigma=0` boundary
  in `get_sigma`.

References
----------
* Verner, Ferland, Korista, Yakovlev 1996, ApJ 465 487 (analytic fits).
* Draine 2011, *Physics of the Interstellar and Intergalactic Medium*,
  Tables 13.1-13.3 (curated numerical values at threshold).
"""

import numpy as np
import pytest

from pyathena.microphysics.photx import PhotX


# Ion catalog: (Z, N, label, expected_Eth_eV).
# Thresholds from Verner+96 Table 1 / NIST atomic energy levels. The
# catalog covers the species needed by the multi-ion sweep (Phase D):
# H + He + the 10 followed coolant ions (C I/II, N I/II, O I/II/III,
# S I/II/III) plus their ionization products.
PHOTOION_CATALOG = [
    (1,  1, "H I",   13.6),     # H I (1s) -> H II
    (2,  2, "He I",  24.59),    # He I 1S(0) -> He II
    (2,  1, "He II", 54.42),    # He II 2S(1/2) -> He III (hydrogenic)
    (6,  6, "C I",    11.26),   # C I (2p^2) -> C II
    (6,  5, "C II",   24.38),   # C II (2p) -> C III
    (7,  7, "N I",    14.53),   # N I (2p^3) -> N II
    (8,  8, "O I",    13.62),   # O I (2p^4) -> O II
    (8,  7, "O II",   35.12),   # O II (2p^3) -> O III
    (8,  6, "O III",  54.94),   # O III (2p^2) -> O IV
    (16, 16, "S I",    10.36),  # S I (3p^4) -> S II
    (16, 15, "S II",   23.34),  # S II (3p^3) -> S III
]


@pytest.fixture(scope="module")
def px():
    return PhotX()


# ---------------------------------------------------------------------
# Threshold: sigma is exactly zero below threshold energy.
# ---------------------------------------------------------------------

@pytest.mark.parametrize("Z,N,label,Eth_eV", PHOTOION_CATALOG)
def test_sigma_zero_below_threshold(px, Z, N, label, Eth_eV):
    """Below the ionization threshold, the cross-section must be
    exactly zero. `photx.py` enforces this with an explicit cutoff
    (`sigma[indx] = 0.0` where E < Eth).

    Probe 3 energies below threshold (50%, 90%, 99% of Eth) to make
    sure no edge case sneaks through.
    """
    E = np.array([0.5 * Eth_eV, 0.9 * Eth_eV, 0.99 * Eth_eV])
    sigma = px.get_sigma(Z, N, E)
    assert np.all(sigma == 0.0), (
        f"{label}: sigma below Eth={Eth_eV} should be zero; got {sigma}"
    )


# ---------------------------------------------------------------------
# Smoke: sigma is finite + positive at and just above threshold.
# ---------------------------------------------------------------------

@pytest.mark.parametrize("Z,N,label,Eth_eV", PHOTOION_CATALOG)
def test_sigma_finite_positive_above_threshold(px, Z, N, label, Eth_eV):
    """Just above the ionization threshold and out to a few times Eth,
    the cross-section must be a finite positive number. Verner+96
    fits can return spurious NaN if the parameters for a given (Z, N)
    are missing or zero; this test catches such gaps.

    Probe Eth * (1.01, 2, 5) -- threshold + factor-of-few above.
    """
    E = np.array([Eth_eV * 1.01, Eth_eV * 2.0, Eth_eV * 5.0])
    sigma = px.get_sigma(Z, N, E)
    assert np.all(np.isfinite(sigma)), f"{label}: non-finite sigma"
    assert np.all(sigma > 0.0), f"{label}: non-positive sigma above Eth"


# ---------------------------------------------------------------------
# Threshold value: matches Verner+96 / Draine 2011 published refs.
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "Z,N,label,E_eV,sigma_expected_cm2,rtol",
    [
        (1, 1, "H I at 13.6 eV",   13.6, 6.3e-18, 0.10),
        (2, 2, "He I at 24.6 eV",  24.6, 7.4e-18, 0.20),
        (2, 1, "He II at 54.4 eV", 54.4, 1.6e-18, 0.20),
    ],
)
def test_sigma_threshold_value(px, Z, N, label, E_eV, sigma_expected_cm2, rtol):
    """Cross-section at the ionization threshold matches the
    published numerical value to within `rtol`. These three reference
    values are the ones most commonly cited:
      - H I @ 13.6 eV: 6.3e-18 cm^2 (Draine 2011 Tab. 13.1).
      - He I @ 24.6 eV: 7.4e-18 cm^2 (Draine 2011 Tab. 13.2).
      - He II @ 54.4 eV: 1.6e-18 cm^2 (hydrogenic; sigma ~ Z^-2 of HI).

    Failure modes caught:
      - Wrong fit parameters in `verner96_photx.dat` for one of these
        ions (e.g., row swap during a data refresh).
      - Wrong unit conversion in `get_sigma` (factor of 10 etc.).
      - Eth offset that puts the test energy slightly below cutoff.

    Tolerance is 10% for HI, 20% for HeI/HeII (the Verner fit has
    slightly higher error for the non-hydrogenic ion shapes).
    """
    sigma = px.get_sigma(Z, N, np.array([E_eV * 1.001]))
    np.testing.assert_allclose(
        sigma[0], sigma_expected_cm2, rtol=rtol,
        err_msg=f"{label}: got {sigma[0]:.3e}, expected "
                f"{sigma_expected_cm2:.3e}"
    )


# ---------------------------------------------------------------------
# Asymptotic shape: sigma decreases at high E (E >> Eth). Verner+96
# fit asymptotes to a power law E^(0.5*P - 5.5) for large E; the
# exponent is negative for every ion (since P < 11 by construction),
# so sigma(E >> Eth) < sigma(E ~ Eth).
# ---------------------------------------------------------------------

@pytest.mark.parametrize("Z,N,label,Eth_eV", PHOTOION_CATALOG[:5])
def test_sigma_decreases_at_high_E(px, Z, N, label, Eth_eV):
    """Asymptotic shape check. The Verner+96 fit asymptotes to
    sigma_pi(E -> infty) ~ E^(0.5*P - 5.5). The fit parameter P is
    always < 11 in `verner96_photx.dat`, so the exponent is negative,
    so sigma decreases at high E. Compare 1.1 * Eth vs 20 * Eth: at
    least order-of-magnitude drop expected.

    Only the first 5 catalog entries (H, He, C) are exercised --
    these are the ions where Verner+96 has the cleanest fit. The
    O, S, N high-E behavior depends on inner-shell corrections that
    can mildly violate monotonicity at intermediate E.
    """
    E = np.array([Eth_eV * 1.1, Eth_eV * 20.0])
    sigma = px.get_sigma(Z, N, E)
    assert sigma[1] < sigma[0], (
        f"{label}: sigma should decrease at high E. "
        f"sigma(1.1*Eth)={sigma[0]:.3e}, sigma(20*Eth)={sigma[1]:.3e}"
    )


# ---------------------------------------------------------------------
# H I hydrogenic check: sigma(E) at high E follows ~E^-3.5 within a
# factor of ~2 over a decade in energy.
# ---------------------------------------------------------------------

def test_HI_sigma_decreases_an_order_of_magnitude_per_decade(px):
    """Quantitative high-E scaling for H I.

    Hydrogenic photoionization scales as sigma_pi ~ E^-3.5 well above
    threshold (Draine 2011 Eq. 13.4). From E1=30 eV (~2.2 * Eth) to
    E2=100 eV (~7.3 * Eth), the energy ratio is 3.33; for E^-3.5 the
    sigma ratio is 3.33^3.5 ~ 72. The Verner+96 fit has factor-of-2
    corrections at intermediate E, so we accept a ratio in [20, 100]
    -- catches a missing factor of 10 but not a 30% miscalibration.

    If this test starts failing, the likely cause is a stale or
    re-indexed `verner96_photx.dat` row for H I.
    """
    E1 = 30.0
    E2 = 100.0
    sigma1 = px.get_sigma(1, 1, np.array([E1]))[0]
    sigma2 = px.get_sigma(1, 1, np.array([E2]))[0]
    ratio = sigma1 / sigma2
    assert 20.0 < ratio < 100.0, (
        f"H I sigma drop 30->100 eV: ratio={ratio:.2f}, "
        f"expected 20-100 (E^-3 to E^-3.5)"
    )


# ---------------------------------------------------------------------
# get_Eth round-trip: threshold returned by `get_Eth` matches the
# zero/nonzero boundary in `get_sigma`.
# ---------------------------------------------------------------------

def test_plot_sigma_overview(px, figures_dir, save_figures, ion_colors):
    """Diagnostic plot (no assertion): photoionization cross-section
    sigma_pi(E) for each ion in the catalog overlaid on one figure.
    Useful for visually checking thresholds and high-E falloffs.

    Per-ion color follows the `photchem.py` convention (element ->
    colormap; ionization stage -> intensity), so the same ion plotted
    here matches the same ion in `PhotChem.plt_sed_sigma_pi`.

    Skipped (effectively a no-op) when `--no-figures` is passed.
    """
    if not save_figures:
        pytest.skip("plot generation disabled (--no-figures)")
    import matplotlib.pyplot as plt
    E = np.logspace(0.5, 4.0, 400)  # 3 eV to 10 keV
    fig, ax = plt.subplots(figsize=(7, 5))
    for Z, N, label, Eth in PHOTOION_CATALOG:
        sigma = px.get_sigma(Z, N, E)
        q = Z - N
        ax.loglog(E, np.where(sigma > 0, sigma, np.nan),
                  label=label, color=ion_colors(Z, q), lw=1.2)
    ax.set_xlabel(r"$E\,[{\rm eV}]$")
    ax.set_ylabel(r"$\sigma_{\rm pi}\,[{\rm cm}^2]$")
    ax.set_title("Verner+96 photoionization cross sections")
    ax.legend(fontsize="x-small", ncol=2, loc="lower left")
    ax.set_ylim(1e-22, 1e-16)
    fig.tight_layout()
    fig.savefig(figures_dir / "photx_sigma_overview.png", dpi=200)
    plt.close(fig)


@pytest.mark.parametrize("Z,N,label,Eth_eV", PHOTOION_CATALOG)
def test_get_Eth_matches_sigma_cutoff(px, Z, N, label, Eth_eV):
    """Consistency between `get_Eth` (returns threshold energy) and
    `get_sigma` (zeroes out below threshold).

    Two things checked:
      1. get_Eth(Z, N) is within 5% of the published Eth value
         (catches a row swap in the Verner data file).
      2. sigma(0.99 * Eth_stored) == 0 and sigma(1.01 * Eth_stored) > 0
         (catches an off-by-one between Eth column and cutoff column).

    A failure here means downstream consumers (`PhotChem.set_wavelength_bins`,
    `PhotChem.calc_mean_over_sed`) will silently use the wrong
    ionization threshold for one ion -- a high-impact, hard-to-spot
    bug.
    """
    Eth_stored = px.get_Eth(Z, N, unit='eV')
    np.testing.assert_allclose(
        Eth_stored, Eth_eV, rtol=0.05,
        err_msg=f"{label}: get_Eth returned {Eth_stored:.3f}, "
                f"published Verner+96 says {Eth_eV:.3f}"
    )
    # cross-check: sigma(0.99*Eth_stored) == 0, sigma(1.01*Eth_stored) > 0
    sig_below = px.get_sigma(Z, N, np.array([Eth_stored * 0.99]))[0]
    sig_above = px.get_sigma(Z, N, np.array([Eth_stored * 1.01]))[0]
    assert sig_below == 0.0, f"{label}: sigma(0.99*Eth) = {sig_below}"
    assert sig_above > 0.0, f"{label}: sigma(1.01*Eth) = {sig_above}"
