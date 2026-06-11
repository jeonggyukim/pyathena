"""H + He CIE cooling function vs T with per-channel decomposition.

Uses CIE x_q(T) from data/microphysics/chianti_v11/ioneq_{H,He}.txt
to weight each cooling channel by ion fraction. Channels:

  BB:   collisional excitation -> line emission (HI Ly alpha, HeI,
        HeII line cooling)
  2g:   two-photon continuum (HI 2s, HeI/HeII 2s metastable decay)
  CI:   collisional ionization (gamma * E_ion thermal-energy loss)
  FB:   radiative recombination + cascade continuum
  FF:   bremsstrahlung
  Tot:  sum

Rate per volume for each channel:
  rate / volume = n_X^q * n_e * Lambda^chan_q(T)
Divided by n_H * n_e for display:
  Lambda^chan_q(T) / (nH ne) = A_X * x_q(T) * Lambda^chan_per_ion(T)

x_q(T) is read from the CIE ioneq table, so the figure shows the
full CIE cooling-curve structure (not a single fixed-x snapshot).
"""
import os
import warnings
import pytest


@pytest.fixture(scope='module', autouse=True)
def _xuvtop():
    os.environ.setdefault(
        'XUVTOP', os.path.expanduser('~/Dropbox/Projects/CHIANTI_db'))
    warnings.filterwarnings('ignore')


# Asplund09 / Draine 11 Table 1.4 abundances.
A_H = 1.0
A_He = 9.55e-2

# Ionization potentials in erg (Draine Appendix C).
EV2ERG = 1.602176634e-12
IP_HI   = 13.5984 * EV2ERG
IP_HeI  = 24.5874 * EV2ERG
IP_HeII = 54.4178 * EV2ERG
KB_CGS  = 1.380649e-16
E_LYA   = 10.2 * EV2ERG


# ---- Draine 2011 analytic formulas (cm^3 s^-1 ... or per-volume) --
def draine_LyA_per_HI_per_e(T):
    """H I Ly-alpha collisional excitation cooling per HI per e.

    Draine 2011 Eq 14.16 (effective collision strength) combined
    with the standard q_12 = beta * Y / (g_l * sqrt(T)) form;
    matches Spitzer 1978 Eq 5-46. Final formula:
        gamma_12 = 7.5e-9 * T_4^-0.5 * exp(-118400/T)  cm^3/s
    Times the Ly-alpha photon energy E_LyA = 10.2 eV per excitation.
    """
    import numpy as np
    T = np.asarray(T, dtype=float)
    gamma_12 = 7.5e-9 * (T / 1e4)**(-0.5) * np.exp(-1.184e5 / T)
    return gamma_12 * E_LYA


def draine_HI_CI_per_HI_per_e(T):
    """H I collisional ionization cooling per HI per e.

    Draine 2011 Eq 13.11 / Spitzer 1978 Eq 5-43 (Arnaud & Rothenflug
    1985 fit). Final formula:
        gamma_ci = 5.85e-11 * sqrt(T) * exp(-157809/T)
                   / (1 + sqrt(T/1e5))                cm^3/s
    Times IP_HI = 13.6 eV (binding energy removed from thermal pool
    per ionization event).
    """
    import numpy as np
    T = np.asarray(T, dtype=float)
    gamma_ci = (5.85e-11 * np.sqrt(T) * np.exp(-1.578e5 / T)
                / (1.0 + np.sqrt(T / 1e5)))
    return gamma_ci * IP_HI


def draine_HII_rec_per_HII_per_e(T):
    """H II radiative recombination cooling -- case-B narrow.

    Draine 2011 Eq 27.23:
        Lambda_rec^B = 0.685 * k_B * T * alpha_B(T)
    Counts only the thermal kinetic energy of the captured
    electron (~0.685 * k_B * T per recombination). The Balmer-
    continuum and cascade-line emission are accounted as separate
    channels (continuum + BB Ly-alpha / Balmer lines) to avoid
    double-counting.

    Hummer 1994 case-B recombination coefficient fit:
        alpha_B(T) ~ 2.59e-13 * T_4^(-0.833 - 0.034 * log10 T_4)
    """
    import numpy as np
    T = np.asarray(T, dtype=float)
    T4 = T / 1e4
    alpha_B = 2.59e-13 * T4**(-0.833 - 0.034 * np.log10(T4))
    return alpha_B * 0.685 * KB_CGS * T


def draine_FF_per_HII_per_e(T):
    """Free-free (bremsstrahlung) cooling for HII per HII per e.

    Draine 2011 Eq 27.24:
        Lambda_FF / (n_e n_X) = 1.42e-27 * g_ff * sqrt(T) * Z^2
    Adopt g_ff = 1.1 (typical for HII region temperatures); Z = 1
    for HII.
    """
    import numpy as np
    T = np.asarray(T, dtype=float)
    g_ff = 1.1
    return 1.42e-27 * g_ff * np.sqrt(T)


def draine_HeII_rec_per_HeII_per_e(T):
    """He II -> He I recombination cooling -- case-B narrow.

    Same 0.685 * k_B * T thermal-loss convention as Draine 27.23,
    applied to the He II case-B recombination coefficient. The
    alpha_B fit follows Verner & Ferland 1996 / Hummer & Storey
    1998 (analogous to Hummer 1994 H but for He II):
        alpha_B(He II) ~ 2.72e-13 * T_4^-0.789  cm^3/s
    """
    import numpy as np
    T = np.asarray(T, dtype=float)
    T4 = T / 1e4
    alpha_B = 2.72e-13 * T4**(-0.789)
    return alpha_B * 0.685 * KB_CGS * T


def draine_HeIII_rec_per_HeIII_per_e(T):
    """He III -> He II recombination cooling -- case-B narrow.

    Z=2 hydrogenic ion; same 0.685 * k_B * T convention.
    alpha_B fit from Hummer 1994 Z=2 hydrogenic scaling:
        alpha_B(He III) ~ 1.50e-12 * T_4^-0.700  cm^3/s
    (Verner & Ferland 1996 give compatible numbers within ~10%.)
    """
    import numpy as np
    T = np.asarray(T, dtype=float)
    T4 = T / 1e4
    alpha_B = 1.50e-12 * T4**(-0.700)
    return alpha_B * 0.685 * KB_CGS * T


def draine_FF_HeII_per_HeII_per_e(T):
    """FF for He II (charge Z = 1). Draine 27.24 with Z = 1; same
    formula as HII."""
    return draine_FF_per_HII_per_e(T)


def draine_FF_HeIII_per_HeIII_per_e(T):
    """FF for He III (charge Z = 2). Draine 27.24 with Z = 2:
    factor of 4 (Z^2) over the H II coefficient."""
    return 4.0 * draine_FF_per_HII_per_e(T)


def _load_cie_ioneq(element):
    """Load CIE ionization fractions x_q(log10 T) from the
    chianti_v11/ioneq_<X>.txt table built by build_ioneq.py.

    Returns (log_T, x_q array of shape (Z+1, nT)).
    """
    import numpy as np
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))),
        'data', 'microphysics', 'chianti_v11',
        f'ioneq_{element}.txt')
    arr = np.loadtxt(path)
    log_T = arr[:, 0]
    x_q = arr[:, 1:].T   # shape (Z+1, nT)
    return log_T, x_q


def _interp_xq(log_T_table, x_q_table, T_grid):
    """Bilinear-in-log-T interpolation of x_q onto target T grid."""
    import numpy as np
    out = np.zeros((x_q_table.shape[0], len(T_grid)))
    log_T_target = np.log10(T_grid)
    for q in range(x_q_table.shape[0]):
        out[q] = np.interp(log_T_target, log_T_table, x_q_table[q],
                           left=x_q_table[q, 0],
                           right=x_q_table[q, -1])
    return out


def _bb_per_ion_per_e(ion_name, T_grid, n_e):
    import ChiantiPy.core as ch
    import numpy as np
    try:
        ion = ch.ion(ion_name, temperature=T_grid,
                     eDensity=np.full(len(T_grid), n_e))
        ion.Abundance = 1.0
        ion.IoneqOne = np.ones(len(T_grid))
        ion.boundBoundLoss()
        return np.asarray(ion.BoundBoundLoss['rate']) / n_e
    except Exception:
        return np.zeros(len(T_grid))


def _two_g_per_ion_per_e(ion_name, T_grid, n_e):
    import ChiantiPy.core as ch
    import numpy as np
    try:
        ion = ch.ion(ion_name, temperature=T_grid,
                     eDensity=np.full(len(T_grid), n_e))
        ion.Abundance = 1.0
        ion.IoneqOne = np.ones(len(T_grid))
        ion.twoPhotonLoss()
        return np.asarray(ion.TwoPhotonLoss['rate']) / n_e
    except Exception:
        return np.zeros(len(T_grid))


def _ionization_rate(ion_name, T_grid):
    """Total collisional ionization rate coefficient [cm^3 / s]
    for the ion (sum over all destination channels), per ion per
    electron. ChiantiPy.ion.ionizRate() returns this directly."""
    import ChiantiPy.core as ch
    import numpy as np
    try:
        ion = ch.ion(ion_name, temperature=T_grid,
                     eDensity=np.full(len(T_grid), 1.0))
        ion.ionizRate()
        return np.asarray(ion.IonizRate['rate'])
    except Exception:
        return np.zeros(len(T_grid))


def _fb_per_recombining_ion_per_e(ion_name, T_grid):
    """Case-A FB cooling per recombining ion per electron, for the
    given ion_name (e.g. 'h_2' -> HII recombining). Total radiation
    emitted including LyC photons that re-ionize neutral H."""
    import ChiantiPy.core as ch
    import numpy as np
    try:
        cont = ch.continuum(ion_name, temperature=T_grid)
        cont.Abundance = 1.0
        cont.IoneqOne = np.ones(len(T_grid))
        cont.freeBoundLoss()
        return np.asarray(cont.FreeBoundLoss['rate'])
    except Exception:
        return np.zeros(len(T_grid))


def _fb_caseB_per_recombining_ion_per_e(ion_name, T_grid,
                                        lambda_cut=911.75):
    """Case-B FB cooling: integrate freeBound spectrum at
    wavelengths longer than the lambda_cut (default = Lyman limit
    911.75 Angstrom for HII). Excludes direct-to-ground (n=1)
    captures whose LyC photons re-ionize neutral H locally and
    do not actually cool the gas."""
    import ChiantiPy.core as ch
    import numpy as np
    try:
        # Wavelength grid spanning UV to far IR.
        wvl = np.logspace(2.0, 7.0, 5000)
        cont = ch.continuum(ion_name, temperature=T_grid)
        cont.Abundance = 1.0
        cont.IoneqOne = np.ones(len(T_grid))
        cont.freeBound(wvl, verner=True)
        em = cont.FreeBound['intensity']   # (n_T, n_wvl)
        mask = wvl > lambda_cut
        out = np.zeros(len(T_grid))
        for k in range(len(T_grid)):
            out[k] = 4.0 * np.pi * np.trapezoid(
                em[k][mask], x=wvl[mask])
        return out
    except Exception:
        return np.zeros(len(T_grid))


def _ff_per_ion_per_e(ion_name, T_grid):
    import ChiantiPy.core as ch
    import numpy as np
    try:
        cont = ch.continuum(ion_name, temperature=T_grid)
        cont.Abundance = 1.0
        cont.IoneqOne = np.ones(len(T_grid))
        cont.freeFreeLoss()
        return np.asarray(cont.FreeFreeLoss['rate'])
    except Exception:
        return np.zeros(len(T_grid))


def test_plot_HHe_CIE_cooling(figures_dir, save_figures):
    """Make cool_HHe_CIE.png: CIE-weighted per-channel H + He
    cooling function vs T."""
    if not save_figures:
        pytest.skip('plot generation disabled (--no-figures)')
    import matplotlib.pyplot as plt
    import numpy as np

    T_grid = np.logspace(np.log10(3e3), 5.0, 50)
    n_e = 1.0   # all per-ion-per-e quantities are n_e-independent
                # at this density for the FB/FF/CI channels (BB is
                # in the low-density limit at n_e=1).

    # CIE x_q(T) for H and He.
    log_T_H,  xH  = _load_cie_ioneq('H')
    log_T_He, xHe = _load_cie_ioneq('He')
    xH  = _interp_xq(log_T_H,  xH,  T_grid)   # (2, nT)
    xHe = _interp_xq(log_T_He, xHe, T_grid)   # (3, nT)
    x_HI, x_HII = xH[0], xH[1]
    x_HeI, x_HeII, x_HeIII = xHe[0], xHe[1], xHe[2]

    # Per-ion per-electron cooling channels.
    BB_HI    = _bb_per_ion_per_e('h_1', T_grid, n_e)
    twoG_HI  = _two_g_per_ion_per_e('h_1', T_grid, n_e)
    CI_HI    = _ionization_rate('h_1', T_grid) * IP_HI
    FB_HII   = _fb_per_recombining_ion_per_e('h_2', T_grid)
    FB_HII_B = _fb_caseB_per_recombining_ion_per_e('h_2', T_grid)
    FF_HII   = _ff_per_ion_per_e('h_2', T_grid)

    BB_HeI   = _bb_per_ion_per_e('he_1', T_grid, n_e)
    twoG_HeI = _two_g_per_ion_per_e('he_1', T_grid, n_e)
    CI_HeI   = _ionization_rate('he_1', T_grid) * IP_HeI
    BB_HeII  = _bb_per_ion_per_e('he_2', T_grid, n_e)
    twoG_HeII = _two_g_per_ion_per_e('he_2', T_grid, n_e)
    CI_HeII  = _ionization_rate('he_2', T_grid) * IP_HeII
    FB_HeII  = _fb_per_recombining_ion_per_e('he_2', T_grid)
    FB_HeIII = _fb_per_recombining_ion_per_e('he_3', T_grid)
    FF_HeII  = _ff_per_ion_per_e('he_2', T_grid)
    FF_HeIII = _ff_per_ion_per_e('he_3', T_grid)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # H panel: per relevant-ion per-electron rates (no x_q(T)
    # weighting). Each curve shows the intrinsic per-ion cooling
    # efficiency; channels with different source ions normalized
    # to their own ion (HI for BB/CI/2g, HII for FB/FF).
    ax = axes[0]
    ax.semilogy(T_grid, BB_HI,
                color='C0', lw=1.6,
                label=r'H I Ly$\alpha$ (BB) / $n_{\rm HI} n_e$')
    ax.semilogy(T_grid, twoG_HI,
                color='C0', lw=1.0, ls=':',
                label=r'H I 2$\gamma$ / $n_{\rm HI} n_e$')
    ax.semilogy(T_grid, CI_HI,
                color='C1', lw=1.6,
                label=r'H I coll. ioniz. / $n_{\rm HI} n_e$')
    # Adopted: Draine 27.23 case-B thermal cooling
    #   Lambda_rec = alpha_B(T) * 0.685 * k_B * T
    # ChiantiPy case-A (full freeBoundLoss) and case-B (integral
    # of freeBound spectrum at wvl > 912 A) shown as thin
    # reference curves -- they include cascade radiation that we
    # account for SEPARATELY in the BB / Ly-alpha channel to avoid
    # double-counting.
    FB_HII_draineB = draine_HII_rec_per_HII_per_e(T_grid)
    ax.semilogy(T_grid, FB_HII_draineB,
                color='C2', lw=1.6,
                label=r'H II rec (case-B Draine) / '
                      r'$n_{\rm HII} n_e$')
    ax.semilogy(T_grid, FB_HII,
                color='C2', lw=0.8, ls=':',
                label=r'H II rec (case-A ChiantiPy)')
    ax.semilogy(T_grid, FB_HII_B,
                color='C2', lw=0.8, ls='-.',
                label=r'H II rec (case-B ChiantiPy '
                      r'$\lambda>912$)')
    ax.semilogy(T_grid, FF_HII,
                color='C3', lw=1.2,
                label=r'H II free-free / $n_{\rm HII} n_e$')
    # Draine 2011 analytic overlays (per relevant ion per e).
    LyA_dr = draine_LyA_per_HI_per_e(T_grid)
    CI_dr  = draine_HI_CI_per_HI_per_e(T_grid)
    FB_dr  = draine_HII_rec_per_HII_per_e(T_grid)
    FF_dr  = draine_FF_per_HII_per_e(T_grid)
    ax.semilogy(T_grid, LyA_dr,
                color='C0', lw=3.5, ls='--', alpha=0.45)
    ax.semilogy(T_grid, CI_dr,
                color='C1', lw=3.5, ls='--', alpha=0.45)
    ax.semilogy(T_grid, FB_dr,
                color='C2', lw=3.5, ls='--', alpha=0.45)
    ax.semilogy(T_grid, FF_dr,
                color='C3', lw=3.5, ls='--', alpha=0.45)
    ax.set_title('Hydrogen (solid: ChiantiPy; dashed: Draine 27)')
    ax.set_xlabel(r'$T\,[{\rm K}]$')
    ax.set_ylabel(r'$\Lambda^{\rm chan}/(n_{X^q}\,n_e)\,'
                  r'[\rm erg\,cm^3\,s^{-1}]$')
    ax.set_xlim(4e3, 2e4)
    ax.set_xscale('linear')
    ax.set_ylim(1e-28, 1e-22)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize='small', loc='lower right',
              framealpha=0.85, handlelength=3.0)

    # He panel: per relevant-ion per-electron rates.
    ax = axes[1]
    ax.semilogy(T_grid, BB_HeI,
              color='C0', lw=1.4, label='He I (BB)')
    ax.semilogy(T_grid, twoG_HeI,
              color='C0', lw=1.0, ls=':',
              label=r'He I 2$\gamma$')
    ax.semilogy(T_grid, CI_HeI,
              color='C1', lw=1.4, ls='--',
              label=r'He I coll. ioniz.')
    ax.semilogy(T_grid, BB_HeII,
              color='C4', lw=1.4, label='He II (BB)')
    ax.semilogy(T_grid, twoG_HeII,
              color='C4', lw=1.0, ls=':',
              label=r'He II 2$\gamma$')
    ax.semilogy(T_grid, CI_HeII,
              color='C5', lw=1.4, ls='--',
              label=r'He II coll. ioniz.')
    # Case-B narrow Draine for HeII and HeIII rec (adopted);
    # ChiantiPy case-A FB shown as thin dotted for reference.
    FB_HeII_dr  = draine_HeII_rec_per_HeII_per_e(T_grid)
    FB_HeIII_dr = draine_HeIII_rec_per_HeIII_per_e(T_grid)
    ax.semilogy(T_grid, FB_HeII_dr,
              color='C2', lw=1.6,
              label=r'He II rec (case-B Draine)')
    ax.semilogy(T_grid, FB_HeII,
              color='C2', lw=0.8, ls=':',
              label=r'He II rec (case-A ChiantiPy)')
    ax.semilogy(T_grid, FB_HeIII_dr,
              color='C6', lw=1.6,
              label=r'He III rec (case-B Draine)')
    ax.semilogy(T_grid, FB_HeIII,
              color='C6', lw=0.8, ls=':',
              label=r'He III rec (case-A ChiantiPy)')
    ax.semilogy(T_grid, FF_HeII,
              color='C3', lw=1.0, ls='-.',
              label='He II free-free')
    ax.semilogy(T_grid, FF_HeIII,
              color='C3', lw=1.0, ls=':',
              label='He III free-free')
    ax.set_title('Helium (per relevant ion per e)')
    ax.set_xlabel(r'$T\,[{\rm K}]$')
    ax.set_xlim(4e3, 2e4)
    ax.set_xscale('linear')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize='small', loc='lower right',
              framealpha=0.85, handlelength=3.0, ncol=2)

    fig.suptitle(
        'H + He CIE cooling function vs T: per-channel '
        'decomposition using CHIANTI v11 ioneq')
    fig.tight_layout()
    fig.savefig(figures_dir / 'cool_HHe_CIE.png', dpi=150)
    plt.close(fig)
