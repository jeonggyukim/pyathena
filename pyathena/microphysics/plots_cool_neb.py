
# Requires cool_tigress module (separate repo)
import sys
sys.path.insert(0, '/tigress/jk11/code/tigress_cooling/python/')

import cool_tigress as ct

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cmocean
import astropy.units as au
import astropy.constants as ac

from .cool import coolRec, heatPE, coeff_alpha_rr_H

from ..plt_tools.set_plt import set_plt_fancy
from ..plt_tools.line_annotation import LineAnnotation

T = np.linspace(1e3, 1e5, num=1000)

def get_lc_tot(n_e, T, kind='Asplund', Z_g=1.0, NII_N=0.8, NeII_Ne=0.8,
               OII_O=0.8, CII_C=1.0, SIII_S=0.5):

    linecool = ct.LineCool(n_e=n_e, T=T, kind=kind,
                           CI_C=0.0, NI_N=0.0, OI_O=0.0, NeI_Ne=0.0, SI_S=0.0,
                           CII_C=CII_C, NII_N=NII_N, OII_O=OII_O, NeII_Ne=NeII_Ne,
                           SII_S=1.0 - SIII_S, SIII_S=SIII_S, Z_g=Z_g)
    lc = linecool.get_linecool_all()

    # Total cooling efficiency
    lc_tot = np.zeros_like(T)
    for k in lc.keys():
        lc_tot = lc_tot + lc[k]['tot']

    lc_tot_minus_C = lc_tot - (lc['CII']['tot'] + lc['CIII']['tot'])

    return linecool, lc, lc_tot, lc_tot_minus_C

def Lambda_neb_approx(T, nH, xe, xHII, Z_g):
    prefactor = (ac.h**2/ac.k_B**0.5/(2.0*np.pi*ac.m_e)**1.5).cgs.value
    xOstd = 3.2e-4
    E21 = (ac.h*ac.c/(3728.8*au.angstrom)).to('erg').value
    T21 = E21/ac.k_B.cgs.value
    g1 = 4
    x = np.log(T*1e-4)
    aa = [-0.0050817 ,  0.00765822,  0.11832144, -0.50515842,  0.81569592,
          -0.58648172,  0.69170381]
    Upsilon_fit = 10.0**(aa[0]*x**6 + aa[1]*x**5 + aa[2]*x**4 + aa[3]*x**3 +
                         aa[4]*x**2 + aa[5]*x + aa[6])
    fred = 1/(1 + 0.12*(nH*xe/1e2)**(0.38 - 0.12*np.log(T*1.0e-4)))

    #return Z_g*xHII*xOstd*xe*E21*prefactor/(T**0.5*g1)*np.exp(-T21/T)*Upsilon_fit*fred
    return Z_g*xHII*xOstd*xe*E21*prefactor/(T**0.5*g1)*np.exp(-T21/T)*Upsilon_fit*fred
    # return Z_g*xHII*xe/T**0.5*np.exp(-T21/T)*Upsilon_fit*fred

def plot_Lambda_nebular(Z_g1=1.0, Z_d1=1.0, Z_g2=0.1, Z_d2=0.1,
                        chi_PE=1e2, hnu_eV=3.45, annotate_arrow=True):
    from pyathena.microphysics import cool
    hnu = (hnu_eV*au.eV).cgs.value

    c_approx = 'C1'
    lw_approx = 2
    dashes_approx = [6,3]
    lw = 1.2
    set_plt_fancy()
    fig, axes = plt.subplots(1, 2, figsize=(15, 8), constrained_layout=True)

    label_all = [r'$1$', r'$10$', r'$10^2$',
                 r'$10^3$', r'$10^4$', r'$10^5$']
    nH = 1.0
    ne = 1.0
    linecool, lc, lc_tot, lc_tot_minus_C = get_lc_tot(ne, T, Z_g=Z_g1)

    # Photoionization heating: nH*Gamma_pi = nHI*zeta_pi*dhnu_HI
    # zeta_pi = sigma_pi*F_phot, where F_phot is the ionizing photon flux [cm^2 s^-1]
    # Assuming photoionization-recombination equilibrium, nHI*zeta_pi = alphaB*nHII*ne
    # Therefore, we have nH*Gamma_pi = alphaB*nHII*ne*dhnu_HI
    # What if we account for collisional/CR ionization..?
    # Need to calculate xHI and xe self-consistently

    # Gamma_pi_over_ne, but assume nH=nHII=ne
    Gamma_pi = coeff_alpha_rr_H(T)*hnu
    Gamma_pe1 = heatPE(nH, T, xe=1.0, Z_d=Z_d1, chi_PE=chi_PE)
    Gamma_pe1 -= coolRec(nH, T, xe=1.0, Z_d=Z_d1, chi_PE=chi_PE)
    # CII cooling (note that higher ionization is ignored)
    Lambda_CII1 = cool.coolCII(nH, T, xe=1.0, xHI=0.0, xH2=0.0, xCII=1.6e-4*Z_g1)/ne

    plt.sca(axes[0])
    # Total
    ltot1, = plt.plot(T/1e4, Lambda_CII1 +\
                      (lc_tot_minus_C + linecool.cooling_rr + linecool.cooling_ff)/ne,
                      c='k', ls='-', lw=4)
    # OII
    lOII1, = plt.plot(T/1e4, (lc['OII']['3728.8A'] + lc['OII']['3726.0A'])/ne,
                      c='k', ls='-', lw=lw)
    # lOII2, = plt.plot(T/1e4, (lc['OII']['2470.2A'] + lc['OII']['2470.3A'])/ne,
    #                   c=c_approx, ls='-', label='OIII OPT')

    # OIII
    lOIII1, = plt.plot(T/1e4, (lc['OIII']['88.36micron'] + lc['OIII']['51.81micron'])/ne,
                       c='k', ls='-', lw=lw)
    lOIII2, = plt.plot(T/1e4, (lc['OIII']['5008.2A'] + lc['OIII']['4960.3A'] \
                               # + lc['OIII']['4364.4A'] # small
                               # + lc['OIII']['2321.7A'] # small
                               )/ne, c='k', ls='-', lw=lw)

    # NII
    lNII1, = plt.plot(T/1e4, (lc['NII']['6549.9A']+lc['NII']['6585.3A'])/ne,
                      c='k', ls='-', lw=lw, label='NII OPT')
    # lNII1, = plt.plot(T/1e4, (lc['NII']['tot'])/ne, c='C3', ls='-', label='NII tot')

    # NeII
    lNeII1, = plt.plot(T/1e4, (lc['NeII']['tot'])/ne, c='k', ls='-', lw=lw)

    # SIII
    lSIII1, = plt.plot(T/1e4, (lc['SIII']['9071.1A']+lc['SIII']['9533.7A'])/ne,
                       c='k', ls='-', lw=lw)
    lSIII2, = plt.plot(T/1e4, (lc['SIII']['33.48micron']+lc['SIII']['87.13micron'])/ne,
                       c='k', ls='-', lw=lw)

    # lNII2, = plt.plot(T/1e4, (lc['NII']['6549.9A'])/ne, c='C3', ls='--', label='NII IR')
    # '6585.3A', '6549.9A', '5756.2A', '3063.7A', '122micron', '204micron'

    # Examine individual lines
    # plt.plot(T, (lc['OIII']['5008.2A'])/ne, c='C0', ls='--')
    # plt.plot(T, (lc['OIII']['4960.3A'])/ne, c='C0', ls='-.')
    # plt.plot(T, (lc['OIII']['4364.4A'])/ne, c='C0', ls=':') # small
    # plt.plot(T, (lc['OIII']['2321.7A'])/ne, c='C0', ls='--') # very small

    # CII (use our (NCR) cooling)
    lCII1, = plt.plot(T/1e4, Lambda_CII1, c='k', ls='-.', label='CII')

    # Recombination and free-free
    lrr, = plt.plot(T/1e4, linecool.cooling_rr/ne,
                    c='C0', ls=':', label='rad. rec.')
    lff, = plt.plot(T/1e4, linecool.cooling_ff/ne,
                    c='C0', ls='--', label='Free-free')

    # Heating by PI and PE
    lGammapipe, = plt.plot(T/1e4, Gamma_pi + Gamma_pe1,
                           c='k', ls='--', lw=3, alpha=1.0,
                           label=r'$\Gamma_{\rm pi+pe}/n_{\rm e}$')
    lGammapi, = plt.plot(T/1e4, Gamma_pi,
                         c='k', ls=':', lw=3, alpha=1.0,
                         label=r'$\Gamma_{\rm pi}/n_{\rm e}$')

    # Approximation
    lapprox, = plt.loglog(T/1e4, Lambda_CII1 +\
                          (linecool.cooling_ff + linecool.cooling_ff)/ne +\
                          Lambda_neb_approx(T, ne, xe=1.0, xHII=1.0, Z_g=Z_g1),
                          c=c_approx, lw=lw_approx,
                          dashes=dashes_approx, label='Total (approx.)')

    # Annotate lines
    axes[0].add_artist(LineAnnotation(r'[O II]3726,3729$\,\AA$',
                                      lOII1, 1.5))
    axes[0].add_artist(LineAnnotation(r'[O III]88.36,51.81$\,\mu {\rm m}$',
                                      lOIII1, 1.15))
    axes[0].add_artist(LineAnnotation(r'[O III]4960,5008$\,\AA$',
                                      lOIII2, 1.75))
    axes[0].add_artist(LineAnnotation(r'[N II]6550,6585$\,\AA$',
                                      lNII1, 1.75))
    axes[0].add_artist(LineAnnotation(r'[Ne II]12.81$\,\mu {\rm m}$',
                                      lNeII1, 1.75))
    axes[0].add_artist(LineAnnotation(r'[C II]157.6$\,\mu {\rm m}$',
                                      lCII1, 1.78))
    axes[0].add_artist(LineAnnotation(r'[S III]9071,9533$\,\AA$',
                                      lSIII1, 1.75))
    axes[0].add_artist(LineAnnotation(r'[S III]33.48,87.13$\,\mu {\rm m}$',
                                      lSIII2, 1.65))

    plt.sca(axes[1])
    lines = []
    ne_all = [1e0, 1e2, 1e4, 1e6]
    cmap = cmocean.cm.turbid_r
    norm = mpl.colors.LogNorm(1e0, 1e8)
    for i, (ne,label) in enumerate(zip(ne_all, label_all)):
        if i == 0:
            c = 'k'
        else:
            c = cmap(norm(ne))
        linecool, lc, lc_tot, lc_tot_minus_C = get_lc_tot(ne, T, Z_g=Z_g1)
        Lambda_CII1 = cool.coolCII(ne, T, xe=1.0, xHI=0.0, xH2=0.0, xCII=1.6e-4*Z_g1)/ne
        l_, = plt.loglog(T/1e4, Lambda_CII1 +\
                         (lc_tot_minus_C +\
                          linecool.cooling_ff + linecool.cooling_ff)/ne,
                         ls='-', lw=4, c=c)
        lines.append(l_)

    # Approximate
    for ne, label in zip(ne_all, label_all):
        linecool, lc, lc_tot, lc_tot_minus_C = get_lc_tot(ne, T, Z_g=Z_g1)
        Lambda_CII1 = cool.coolCII(ne, T, xe=1.0, xHI=0.0, xH2=0.0,
                                   xCII=1.6e-4*Z_g1)/ne
        plt.loglog(T/1e4, Lambda_CII1 +\
                   (linecool.cooling_ff + linecool.cooling_ff)/ne +\
                   Lambda_neb_approx(T, ne, xe=1.0, xHII=1.0, Z_g=Z_g1),
                   c=c_approx, lw=lw_approx, dashes=dashes_approx)

    # Z2
    ne = 1.0
    cmap = cmocean.cm.haline
    norm = mpl.colors.LogNorm(1e0, 1e8)
    for i, (ne, label) in enumerate(zip(ne_all, label_all)):
        linecool, lc, lc_tot, lc_tot_minus_C = get_lc_tot(ne, T, Z_g=Z_g2)
        Lambda_CII2 = cool.coolCII(ne, T, xe=1.0, xHI=0.0, xH2=0.0, xCII=1.6e-4*Z_g2)/ne
        l_, = plt.loglog(T/1e4, Lambda_CII2 + (linecool.cooling_ff + linecool.cooling_ff)/ne + \
                         Lambda_neb_approx(T, ne, xe=1.0, xHII=1.0, Z_g=Z_g2),
                         ls='-', lw=4, c=cmap(norm(ne)))
        plt.loglog(T/1e4, Lambda_CII2 + (linecool.cooling_ff + linecool.cooling_ff)/ne + \
                   Lambda_neb_approx(T, ne, xe=1.0, xHII=1.0, Z_g=Z_g2),
                   c=c_approx, lw=lw_approx, dashes=dashes_approx)
        lines.append(l_)

    lrr2, = plt.plot(T/1e4, linecool.cooling_rr/ne, c='C0', ls=':', label='rad. rec.')
    lff2, = plt.plot(T/1e4, linecool.cooling_ff/ne, c='C0', ls='--', label='Free-free')

    # Heating by PI and PE
    plt.plot(T/1e4, Gamma_pi, c='k', ls=':', lw=3, alpha=1, label=r'$\Gamma_{\rm pi}/n_{\rm e}$')
    # Z=1
    Gamma_pe1 = heatPE(1.0,T,1.0, Z_d=Z_d1, chi_PE=chi_PE)
    Gamma_pe1 -= coolRec(1.0, T, 1.0, Z_d=Z_d1, chi_PE=chi_PE)
    lGammapipe1, = plt.plot(T/1e4, Gamma_pi + Gamma_pe1,
                            c='k', ls='--', lw=3, alpha=1.0,
                            label=r'$\Gamma_{\rm pi+pe}/n_{\rm e}$')

    # Z=0.1
    Gamma_pe2 = heatPE(1.0, T, 1.0, Z_d=Z_d2, chi_PE=chi_PE)
    Gamma_pe2 -= coolRec(1.0, T, 1.0, Z_d=Z_d2, chi_PE=chi_PE)
    lGammapipe2, = plt.plot(T/1e4, Gamma_pi + Gamma_pe2,
                            c='k', ls='--', lw=3, alpha=1.0,
                            label=r'$\Gamma_{\rm pi+pe}/n_{\rm e}$')

    axes[1].add_artist(LineAnnotation(r'free-free', lff2, 0.64))
    axes[1].add_artist(LineAnnotation(r'recomb.', lrr2, 0.64))

    if annotate_arrow:
        axes[1].annotate(r'$Z_{\rm d}^{\prime}=1$', xy=(0.48, 3.8e-24),
                         xytext=(2, 50), textcoords='offset points',
                         arrowprops=dict(arrowstyle="->", facecolor='k',
                                         edgecolor='k', lw=1.5),
                         horizontalalignment='left')
        axes[1].annotate(r'$Z_{\rm d}^{\prime}=0.1$', xy=(0.57, 2.3e-24),
                         xytext=(2, 50), textcoords='offset points',
                         arrowprops=dict(arrowstyle="->", facecolor='k',
                                         edgecolor='k', lw=1.5),
                         horizontalalignment='left')
                         # connectionstyle="angle3,angleA=0,angleB=-90"));
    else:
        label_heat1 = r'$Z_{\rm d}^{\prime}=$' + r'{0:g}'.format(Z_d1)
        label_heat2 = r'$Z_{\rm d}^{\prime}=$' + r'{0:g}'.format(Z_d2)
        axes[1].add_artist(LineAnnotation(label_heat1, lGammapipe1, 0.5))
        axes[1].add_artist(LineAnnotation(label_heat2, lGammapipe2, 0.5))

    leg0 = axes[0].legend([ltot1, lapprox, lGammapi, lGammapipe],
                         [r'${\rm All}$', r'${\rm All}$ (approx.)',
                          r'$\Gamma_{\rm pi}/n_{\rm e}$',
                          r'$\Gamma_{\rm pi+pe}/n_{\rm e} - \Lambda_{\rm gr,rec}$'],
                          handlelength=3.0, columnspacing=1.0, loc=4, ncol=2)

    label = [r'$1\;$', r'$10^2\;$', r'$10^4\;$', r'$10^6\;$']
    label_all = [l + r'({0:g})'.format(Z_g1) for l in label] +\
        [l + r'({0:g})'.format(Z_g2) for l in label]
    leg1 = axes[1].legend(lines, label_all, loc=4, ncol=2, fontsize=15, labelspacing=0.4,
                         title=r'$n_{\rm e}/{\rm cm}^{-3}\;(Z_{\rm g}^{\prime})$')

    axes[0].add_artist(leg0)
    axes[1].add_artist(leg1)
    title = r'$n=n_{\rm e}=1\,{\rm cm}^{-3},\,'
    title += r'Z_{\rm g}^{\prime}=Z_{\rm d}^{\prime}=$' + '{:g}'.format(Z_g1)
    axes[0].set_title(title)

    # Set scales, limits, labels
    plt.setp(axes[0],xscale='linear', yscale='log',
             xlim=(4e3/1e4, 2e4/1e4),ylim=(1e-26, 4e-23))
    plt.setp(axes[1],xscale='linear', yscale='log',
             xlim=(4e3/1e4, 2e4/1e4), ylim=(1e-26, 4e-23))

    for ax in axes:
        ax.set_xlabel(r'$T\;[10^4\,{\rm K}$]')
        ax.set_ylabel(r'$\Lambda\;[{\rm erg}\,{\rm cm}^{3}\,{\rm s}^{-1}]$')
        ax.set_ylim(2e-26, 2e-23)

    # plt.savefig('/tigress/jk11/figures/NEWCOOL/fig-cool-photoionized-updated.pdf',png=200)
