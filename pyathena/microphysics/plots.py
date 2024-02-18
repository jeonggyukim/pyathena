import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import astropy.constants as ac
import astropy.units as au

from .get_xe_eq import get_xHII
from .cool_gnat12 import CoolGnat12
from .rec_rate import RecRate
from .cool \
import get_xCII, coeff_kcoll_H, coeff_alpha_rr_H, coeff_alpha_gr_H, coolHI, \
    coolRec, coolCII, coolOI, coolHIion
from .cool import coolCII, coolneb, coolLya, coolrecH, coolffH, coolHI
from .get_cooling import f1, set_CIE_interpolator

from pyathena import set_plt_fancy

def f_xe(xe, xH2, nH, T, xi_CR, G_CI, G_PE, Z_d, Z_g, zeta_pi, xOstd, xCstd,
         gr_rec=True):
    """Calculate free electron abundance xe for a given estimate of xe Need to use root
    finding to find converged solution.

    """
    xCII_eq = get_xCII(nH, xe, xH2, T, Z_d, Z_g, xi_CR, G_PE, G_CI, xCstd, gr_rec)
    xHII_eq = get_xHII(nH, xe, xH2, xCII_eq, T, xi_CR, G_PE, Z_d, zeta_pi, gr_rec)
    xOII_eq = xHII_eq*xOstd

    return xe - xHII_eq - xCII_eq - xOII_eq

def get_xe_arr(nH, T, xH2, xeM, xi_CR, G_PE, G_CI, zeta_pi, Z_d, Z_g, xCstd, xOstd,
               gr_rec):
    xHII = []
    xCII = []
    xOII = []
    for nH_, T_ in zip(nH, T):
        if T_ > 3.5e4:
            xHII_ = get_xHII(nH_, xe, xH2, xeM, T_, xi_CR, G_PE, Z_d, zeta_pi, gr_rec)
            xHII.append(xHII_)
            xCII.append(xCstd*Z_g)
            xOII.append(xHII_*Z_g*xOstd)
        else:
            xe = brentq(f_xe, 1e-3, 1.3, args=(xH2, nH_, T_, xi_CR, G_CI, G_PE, Z_d, Z_g,
                                               zeta_pi, xOstd, xCstd, gr_rec))
            xHII_ = get_xHII(nH_, xe, xH2, xeM, T_, xi_CR, G_PE, Z_d, zeta_pi, gr_rec)
            xCII_ = get_xCII(nH_, xe, xH2, T_, Z_d, Z_g, xi_CR, G_PE, G_CI, xCstd, gr_rec)

            xHII.append(xHII_)
            xCII.append(xCII_)
            xOII.append(xHII_*Z_g*xOstd)

    xHII = np.array(xHII)
    xCII = np.array(xCII)
    xOII = np.array(xOII)
    xOI = xOstd*Z_g - xOII
    xe = xHII + xCII + xOII

    return xe, xHII, xCII, xOII, xOI

def get_cool(nH, cgi_xe_He, cgi_xe_mHHe, xCstd=1.6e-4, xOstd=3.2e-4, Z_g=1.0, Z_d=1.0,
             G_CI=1.0, G_PE=1.0, xH2=0.0, xi_CR=2e-16, gr_rec=True, ionized=False):

    if ionized:
        # Optically thin ionizing photon flux assuming distance and Q
        Q = 1e49
        r = (5.0*au.pc).cgs.value
        sigma_pi = 3e-18
        zeta_pi = Q/(4.0*np.pi*r**2)*sigma_pi
    else:
        zeta_pi = 0.0

    # print('zeta_pi: {0:.3e}'.format(zeta_pi))

    xeM0 = xCstd*Z_g
    T = np.logspace(3, 8, 1201)
    nH = np.full_like(T, nH)
    xe_eq, xHII_eq, xCII_eq, xOII_eq, xOI_eq =\
        get_xe_arr(nH, T, xH2, xeM0, xi_CR, G_PE, G_CI, zeta_pi, Z_d, Z_g, xCstd, xOstd,
                   gr_rec=gr_rec)

    xHI_eq = 1.0 - xHII_eq
    xOII_eq = np.array(xOII_eq)
    xOI_eq = xOstd*Z_g - xOII_eq

    # print(cgi_xe_mHHe(T))
    # Add CIE electron abundance resulting from He and metals
    xe_eq += f1(T)*(cgi_xe_He(T) + Z_g*cgi_xe_mHHe(T))

    cool_H = coolLya(nH,T,xHI_eq,xe_eq)/nH +  coolffH(nH,T,xe_eq,xHII_eq)/nH + \
             coolrecH(nH,T,xe_eq,xHII_eq)/nH + coolHIion(nH,T,xe_eq,xHI_eq)/nH
    cool_Hffrec = coolffH(nH,T,xe_eq,xHII_eq)/nH + coolrecH(nH,T,xe_eq,xHII_eq)/nH
    cool_HLya = coolLya(nH,T,xHI_eq,xe_eq)/nH
    cool_CII_ = coolCII(nH,T,xe_eq,xHI_eq,0.0,xCII_eq)/nH
    cool_OI_ = coolOI(nH,T,xe_eq,xHI_eq,0.0,xOI_eq)/nH
    cool_neb_ = coolneb(nH,T,xe_eq,xHII_eq,Z_g)
    cool_other =  cool_CII_ + cool_OI_ + cool_neb_
    if gr_rec:
        cool_grRec = coolRec(nH,T,xe_eq,Z_d,G_PE)/nH
        cool_other += cool_grRec
    else:
        cool_grRec = 0.0

    res = dict(T=T, xe_eq=xe_eq, xHI_eq=xHI_eq, xHII_eq=xHII_eq,
               xCII_eq=xCII_eq, xOI_eq=xOI_eq, xOII_eq=xOII_eq,
               cool_H=cool_H, cool_other=cool_other,
               cool_CII=cool_CII_, cool_OI=cool_OI_,
               cool_neb=cool_neb_,
               cool_grRec=cool_grRec,
               cool_Hffrec=cool_Hffrec, cool_HLya=cool_HLya)

    return res

def plot_lambda_cool_ncr():
    set_plt_fancy()

    cg = CoolGnat12(abundance='Asplund09')
    cgi_metal, cgi_He, cgi_H, cgi_xe_mH, cgi_xe_mHHe, cgi_xe_He, cgi_xe_H =\
        set_CIE_interpolator(return_xe=True, return_xe_H=True,
                             return_Lambda_hydrogen=True)

    Q = 1e49
    r = (5.0*au.pc).cgs.value
    sigma_pi = 3e-18
    zeta_pi = Q/(4.0*np.pi*r**2)*sigma_pi
    print('zeta_pi:', zeta_pi)

    fig, axes = plt.subplots(2,1,figsize=(12,11),constrained_layout=True)

    ########################################################################
    ## Solar metallicity
    ########################################################################
    nH = 1.0
    Z_g = 1.0
    Z_d = 1.0
    nH = 1.0
    gr_rec = True
    c_other = 'C2'
    lw_other = 1.5

    plt.sca(axes[0])
    # Ieutral
    ls = '-'
    r0 = get_cool(nH, cgi_xe_He, cgi_xe_mHHe, xCstd=1.6e-4, xOstd=3.2e-4, Z_g=Z_g,
                  Z_d=Z_d, G_CI=1.0, G_PE=1.0, xH2=0.0, xi_CR=2e-16, gr_rec=gr_rec,
                  ionized=False)
    r0_Z1 = r0
    T = r0['T']
    xe_eq1 = r0['xe_eq']
    cool_tot1 = r0['cool_H'] + (1.0 - f1(T))*r0['cool_other'] + \
        f1(T)*(Z_g*cgi_metal(T)+cgi_He(T))
    l1, = plt.loglog(T, r0['cool_Hffrec'], ls=ls, c='C0',
                     label=r'$\Lambda_{\rm H}$')
    l1, = plt.loglog(T, r0['cool_HLya'], ls=ls, c='C5',
                     label=r'$\Lambda_{\rm H,Ly\alpha}$')
    l2, = plt.loglog(T, f1(T)*(Z_g*cgi_metal(T)+cgi_He(T)), ls=ls, c='C1',
                     label=r'$\Lambda_{\rm CIE,He+Metals}$')

    l4, = plt.loglog(T, (1.0 - f1(T))*r0['cool_CII'], ls=ls, c='C3', lw=lw_other)
    l5, = plt.loglog(T, (1.0 - f1(T))*r0['cool_grRec'], ls=ls, c='C4', lw=lw_other)
    l6, = plt.loglog(T, (1.0 - f1(T))*r0['cool_OI'], ls=ls, c='tab:purple', lw=lw_other)
    l7, = plt.loglog(T, (1.0 - f1(T))*r0['cool_neb'], ls=ls, c='saddlebrown', lw=lw_other)
    l3, = plt.loglog(T, (1.0 - f1(T))*r0['cool_other'], ls=ls, c=c_other,
                   label=r'$\Lambda_{\rm neb+OI+CII+GrRec}$')
    plt.loglog(T, cool_tot1, c='k', label=r'$\Lambda_{\rm tot}$', lw=3, ls=ls)
    plt.legend([l7,l6,l4,l5],[r'neb',r'${\rm O}$',r'${\rm C}^+$',r'grRec'],
               loc=(0.22,0.04),ncol=2,handlelength=1.0,
               handletextpad=0.3,columnspacing=0.5)

    # Ionized
    ls = '--'
    r1 = get_cool(nH, cgi_xe_He, cgi_xe_mHHe, xCstd=1.6e-4, xOstd=3.2e-4,
                  Z_g=1.0, Z_d=1.0, G_CI=100.0, G_PE=100.0, xH2=0.0, xi_CR=2e-16, gr_rec=gr_rec,
                  ionized=True)
    r1_Z1 = r1
    T = r1['T']
    xe_eq2 = r1['xe_eq']
    cool_tot2 = r1['cool_H'] + (1.0 - f1(T))*r1['cool_other'] + \
        f1(T)*(Z_g*cgi_metal(T)+cgi_He(T))

    l1, = plt.loglog(T, r1['cool_Hffrec'], ls=ls, c='C0')
    l1, = plt.loglog(T, r1['cool_HLya'], ls=ls, c='C5',
                     label=r'$\Lambda_{\rm H,Ly\alpha}$')
    l4,=plt.loglog(T, (1.0 - f1(T))*r1['cool_CII'], ls=ls, c='C3', lw=lw_other)
    l5,=plt.loglog(T, (1.0 - f1(T))*r1['cool_grRec'], ls=ls, c='C4', lw=lw_other)
    l6,=plt.loglog(T, (1.0 - f1(T))*r1['cool_OI'], ls=ls, c='tab:purple', lw=lw_other)
    l7,=plt.loglog(T, (1.0 - f1(T))*r1['cool_neb'], ls=ls, c='saddlebrown', lw=lw_other)
    l3,=plt.loglog(T, (1.0 - f1(T))*r1['cool_other'], ls=ls, c=c_other)
    plt.loglog(T, cool_tot2, c='k', lw=3, ls=ls)

    # #####################
    # # Add equilibrium xe
    # #####################
    inset_box = [0.65, 0.59, 0.29, 0.18]
    left, bottom, width, height = inset_box
    ax_add1 = fig.add_axes([left, bottom, width, height], zorder=100)
    plt.sca(ax_add1)
    plt.loglog(r0_Z1['T'],r0_Z1['xe_eq'], c='k', ls='-')
    plt.loglog(r1_Z1['T'],r1_Z1['xe_eq'], c='k', ls='--')
    plt.loglog(cg.temp, cgi_xe_H(cg.temp)+cgi_xe_mH(cg.temp), c='k', ls=':')

    plt.loglog(r0_Z1['T'], r0_Z1['xHI_eq'], c='r', ls='-')
    plt.loglog(r1_Z1['T'], r1_Z1['xHI_eq'], c='r', ls='--')
    temp_ = cg.temp[:29]
    plt.loglog(temp_, 1.0-cgi_xe_H(temp_), c='r', ls=':')

    fs = 14
    plt.setp(ax_add1, xlim=(5e3,1e5), ylim=(1e-5,2))
    ax_add1.set_xlabel(r'$T\;[{\rm K}]$', fontsize=fs, labelpad=2)
    ax_add1.set_ylabel(r'$x_{\rm e},\,x_{\rm H}$', fontsize=fs, labelpad=2)
    ax_add1.xaxis.set_tick_params(labelsize=fs)
    ax_add1.yaxis.set_tick_params(labelsize=fs)

    ax_add1.annotate(r'$x_{\rm e}$', (0.7,0.83), xycoords='axes fraction', fontsize=20)
    ax_add1.annotate(r'$x_{\rm H}$', (0.55,0.15), xycoords='axes fraction',
                     color='r',fontsize=20)
    ax_add1.xaxis.set_ticks_position('top')
    ax_add1.xaxis.set_label_position('top')

    ########################################################################
    ## Low metallicity
    ########################################################################
    plt.sca(axes[1])
    Z_g = 0.01
    Z_d = 0.01
    ls = '-'
    r0 = get_cool(nH, cgi_xe_He, cgi_xe_mHHe, xCstd=1.6e-4, xOstd=3.2e-4, Z_g=Z_g,
                  Z_d=Z_d, G_CI=1.0, G_PE=1.0, xH2=0.0, xi_CR=2e-16, gr_rec=gr_rec,
                  ionized=False)
    r0_Z001 = r0
    T = r0['T']
    xe_eq3 = r0['xe_eq']
    cool_tot3 = r0['cool_H'] + (1.0 - f1(T))*r0['cool_other'] + \
        f1(T)*(Z_g*cgi_metal(T)+cgi_He(T))

    l4,=plt.loglog(T, (1.0 - f1(T))*r0['cool_CII'], ls=ls, c='C3', lw=lw_other)
    l5,=plt.loglog(T, (1.0 - f1(T))*r0['cool_grRec'], ls=ls, c='C4', lw=lw_other)
    l6,=plt.loglog(T, (1.0 - f1(T))*r0['cool_OI'], ls=ls, c='tab:purple', lw=lw_other)
    l7,=plt.loglog(T, (1.0 - f1(T))*r0['cool_neb'], ls=ls, c='saddlebrown', lw=lw_other)

    l1, = plt.loglog(T, r0['cool_Hffrec'], ls=ls, c='C0',
                     label=r'$\Lambda_{\rm H}$')
    l1, = plt.loglog(T, r0['cool_HLya'], ls=ls, c='C5',
                     label=r'$\Lambda_{\rm H,Ly\alpha}$')
    l2, = plt.loglog(T, f1(T)*(Z_g*cgi_metal(T)+cgi_He(T)), ls=ls, c='C1',
                     label=r'$\Lambda_{\rm CIE,He+Metals}$')
    l3, = plt.loglog(T, (1.0 - f1(T))*r0['cool_other'], ls=ls, c=c_other,
                     label=r'$\Lambda_{\rm others}$')
    plt.loglog(T, cool_tot3, c='k', label=r'$\Lambda_{\rm tot}$', lw=3, ls=ls)

    # Ionized
    r1 = get_cool(nH, cgi_xe_He, cgi_xe_mHHe, xCstd=1.6e-4, xOstd=3.2e-4, Z_g=Z_g,
                  Z_d=Z_d, G_CI=100.0, G_PE=100.0, xH2=0.0, xi_CR=2e-16, gr_rec=gr_rec,
                  ionized=True)
    r1_Z001 = r0
    ls = '--'
    T = r1['T']
    xe_eq4 = r1['xe_eq']
    cool_tot4 = r1['cool_H'] + (1.0 - f1(T))*r1['cool_other'] + \
        f1(T)*(Z_g*cgi_metal(T)+cgi_He(T))
    l1, = plt.loglog(T, r1['cool_Hffrec'], ls=ls, c='C0',
                     label=r'$\Lambda_{\rm H}$')
    l1, = plt.loglog(T, r1['cool_HLya'], ls=ls, c='C5',
                     label=r'$\Lambda_{\rm H,Ly\alpha}$')

    l4,=plt.loglog(T, (1.0 - f1(T))*r1['cool_CII'], ls=ls, c='C3', lw=lw_other)
    l5,=plt.loglog(T, (1.0 - f1(T))*r1['cool_grRec'], ls=ls, c='C4', lw=lw_other)
    l6,=plt.loglog(T, (1.0 - f1(T))*r1['cool_OI'], ls=ls, c='tab:purple', lw=lw_other)
    l7,=plt.loglog(T, (1.0 - f1(T))*r1['cool_neb'], ls=ls, c='saddlebrown', lw=lw_other)

    l3,=plt.loglog(T, (1.0 - f1(T))*r1['cool_other'], ls=ls, c=c_other,
                   label=r'$\Lambda_{\rm neb+OI+CII+GrRec}$')
    plt.loglog(T, cool_tot4, c='k', label=r'$\Lambda_{\rm tot}$', lw=3, ls=ls)

    for ax in axes[:]:
        plt.sca(ax)
        plt.xlim(1e3,1e8)
        plt.ylim(1e-28,1e-21)
        plt.axvspan(2e4,3.5e4, alpha=0.2, color='grey')
        plt.grid(linestyle=':')
        plt.xlabel(r'$T\;[{\rm K}]$')
        plt.ylabel(r'$\mathcal{L}/n^2 \equiv \Lambda\;'+\
                   r'[{\rm erg}\,{\rm cm}^{3}\,{\rm s}^{-1}]$')

    axes[0].annotate(r'$Z^{\prime}=1$', (0.02,0.91), xycoords='axes fraction',
                     fontsize='large')
    axes[1].annotate(r'$Z^{\prime}=0.01$', (0.02,0.91), xycoords='axes fraction',
                     fontsize='large')

    cool_H = cgi_H(cg.temp)
    cool_He = cgi_He(cg.temp)
    cool_metal = cgi_metal(cg.temp)
    axes[0].plot(cg.temp, cool_H + cool_He + cool_metal, c='grey', ls=':')
    axes[1].plot(cg.temp, cool_H + cool_He + Z_g*cool_metal,
                 c='grey', ls=':')

    plt.sca(axes[0])
    arr_kwargs = dict(arrowprops=dict(arrowstyle='->',color='k', lw=1.5))
    plt.annotate(r'He+Metals (CIE)', (6e7,9e-24), (3e7,1e-22),
                 xycoords='data',textcoords='data', horizontalalignment='center',
                 verticalalignment='top', **arr_kwargs)
    plt.annotate(r'${\rm H}^+$(ff + rr)', (5e6,3e-24), (3e5,3.5e-24), xycoords='data',
                 textcoords='data')
    plt.annotate(r'H (Ly$\alpha$)', (5e6,3e-24), (2.3e5,2e-25), xycoords='data',
                 textcoords='data')
    plt.annotate(r'neb + ${\rm C}^+$ + ${\rm O}$ + ${\rm grRec}$', (3.0e4,5e-24),
                 (5e4,3e-23), xycoords='data', textcoords='data',) # somehow ha, va don't work
    plt.annotate(r'', (3.0e4,5e-24), (5e4,3e-23), xycoords='data', textcoords='data',
                 ha='left', **arr_kwargs)

    leg1 = axes[1].legend([mpl.lines.Line2D([0],[0],c='k',ls='-'),
                           mpl.lines.Line2D([0],[0],c='k',ls='--'),
                           mpl.lines.Line2D([0],[0],c='k',ls=':')],
                          [r'non-photoionized',
                           r'photoionized',
                           'CIE (Gnat & Ferland 2012)'],loc=1)
    # plt.tight_layout()

    rect = mpl.patches.Rectangle((inset_box[0]-0.075,inset_box[1]-0.015),
                                 inset_box[2]+0.075+0.02 , inset_box[3]+0.015+0.05,
                                 fill=True, color='w', alpha=1.0, zorder=2.5,
                                 transform=fig.transFigure, figure=fig)
    fig.patches.extend([rect])

    return fig

def get_den_t_cool(nH, Z, ionized):
    gamma = 5.0/3.0
    cgi_metal, cgi_He, cgi_xe_mH, cgi_xe_mHHe, cgi_xe_He =\
        set_CIE_interpolator(return_xe=True)
    rr = get_cool(nH, cgi_xe_He, cgi_xe_mHHe, xCstd=1.6e-4, xOstd=3.2e-4,
                  Z_g=Z, Z_d=Z, G_CI=1.0, G_PE=1.0, xH2=0.0, xi_CR=2e-16,
                  ionized=ionized, gr_rec=True)
    T = rr['T']
    cool_tot_ = rr['cool_H'] + (1.0 - f1(T))*rr['cool_other'] + \
               f1(T)*(Z*cgi_metal(T) + cgi_He(T))
    den_t_cool = (1.1 + rr['xe_eq'])*ac.k_B.cgs*T/((gamma-1.0)*cool_tot_)

    return rr, den_t_cool*(1.0*au.second).to('yr')

def plot_t_cool(nH=1.0, Z=[0.001, 0.01, 0.1, 1.0, 3.0]):

    fig, axes = plt.subplots(1, 1, figsize=(6, 5),
                             constrained_layout=True)
    cmap = mpl.cm.plasma_r
    norm = mpl.colors.LogNorm(0.003, 3.0)
    for Z_ in Z:
        rr1, t_cool1 = get_den_t_cool(nH, Z_, False)
        rr2, t_cool2 = get_den_t_cool(nH, Z_, True)
        plt.loglog(rr1['T'], t_cool1, c=cmap(norm(Z_)),
                   label=r'{0:g}'.format(Z_))
        plt.loglog(rr2['T'], t_cool2, c=cmap(norm(Z_)), ls='--')

    plt.xlabel(r'$T\;[{\rm K}]$')
    plt.ylabel(r'$n \times t_{\rm cool}\;[{\rm cm}^{-3}\,{\rm yr}]$')
    plt.xlim(3e3, 1e8)
    plt.ylim(1e3, 1e8)
    plt.axvspan(2e4, 3.5e4, alpha=0.2, color='grey')
    plt.grid(linestyle=':')
    plt.legend(loc=4, title=r'$Z^{\prime}$')

    return fig
