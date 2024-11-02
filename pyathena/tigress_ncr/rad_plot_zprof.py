import matplotlib.pyplot as plt
import matplotlib as mpl
import astropy.units as au
import astropy.constants as ac

import numpy as np
import cmasher as cmr
import cmocean.cm as cmo
from .rad_load_all import load_sim_ncr_rad_all, read_zpa_and_hst

# Some convenience functions (works for specific purposes only)
def plot_quantiles(ax, x, y, dim, dim_mean, q1=[0.16, 0.84], q2=None, plot_median=True,
                   plot_mean=False, color='k', lw=3):
    if q1 is not None:
        ax.fill_between(x, y.quantile(q1[0], dim=dim), y.quantile(q1[1], dim=dim),
                        alpha=0.2, lw=0, color=color)
    if q2 is not None:
        ax.fill_between(x, y.quantile(q2[0], dim=dim), y.quantile(q2[1], dim=dim),
                        alpha=0.15, lw=0, color=color)
    if plot_median:
        ax.plot(x, y.quantile(0.5, dim=dim), alpha=0.8, lw=lw, c=color)
    if plot_mean:
        ax.plot(x, y.mean(dim=dim_mean), alpha=1, lw=2, ls=':', c=color)

def plot_zprof2(model_set, model,
                sa=None, df=None, phase_set_name='warm_eq_LyC_ma',
                force_override=False, savefig=True):

    if sa is None or df is None:
        sa, df = load_sim_ncr_rad_all(model_set=model_set, verbose=False)

    s, zpa, h = read_zpa_and_hst(sa, df, model,
                                 phase_set_name=phase_set_name,
                                 force_override=force_override)

    u = s.u
    zpa = zpa.rename(dict(time='time_code'))
    zpa = zpa.assign_coords(tMyr=zpa.time_code*u.Myr)
    zpa = zpa.swap_dims(dict(time_code='tMyr'))
    zpa = zpa.assign_coords(z_kpc=zpa.z*u.kpc)

    # Non-photoionized and non-equilibrium
    zp_neq = zpa.sel(phase='w1_geq_noLyC')
    # Non-photoionized and equilibrium
    zp_eq = zpa.sel(phase='w1_eq_noLyC')
    # Exposed to radiation
    zp_pi = zpa.sel(phase='w1_eq_LyC_pi') + zpa.sel(phase='w1_eq_LyC') + zpa.sel(phase='w1_geq_LyC') + zpa.sel(phase='w1_geq_LyC_pi')

    # Exposed to strong radiation only (note that it is a subset of zp_pi)
    zp_pi_strong = zpa.sel(phase='w1_eq_LyC_pi')

    zp_dict = dict(zpa=zpa, zp_eq=zp_eq, zp_neq=zp_neq,
                   zp_pi=zp_pi, zp_pi_strong=zp_pi_strong)

    # zp0 = [zp_dict['zp_eq'], zp_dict['zp_pi'], zp_dict['zp_neq']]
    zp0 = [zp_dict['zp_pi'], zp_dict['zp_neq']]
    zp1 = [zp_dict['zp_eq'], zp_dict['zp_pi'], zp_dict['zp_neq']]
    cmap = [cmo.tempo,
            cmr.get_sub_cmap(cmo.balance, 0.0, 0.5).reversed(),
            cmr.get_sub_cmap(cmo.balance, 0.5, 1.0),
            mpl.cm.Wistia]
    # colors0 = [cmap[0](0.5), cmap[1](0.5), cmap[2](0.5)]
    colors0 = [cmap[1](0.5), cmap[2](0.5)]
    colors1 = [cmap[0](0.5), cmap[1](0.5), cmap[2](0.5)]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    axes = axes.flatten()

    # f_V
    [plot_quantiles(axes[0], zp['z_kpc'], zp['frac'], 'tMyr', 'tMyr', \
                    color=c) for zp, c in zip(zp0, colors0)]
    plt.setp(axes[0], ylim=(0, 0.5), yscale='linear', ylabel=r'$f_{\rm V}$')

    # xHII (ne^2-weighted)
    [plot_quantiles(axes[1], zp['z_kpc'], zp['xHII_w_nesq']/zp['frac_w_nesq'], 'tMyr',
                    'tMyr', color=c) for zp, c in zip(zp0, colors0)]

    l, = axes[1].plot(zp0[1]['z_kpc'],
                      (zp0[1]['xHII_eq_w_nesq']/zp0[1]['frac_w_nesq']).quantile(0.5, dim='tMyr'),
                      alpha=0.8, lw=1.5, c=colors0[1], ls='-')

    if model in ['R8_4pc', 'R8_8pc', 'R8_Z1']:
        axes[1].text(-1.4, 0.22, r'$\langle x_{\rm H^+,eq}\rangle_{n_e^2}$', fontsize=18)

    # axes[1].plot(zp1[0]['z_kpc'], (zp1[0]['xHII_eq_w_nesq']/zp1[0]['frac_w_nesq']).quantile(0.5, dim='tMyr'),
    #              alpha=0.8, lw=1.5, c=colors1[0], ls='-')

    plt.setp(axes[1], ylim=(0, 1.1), yscale='linear',
             ylabel=r'$\langle x_{\rm H^+} \rangle_{n_e^2}$')

    # Try sum_i \int xHII ne^2 Theta dA / sum_i \int ne^2 Theta dA, where i is the snapshot number
    # zp_whole = zp_dict['zpa'].sel(phase='whole')
    # zp = zp_dict['zp_pi']
    # axes[1].plot(zp['z_kpc'], (zp['xHII_eq_w_nesq']*zp_whole['nesq']).sum(dim='tMyr')/\
        #              (zp['frac_w_nesq']*zp_whole['nesq']).sum(dim='tMyr'),
    #              lw=5, c='k', alpha=0.3)

    # Temperature (ne^2-weighted)
    [plot_quantiles(axes[2], zp['z_kpc'], zp['T_w_nesq']/zp['frac_w_nesq'], 'tMyr', 'tMyr',
                    color=c) for zp, c in zip(zp0, colors0)]
    plt.setp(axes[2], ylim=(6000, 13000), yscale='linear',
             ylabel=r'$\langle T\rangle_{n_e^2}\;[{\rm K}]$')

    # ne
    [plot_quantiles(axes[3], zp['z_kpc'], zp['ne'], 'tMyr', 'tMyr',
                    color=c) for zp, c in zip(zp0, colors0)]
    # Overplot total electron
    plot_quantiles(axes[3], zp_dict['zpa'].sel(phase='whole')['z_kpc'],
                   zp_dict['zpa'].sel(phase='whole')['ne'], 'tMyr', 'tMyr', q1=None, color='k')
    plt.setp(axes[3], ylim=(1e-5, 1e0), yscale='log', ylabel=r'$\langle n_e \rangle\;[{\rm cm}^{-3}]$')

    # Overplot Reynolds layer
    if model in ['R8_4pc', 'R8_8pc', 'R8_Z1']:
        zz = np.linspace(-2.0, 2.0, 1000)
        axes[3].plot(zz, 0.025*np.exp(-np.abs(zz)), c='grey', ls='--', label='Reynolds')

    axes[3].legend([mpl.lines.Line2D([0],[0],c='k',ls='-',lw=3),
                    mpl.lines.Line2D([0],[0],c='grey',ls='--')], ['all', 'Reynolds layer'],
                   loc='upper center', ncols=2, fontsize=18, labelspacing=0.3, columnspacing=0.8)

    # ne^2
    [plot_quantiles(axes[4], zp['z_kpc'], zp['nesq'], 'tMyr', 'tMyr',
                    color=c) for zp, c in zip(zp0, colors0)]
    # ne^2 (ne^2-weighted)
    # [plot_quantiles(axes[4], zp['z_kpc'], zp['nesq_w_nesq']/zp['frac_w_nesq'], 'tMyr', 'tMyr',
    #                 color=c, lw=1.5) for zp, c in zip(zp0, colors0)]
    plt.setp(axes[4], ylim=(1e-6, 1e0), yscale='log',
             ylabel=r'$\langle n_e^2 \rangle\;[{\rm cm}^{-6}]$')
    # ylabel=r'$\langle n_e^2 \rangle$ and $\langle n_e^2 \rangle_{n_e^2}\;[{\rm cm}^{-6}]$')

    # Ionization parameter
    zz = zp_dict['zp_neq'] + zp_dict['zp_pi']
    hnu = (s.par['radps']['hnu_PH']*au.eV).cgs.value
    # axes[5].semilogy(zz['z_kpc'], zz['Erad_LyC'].mean(dim='tMyr')/hnu/zz['nH'].mean(dim='tMyr'), c='purple')
    # axes[5].semilogy(zz['z_kpc'], (zz['Erad_LyC']/hnu/zz['nH']).mean(dim='tMyr'), c='purple',ls='--')
    # axes[5].semilogy(zz['z_kpc'], zz['Uion'].mean(dim='tMyr')/zz['frac'].mean(dim='tMyr'), c='purple', ls=':')
    plot_quantiles(axes[5], zp0[0]['z_kpc'], zp0[0]['Uion_w_nesq']/zp0[0]['frac_w_nesq'],
                   'tMyr', 'tMyr', color=colors0[0])
    # axes[5].semilogy(zz['z'], zz['Erad_LyC'].quantile(0.5, dim='tMyr')/hnu/zz['nH'].quantile(0.5, dim='tMyr'))
    # plt.setp(axes[5], ylim=(3e-5, 3e-2), yscale='log',
    #          ylabel=r'$\langle U \rangle_{n_e^2}$ and $\frac{(\mathcal{E}_{\rm LyC}/h\nu)_{\rm mean}}{n_{\rm H,mean}}$')
    plt.setp(axes[5], ylim=(3e-5, 3e-2), yscale='log',
             ylabel=r'$\langle U \rangle_{n_e^2}$')

    from matplotlib.legend_handler import HandlerBase

    class AnyObjectHandler(HandlerBase):
        def create_artists(self, legend, orig_handle,
                           x0, y0, width, height, fontsize, trans):
            l1 = plt.Line2D([x0,y0+width],[0.5*height,0.5*height], c=orig_handle[0], lw=3, ls=orig_handle[1])
            l2 = plt.Line2D([x0,y0+width],[0.5*height,0.5*height], c=orig_handle[0], lw=15, alpha=0.2)
            return [l1, l2]

    axes[0].legend([(colors0[0],'-'), (colors0[1],'-')], [r'${\tt pi}$', r'${\tt neq}$'],
                   handler_map={tuple: AnyObjectHandler()}, fontsize=18, ncols=2, loc='upper center',
                   labelspacing=0.3, columnspacing=0.8)

    plt.setp(axes, xlim=(-1.5, 1.5), xlabel=r'$z\;[{\rm kpc}]$')

    if savefig:
        fig.savefig('/tigress/jk11/figures/NCR-RADIATION/zprof2-{0:s}-{1:s}.png'.\
                    format(model, phase_set_name), dpi=200, bbox_inches='tight')

    return fig, s, zp_dict
