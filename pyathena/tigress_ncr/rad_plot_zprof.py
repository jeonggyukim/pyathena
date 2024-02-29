import matplotlib.pyplot as plt
import numpy as np

from .rad_load_all import read_zpa_and_hst

# Some convenience functions (works for specific purposes only)
def plot_quantiles(ax, z, q, dim='time',
                   plt_25_75=True, plt_5_95=True, plt_mean=True, color='k'):
    plt.sca(ax)
    if plt_25_75:
        plt.fill_between(z, q.quantile(0.25, dim=dim),
                            q.quantile(0.75, dim=dim), alpha=0.2, lw=0, color=color)
    if plt_5_95:
        plt.fill_between(z, q.quantile(0.05, dim=dim),
                            q.quantile(0.95, dim=dim), alpha=0.15, lw=0, color=color)
    if plt_mean:
        plt.plot(z, q.mean(axis=0), alpha=1, lw=2, ls=':', c=color)

    plt.plot(z, q.quantile(0.5, dim=dim), alpha=0.8, lw=3, c=color)

def plot_quantiles_1(axes, zp, f, dim='time', plt_25_75=True, plt_5_95=True,
                     plt_mean=True, colors='k'):
    if len(np.atleast_1d(colors)) == 1:
        colors = np.repeat(colors, len(zp))

    for ax, z, color in zip(axes, zp, colors):
        plot_quantiles(ax, z['z'], z[f], dim=dim, plt_25_75=True, plt_5_95=True,
                       plt_mean=True, color=color)

def plot_quantiles_2(axes, zp, f, zp2, f2, dim='time', plt_25_75=True, plt_5_95=True,
                     plt_mean=True, colors='k'):
    if len(np.atleast_1d(colors)) == 1:
        colors = np.repeat(colors, len(zp))

    for ax, z, z2, color in zip(axes, zp, zp2, colors):
        plot_quantiles(ax, z['z'], z[f]/z2[f2], dim=dim,
                       plt_25_75=plt_25_75, plt_5_95=True, plt_mean=True, color=color)

def plot_zprof(sa, df, mdl, phase_set_name=None, force_override=False):

    read_zpa_and_hst(sa, df, mdl)

    zp_whole = zpa.sel(phase='whole')
    zp_w1_geq = zpa.sel(phase='w1_geq_noLyC') + zpa.sel(phase='w1_geq_LyC') +\
        zpa.sel(phase='w1_geq_LyC_pi')
    zp_w1_eq_noLyC = zpa.sel(phase='w1_eq_noLyC')
    zp_w1_eq_LyC = zpa.sel(phase='w1_eq_LyC')
    zp_w1_eq_LyC_pi = zpa.sel(phase='w1_eq_LyC_pi')

    zp_w2_geq = zpa.sel(phase='w2_geq_noLyC') + zpa.sel(phase='w2_geq_LyC') +\
        zpa.sel(phase='w2_geq_LyC_pi')
    zp_w2_eq_noLyC = zpa.sel(phase='w2_eq_noLyC')
    zp_w2_eq_LyC = zpa.sel(phase='w2_eq_LyC')
    zp_w2_eq_LyC_pi = zpa.sel(phase='w2_eq_LyC_pi')

    fig, axes = plt.subplots(6, 4, figsize=(20, 30), constrained_layout=True)
    for ax, title in zip(axes[0,:],['w1-eq-noLyC','w1-eq-LyC-pi', 'w1-neq', 'w1-eq-LyC']):
        ax.set_title(title)

    zp = [zp_w1_eq_noLyC, zp_w1_eq_LyC_pi, zp_w1_geq, zp_w1_eq_LyC]
    plot_quantiles_1(axes[0,:], zp, 'frac')
    plot_quantiles_2(axes[1,:], zp, 'nH', zp, 'frac')
    plot_quantiles_1(axes[2,:], zp, 'frac_w_nesq')
    plot_quantiles_2(axes[3,:], zp, 'nesq_w_nesq', zp, 'frac_w_nesq')
    plot_quantiles_2(axes[4,:], zp, 'xe_w_nesq', zp, 'frac_w_nesq')
    plot_quantiles_2(axes[5,:], zp, 'T_w_nesq', zp, 'frac_w_nesq')

    plt.setp(axes[0,:], xlim=(-1500,1500), yscale='linear', ylim=(0,1),
             ylabel=r'$f_V$')
    plt.setp(axes[1,:], xlim=(-1500,1500), yscale='log', ylim=(1e-3, 1e0),
             ylabel=r'$\langle n_{\rm H}\rangle /f_V\;[{\rm cm}^{-3}]$')
    plt.setp(axes[2,:], xlim=(-1500,1500), yscale='linear', ylim=(0, 1),
             ylabel=r'$f_{n_{\rm e}^2}$')
    plt.setp(axes[3,:], xlim=(-1500,1500), yscale='log', ylim=(1e-5, 1e2),
             ylabel=r'$\langle n_{\rm e}^2 \rangle_{n_{\rm e}^2}/f_{n_{\rm e}^2}\;[{\rm cm}^{-6}]$')
    plt.setp(axes[4,:], xlim=(-1500,1500), yscale='log', ylim=(1e-2, 1e0),
             ylabel=r'$\langle x_{\rm e} \rangle_{n_{\rm e}^2}/f_{n_{\rm e}^2}$')
    plt.setp(axes[5,:], xlim=(-1500,1500), yscale='linear', ylim=(0,1.5e4),
             ylabel=r'$\langle T \rangle_{n_{\rm e}^2}/f_{n_{\rm e}^2}\;[{\rm K}]$');

    plt.suptitle(mdl)
    fig.savefig('/tigress/jk11/figures/NCR-RADIATION/zprof-{0:s}-{1:s}.png'.\
                format(mdl, phase_set_name), dpi=200)

    return fig, axes
