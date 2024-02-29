import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

from .rad_load_all import load_sim_ncr_rad_all, read_zpa_and_hst

def get_fig_axes_caxes(nrow=7, ncol=2, ncaxes=3):
    fig = plt.figure(figsize=(16, 2.5*nrow),
                     constrained_layout=False)
    gs = fig.add_gridspec(nrow, ncol, width_ratios=(50, 1),
                          height_ratios=np.repeat(1, nrow),
                          wspace=0.05, hspace=0.12)
                      # left=0.1, right=0.9, bottom=0.1, top=0.9,
    axes = [fig.add_subplot(gs[i, 0]) for i in range(nrow)]
    caxes = [fig.add_subplot(gs[i+1, 1]) for i in range(ncaxes)]

    return fig, axes, caxes

def plot_hist(model_set, model,
              sa=None, df=None, phase_set_name='warm_eq_LyC_ma',
              force_override=False, savefig=True):
    if sa is None or df is None:
        sa, df = load_sim_ncr_rad_all(model_set=model_set, verbose=False)

    if phase_set_name is None:
        phase_set_name = list(s.phase_set.keys())[-1]

    s, zpa, h = read_zpa_and_hst(sa, df, model,
                                 phase_set_name=phase_set_name,
                                 force_override=force_override)
    xlim = df.loc[model]['tMyr_range']

    u = s.u
    zpa = zpa.rename(dict(time='time_code'))
    zpa = zpa.assign_coords(tMyr=zpa.time_code*u.Myr)
    zpa = zpa.swap_dims(dict(time_code='tMyr'))

    zp_eq = zpa.sel(phase='w1_eq_noLyC')
    zp_pi = zpa.sel(phase='w1_eq_LyC_pi') + zpa.sel(phase='w1_eq_LyC')
    zp_neq = zpa.sel(phase='w1_geq_noLyC') +\
        zpa.sel(phase='w1_geq_LyC') + zpa.sel(phase='w1_geq_LyC_pi')

    fig, axes, caxes = get_fig_axes_caxes()

    # LyC escape fraction
    l, = axes[0].semilogy(h['tMyr'], h['fesc_LyC'], c='k')
    plt.setp(axes[0], xlabel=r'${\rm time}\;[{\rm Myr}]$', ylim=(1e-3, 1e-1))
    axes[0].set_ylabel(r'$f_{\rm esc,LyC}$', color=l.get_color())
    axes[0].tick_params(axis='y', labelcolor=l.get_color())
    # Ionizing photon rate per unit area
    axt = axes[0].twinx()
    l, = axt.semilogy(h['tMyr'], h['Phi_LyC']/1e49, c='orange')
    axt.set_ylim(3e0, 3e2)
    axt.set_ylabel(r'$\Phi_{\rm LyC,49}$', color=l.get_color())
    axt.tick_params(axis='y', labelcolor=l.get_color())

    def plot_spacetime_diagram(ax, cax, zp, cmap, label):
        for ax_, cax_, zp_, cmap_, label_ in zip(ax, cax, zp, cmap, label):
            pc = ax_.pcolormesh(zp_['tMyr'], zp_['z']*u.kpc, zp_['frac'].T,
                                cmap=cmap_, norm=mpl.colors.Normalize(0, 1.0))
            plt.colorbar(mappable=pc, cax=cax_, label=label_)

    # Spacetime diagrams
    import cmasher as cmr
    import cmocean.cm as cmo
    zp = [zp_eq, zp_pi, zp_neq]
    cmap = [cmo.tempo,
            cmr.get_sub_cmap(cmo.balance, 0.0, 0.5).reversed(),
            cmr.get_sub_cmap(cmo.balance, 0.5, 1.0)]
    label = [r'$f_{V,{\rm w,eq}}$',
             r'$f_{V,{\rm w,pi}}$',
             r'$f_{V,{\rm w,neq}}$']
    plot_spacetime_diagram(axes[1:4], caxes, zp, cmap, label)
    plt.setp(axes[1:4], ylim=(-1.3, 1.3), ylabel=r'$z\;[{\rm kpc}]$')
    for ax in axes[1:4]:
        ax.set_yticks([-1.0,-0.5,0.0,0.5,1.0])

    slc_lo = slice(-100, 100)
    slc_hi = slice(400, 600)
    #slc_hi1 = slice(300, 1000)
    #slc_hi2 = slice(-1000, -300)

    zp_eq_lo = zp_eq.sel(z=slc_lo).mean(dim='z')
    zp_eq_hi = zp_eq.sel(z=slc_hi).mean(dim='z')

    zp_pi_lo = zp_pi.sel(z=slc_lo).mean(dim='z')
    zp_pi_hi = zp_pi.sel(z=slc_hi).mean(dim='z')
    # zp_pi_hi = 0.5*(zp_pi.sel(z=slc_hi1).mean(dim='z') +\
    #                 zp_pi.sel(z=slc_hi2).mean(dim='z'))

    zp_neq_lo = zp_neq.sel(z=slc_lo).mean(dim='z')
    zp_neq_hi = zp_neq.sel(z=slc_hi).mean(dim='z')
    # zp_neq_hi = 0.5*(zp_neq.sel(z=slc_hi1).mean(dim='z') +\
    #                  zp_neq.sel(z=slc_hi2).mean(dim='z'))

    def plot_hist_one(ax, zps, xf, yf1, yf2=None, plt_kwargs=None):
        for zp, kwargs in zip(zps, plt_kwargs):
            x = zp[xf]
            if yf2 is None:
                y = zp[yf1]
            else:
                y = zp[yf1]/zp[yf2]

            ax.plot(x, y, **kwargs)

    zps = [zp_eq_lo, zp_eq_hi,
           zp_pi_lo, zp_pi_hi,
           zp_neq_lo, zp_neq_hi]
    c = 0.5
    plt_kwargs = [dict(c=cmap[0](c)), dict(c=cmap[0](c), ls='--', lw=1.5),
                  dict(c=cmap[1](c)), dict(c=cmap[1](c), ls='--', lw=1.5),
                  dict(c=cmap[2](c)), dict(c=cmap[2](c), ls='--', lw=1.5)]

    # zps = [zp_pi_lo, zp_pi_hi, zp_neq_lo, zp_neq_hi]
    # plt_kwargs = [dict(c='C0'), dict(c='C0', ls='--'),
    #               dict(c='C1'), dict(c='C1', ls='--')]

    plot_hist_one(axes[4], zps, 'tMyr', 'frac_w_nesq', plt_kwargs=plt_kwargs)
    plt.setp(axes[4], ylim=(0, 1.1), ylabel=r'$f_{n_e^2}$')

    plot_hist_one(axes[5], zps, 'tMyr', 'xHII_w_nesq', 'frac_w_nesq', plt_kwargs)
    plt.setp(axes[5], yscale='linear', ylim=(0, 1.1),
             ylabel=r'$\langle x_{\rm H^+} \rangle_{n_{\rm e}^2} / f_{n_{\rm e}^2}$')

    plot_hist_one(axes[6], zps, 'tMyr', 'T_w_nesq', 'frac_w_nesq', plt_kwargs)

    if s.par['problem']['Z_gas'] == 1.0:
        ylim = (5e3, 1.2e4)
    else:
        ylim = (5e3, 1.5e4)

    plt.setp(axes[6], yscale='linear', ylim=ylim,
             ylabel=r'$\langle T \rangle_{n_{\rm e}^2} / f_{n_{\rm e}^2}$')

    plt.setp(axes, xlim=xlim)
    plt.setp(axes[-1], xlabel=r'${\rm time}\;[{\rm Myr}]$')
    for ax in axes[:-1]:
        ax.set_xticklabels([])
    for ax in axes:
        ax.tick_params(direction='out')

    if savefig:
        fig.savefig('/tigress/jk11/figures/NCR-RADIATION/hst-{0:s}-{1:s}.png'.\
                    format(model, phase_set_name), dpi=200, bbox_inches='tight')

    zp_dict = dict(zpa=zpa, zp_eq=zp_eq, zp_neq=zp_neq, zp_pi=zp_pi)

    return fig, s, zp_dict, h
