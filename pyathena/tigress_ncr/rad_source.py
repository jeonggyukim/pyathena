import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

class RadiationSource:

    def get_source_info(self, nums, force_override=False):
        """Function to calculate radiation source statistics
        """

        spa = self.read_starpar_all(nums=nums, savdir=self.savdir,
                                    force_override=force_override)

        # Instantaneous Qi,sp and L_FUV,sp in all snapshots
        Qi = []
        L_FUV = []
        # Total number of radiation sources
        ntot_Qi = []
        ntot_FUV = []
        # Qi,sp and L_FUV,sp that account for >90% of the total
        Qi_90 = []
        L_FUV_90 = []
        # Number of such sources
        n90_Qi = []
        n90_FUV = []
        for i, sp in spa['sp'].items():
            # print(i, end=' ')
            Qi.append(list(sp['Qi'].values))
            L_FUV.append(sp['L_FUV'].values)

            Qi_srt = sp['Qi'].sort_values(ascending=False)
            L_FUV_srt = sp['L_FUV'].sort_values(ascending=False)

            idx_Qi = Qi_srt.cumsum() < 0.9*Qi_srt.sum()
            idx_FUV = L_FUV_srt.cumsum() < 0.9*L_FUV_srt.sum()

            n90_Qi.append(idx_Qi.sum())
            n90_FUV.append(idx_FUV.sum())
            Qi_90.append(Qi_srt[idx_Qi])
            L_FUV_90.append(L_FUV_srt[idx_FUV])

            ntot_Qi.append(len(sp['Qi'].values))
            ntot_FUV.append(len(sp['L_FUV'].values))

        # Convert list of list to 1d array
        import itertools
        Qi = np.array(list(itertools.chain.from_iterable(Qi)))
        L_FUV = np.array(list(itertools.chain.from_iterable(L_FUV)))
        Qi_90 = np.array(list(itertools.chain.from_iterable(Qi_90)))
        L_FUV_90 = np.array(list(itertools.chain.from_iterable(L_FUV_90)))

        time_code = spa['time'].values
        time_Myr = spa['time'].values*self.u.Myr
        r = dict(spa=spa, time_code=time_code, time_Myr=time_Myr,
                 Qi=Qi, L_FUV=L_FUV, ntot_Qi=ntot_Qi, ntot_FUV=ntot_FUV,
                 n90_Qi=n90_Qi, n90_FUV=n90_FUV, Qi_90=Qi_90, L_FUV_90=L_FUV_90)

        return r

def plt_hst_source_z(s, nums, ax=None, r=None,
                     title=True, legend=True, force_override=False):
    """Plot time evolution of mean vertical position of radiation sources and stndard
    deviation

    """
    if ax is not None:
        plt.sca(ax)

    if r is None:
        r = s.get_source_info(nums, force_override=force_override)

    u = s.u
    plt.fill_between(r['spa']['time']*u.Myr, r['spa']['z_min'], r['spa']['z_max'],
                     alpha=0.2, color='grey', label=r'$z_{\rm min/max}$')
    # LyC sources
    plt.fill_between(r['spa']['time']*u.Myr,
                     r['spa']['z_mean_Qi'] - 0.5*r['spa']['stdz_Qi'],
                     r['spa']['z_mean_Qi'] + 0.5*r['spa']['stdz_Qi'],
                     alpha=0.3, color='red', label=r'$z_{\rm sp,mean}\pm \sigma$ (LyC)')
    # FUV sources
    plt.fill_between(r['spa']['time']*u.Myr,
                     r['spa']['z_mean_LFUV'] - 0.5*r['spa']['stdz_LFUV'],
                     r['spa']['z_mean_LFUV'] + 0.5*r['spa']['stdz_LFUV'],
                     alpha=0.3, color='blue', label=r'$z_{\rm sp,mean}\pm \sigma$ (FUV)')

    plt.ylabel(r'$z\;[{\rm pc}]$')

    if title:
        plt.title('{0:s}'.format(s.basename))

    if legend:
        plt.legend()

    return r

def plt_luminosity_distribution(s, nums, ax=None, r=None, title=True, label=True,
                                legend=True, force_override=False):
    if ax is not None:
        plt.sca(ax)

    if r is None:
        r = s.get_source_info(nums, force_override=force_override)

    l, = plt.plot(r['time_Myr'], r['ntot_Qi'], c='C0',
                  label=r'$n_{\rm{src,tot}}$')
    plt.plot(r['time_Myr'], r['n90_Qi'], c=l.get_color(), ls='--',
             label=r'$n_{\rm src,90\%,LyC}$')
    plt.plot(r['time_Myr'], r['n90_FUV'], c=l.get_color(), ls=':',
             label=r'$n_{\rm src,90\%,FUV}$')

    plt.ylim(bottom=0.5)
    plt.yscale('log')
    if legend:
        plt.legend()
    if title:
        plt.title('{0:s}'.format(s.basename))
    if label:
        plt.xlabel(r'${\rm time}\;[{\rm Myr}]$')
        plt.ylabel(r'# of sources (instantaneous)')

    return r

def plt_luminosity_hist(s, nums, axes=None, r=None, lw=2,
                        c='k', legend=False, force_override=False):
    """Plot histograms of Qi and L_FUV of individual sources (from all snapshots)

    axes : two matplotlib axes
    r : return value of get_source_info (dict)
    """
    if axes is None:
        fig, axes = plt.subplots(1,2,figsize=(12,6), constrained_layout=True)

    if r is None:
        r = s.get_source_info(nums, force_override=force_override)

    axes[0].hist(r['Qi'], bins=np.logspace(44, 51.5, 40),
                 histtype='step', color=c, lw=lw)
    axes[0].hist(r['Qi_90'], bins=np.logspace(44, 51.5, 40),
                 histtype='step', color=c, ls='--', lw=lw)
    axes[1].hist(r['L_FUV'], bins=np.logspace(3.5, 7.5, 40),
                 histtype='step', color=c, lw=lw)
    axes[1].hist(r['L_FUV_90'], bins=np.logspace(3.5, 7.5, 40),
                 histtype='step', color=c, ls='--', lw=lw)

    plt.setp(axes, xscale='log', yscale='log')
    plt.setp(axes[0], xlim=(1e44,5e51), ylim=(0.5,5e3),
             xlabel=r'$Q_{\rm i,sp}\;[{\rm s}^{-1}]$',
             ylabel=r'# of sources (all snapshots)')
    plt.setp(axes[1], xlim=(1e3,1e8), ylim=(0.5,5e3),
             xlabel=r'$L_{\rm FUV,sp}\;[L_{\odot}]$',
             ylabel=r'# of sources (all snapshots)')

    if legend:
        handles = [mpl.lines.Line2D([0],[0],c='k'),
                   mpl.lines.Line2D([0],[0],c='k',ls='--')]
        labels = ['all', 'bright srcs (90% of the total)']
        axes[0].legend(handles, labels, loc=1)

    return r

def print_source_summary(s, nums):
    #from decimal import getcontext
    #getcontext().prec = 3

    print('** Source statistics **')
    print('Model: {0:s}'.format(s.basename))
    print('')

    r = s.get_source_info(nums, force_override=False)
    spa = r['spa']
    print('- Mean Qi and L_FUV of all all sources [s^-1, Lsun]: ',
          '{0:.4g} , {1:.4g}'.format(
              r['Qi'].mean(), r['L_FUV'].mean()))
    print('- Qi- and L_FUV-weighted mean Qi and L_FUV of all all sources [s^-1, Lsun]: ',
          '{0:.4g} , {1:.4g}'.format(
              (r['Qi']*r['Qi']).sum()/r['Qi'].sum(),
              (r['L_FUV']*r['L_FUV']).sum()/r['L_FUV'].sum()))
    print('- Mean Qi and L_FUV accounting for 90% of the total instantaneous luminosity [s^-1, Lsun]: ',
          '{0:.4g} , {1:.4g}'.format(
              r['Qi_90'].mean(), r['L_FUV_90'].mean()))
    print('- Mean of instantaneous z_mean_Qi and z_mean_LFUV [pc]: ',
          '{0:.4g} , {1:.4g}'.format(
              np.mean(spa['z_mean_Qi'].values), np.mean(spa['z_mean_LFUV'].values)))
    print('- stddev of z_mean_Qi and z_mean_LFUV [pc]: ',
          '{0:.4g} , {1:.4g}'.format(
              np.std(spa['z_mean_Qi'].values), np.std(spa['z_mean_LFUV'].values)))
    print('- Qi-weighted time-average of Qi-weighted rms distance from z_mean_Qi [pc]: ',
          '{0:.4g}'.format(
              (spa['Qi_tot']*spa['stdz_Qi']).sum()/spa['Qi_tot'].sum()))
    print('- FUV-weighted time-average of L_FUV-weighted rms distance from z_mean_FUV [pc]: ',
          '{0:.4g}'.format(
              (spa['L_FUV_tot']*spa['stdz_LFUV']).sum()/spa['L_FUV_tot'].sum()))
    print('- l-star,90%,Qi: ', np.median(spa['lsrc_90_Qi']))
    print('- l-star,90%,LFUV: ', np.median(spa['lsrc_90_LFUV']))
    print('')

    return r

def plot_source_info(s, nums, savefig=True):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    axes = axes.flatten()
    # print(mdl, end=' ')
    r = s.get_source_info(nums)
    plt_hst_source_z(s, nums, ax=axes[0], title=False);
    plt_luminosity_distribution(s, nums, ax=axes[1], legend=True, title=False);
    # plt.setp(ax, ylim=(-200,200), xlabel='time [code]', ylabel=r'$z\;[{\rm pc}]$');
    plt_luminosity_hist(s, nums, axes[2:], c='k', legend=True);
    plt.suptitle(s.basename)
    if savefig:
        plt.savefig('/tigress/jk11/figures/NCR-RADIATION/src_info_{0:s}.png'.\
                    format(s.basename), dpi=200)

    return fig
