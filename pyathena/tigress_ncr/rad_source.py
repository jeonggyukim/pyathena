import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def get_source_info(s, nums, force_override=False):
    """Function to calculate radiation source statistics
    """

    spa = s.read_starpar_all(nums=nums, savdir=s.savdir,
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
    time_Myr = spa['time'].values*s.u.Myr
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
        r = get_source_info(s, nums, force_override=force_override)

    plt.fill_between(r['spa']['time'], r['spa']['z_min'], r['spa']['z_max'],
                     alpha=0.2, color='grey', label=r'$z_{\rm min/max}$')
    # LyC sources
    plt.fill_between(r['spa']['time'],
                     r['spa']['z_mean_Qi'] - 0.5*r['spa']['stdz_Qi'],
                     r['spa']['z_mean_Qi'] + 0.5*r['spa']['stdz_Qi'],
                     alpha=0.3, color='red', label=r'EUV')
    # FUV sources
    plt.fill_between(r['spa']['time'],
                     r['spa']['z_mean_LFUV'] - 0.5*r['spa']['stdz_LFUV'],
                     r['spa']['z_mean_LFUV'] + 0.5*r['spa']['stdz_LFUV'],
                     alpha=0.3, color='blue', label=r'FUV')

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
        r = get_source_info(s, nums, force_override=force_override)

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

def plt_luminosity_hist(s, nums, axes=None, r=None,
                        c='k', legend=False, force_override=False):
    """Plot histograms of Qi and L_FUV of individual sources (from all snapshots)

    axes : two matplotlib axes
    r : return value of get_source_info (dict)
    """
    if axes is None:
        fig, axes = plt.subplots(1,2,figsize=(12,6), constrained_layout=True)

    if r is None:
        r = get_source_info(s, nums, force_override=force_override)

    axes[0].hist(r['Qi'], bins=np.logspace(44, 51.5, 40),
                 histtype='step', color=c)
    axes[0].hist(r['Qi_90'], bins=np.logspace(44, 51.5, 40),
                 histtype='step', color=c, ls='--')
    axes[1].hist(r['L_FUV'], bins=np.logspace(3.5, 7.5, 40),
                 histtype='step', color=c)
    axes[1].hist(r['L_FUV_90'], bins=np.logspace(3.5, 7.5, 40),
                 histtype='step', color=c, ls='--')

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
