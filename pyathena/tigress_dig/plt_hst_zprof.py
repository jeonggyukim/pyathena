#!/usr/bin/env python

import os
import os.path as osp

import matplotlib as mpl
import matplotlib.pyplot as plt

class PltHstZprof:

    def plt_hst(self, savname=None, force_override=False):
        """Function to draw time evolution of Sigma_SFR, escape fraction, etc.
        """

        hst = self.read_hst(force_override=force_override)
        fig, axes = plt.subplots(5, 1, figsize=(18, 12), sharex=True,
                                 gridspec_kw=dict(hspace=0.1))

        # SFR10, Qi/Area
        plt.sca(axes[0])
        ax = axes[0]
        lunit = (self.u.length.cgs.value)
        LxLy = self.ds.domain['Lx'][0:2].prod()*lunit**2
        l1, = plt.plot(hst.time, hst.Qi/LxLy, '-')
        c = l1.get_color()
        plt.yscale('log')
        ax.spines['right'].set_visible(False)
        ax.set_ylabel(r'$Q_{\rm i}/Area [\#\,{\rm s}^{-1}\,{\rm cm}^{-2}]$')
        ax.yaxis.label.set_color(l1.get_color())
        ax.spines['left'].set_color(c)
        ax.tick_params(axis='y', which='both', colors=c)

        axt = plt.twinx()
        plt.sca(axt)
        l2, = axt.plot(hst.time, hst.sfr10, c='C1')
        c = l2.get_color()
        axt.spines['left'].set_visible(False)
        axt.set_yscale('log')
        axt.set_ylabel(r'$\Sigma_{\rm SFR,10\,{\rm Myr}}$' + \
                       r'$\;[M_{\odot}\,{\rm kpc}^{-2}\,{\rm yr}^{-1}]$')
        axt.yaxis.label.set_color(c)
        axt.spines['right'].set_color(c)
        axt.tick_params(axis='y', which='both', colors=c)

        # Ionized gas mass fraction
        plt.sca(axes[1])
        l3, = plt.plot(hst.time, hst.mf_ion, '-', c='C2',
                      label=r'$M_{\rm ion}/M_{\rm gas}$')
        plt.plot(hst.time, hst.mf_ion_coll, '--', c=l3.get_color(),
                 label=r'$M_{\rm ion,coll}/M_{\rm gas}$')
        plt.yscale('log')
        plt.ylabel('ionized gas\n' + 'mass fraction')
        plt.legend(loc=1, fontsize='x-large', framealpha=0.8)

        # Radiation escape fraction
        plt.sca(axes[2])
        # Ionizing escape fraction
        l1, = plt.plot(hst.time, hst.fesc0_est, 'm-')
        plt.plot(hst.time, hst.fesc0_cum_est, 'm--')
        # Non-ionizing escape fraction
        l2, = plt.plot(hst.time, hst.fesc1_est, 'k-')
        plt.plot(hst.time, hst.fesc1_cum_est, 'k--')

        # H absorption
        #plt.plot(hst.time, hst.Qiphot/hst.Qi, 'b-')
        #plt.plot(hst.time, hst.Qiphot.cumsum()/hst.Qi.cumsum(), 'b--')
        # dust absorption
        #plt.plot(hst.time, hst.Qidust/hst.Qi, 'g-')
        #plt.plot(hst.time, hst.Qidust.cumsum()/hst.Qi.cumsum(), 'g--')

        plt.ylabel('radiation\n' + 'escape fraction')
        plt.legend([l1, l2],
                   [r'$f_{\rm esc,i}$', r'$f_{\rm esc,n}$'], loc=1,
                   fontsize='x-large', framealpha=0.8)

        c = [(param, value) for param, value in plt.rcParams.items() if 'color' in param]

        plt.yscale('log')
        plt.ylim(1e-3, 1e0)

        # Ionizing photon H absorption, dust absorption, and escape fractions
        plt.sca(axes[3])
        alpha=0.5
        p1 = plt.fill_between(hst.time, 0,
                              hst.fesc0_est,
                              color='b', alpha=alpha)
        p2 = plt.fill_between(hst.time,
                              hst.fesc0_est,
                              hst.fesc0_est + (hst.Qiphot/hst.Qi),
                              color='g', alpha=alpha)
        p3 = plt.fill_between(hst.time,
                              hst.fesc0_est + (hst.Qiphot/hst.Qi),
                              hst.fesc0_est + (hst.Qiphot/hst.Qi) + (hst.Qidust/hst.Qi),
                              color='grey', alpha=alpha)
        plt.ylabel('fraction')
        plt.ylim(0, 1)
        plt.legend([p1, p2, p3],
                   ['Escape','H absorption','Dust absorption'],
                   loc=1)

        # Scale heights
        plt.sca(axes[4])
        plt.plot(hst.time, hst.H_wnesq, label=r'$H_{n_e^2}$')
        plt.plot(hst.time, hst.H_wi, label=r'$H_{\rm w,i}$')
        plt.plot(hst.time, hst.H_w, label=r'$H_{\rm w}$')
        plt.plot(hst.time, hst.H_c, label=r'$H_{\rm c}$')
        plt.yscale('log')
        plt.legend(loc=1, fontsize='large')

        # alpha=0.5
        # plt.ylabel('fraction')
        # plt.ylim(0, 1)
        # plt.legend([p1, p2, p3],
        #            ['Escape','H absorption','Dust absorption'],
        #            loc=1)


        for ax in axes:
            ax.set_xlim(hst.time.iloc[0], hst.time.iloc[-1])

        plt.sca(axes[-1])
        plt.xlabel('time [Myr]')

        plt.suptitle(self.basename)
        plt.subplots_adjust(top=0.95)

        if savname is None:
            savname = osp.join(self.savdir, 'hst',
                               self.problem_id + '_hst.png')

        plt.savefig(savname, dpi=200)

        self.logger.info('History plot saved to {:s}'.format(savname))

        return plt.gcf()

    def plt_zprof_median(self, savname=None, force_override=False):
        """Function to draw median z-profiles of nH, ne, xi, etc.
        """

        zp = self.read_zprof(['w', 'h'], force_override=force_override)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        ylog = dict(d=True, ne=True, nebar=True, nesq=True, xi=False, A=False)
        ylim = dict(d=(1e-4, 2e1),
                    ne=(1e-4, 2e1),
                    nesq=(1e-5, 1e3),
                    xi=(0.0, 1.0),
                    nebar=(1e-4, 2e1),
                    A=(0.0, 1.0)
                    )

        ylabel = dict(d=r'$\langle n_{\rm H}\rangle\;[{\rm cm^{-3}}]$',
                      ne=r'$\langle n_{\rm e}\rangle\;[{\rm cm^{-3}}]$',
                      xi=r'$\langle x_{\rm i}\rangle$',
                      nebar=r'$\langle n_{\rm e}\rangle/\langle x_{\rm i}\rangle\;[{\rm cm^{-3}}]$'
                     )

        phase = ['w', 'w', 'w', 'w']
        var = ['d', 'ne', 'xi', 'nebar']
        xlim = (zp['w'].z_kpc.min(),zp['w'].z_kpc.max())

        axes = axes.flatten()
        for ax, ph, v in zip(axes, phase, var):
            plt_zprof_var(ax, zp[ph], v,
                          xlim, ylim[v], ylog[v], ylabel[v], alpha=None)

        # Plot hot phase median
        alpha = 0.5
        lw= 2.0
        ph = 'h'
        plt.sca(axes[0])
        l, = plt.plot(zp[ph].z_kpc, zp[ph]['d'].quantile(0.5, dim='time'),
                 alpha=alpha, c='tab:red', lw=lw)
        plt.sca(axes[1])
        l, = plt.plot(zp[ph].z_kpc, zp[ph]['ne'].quantile(0.5, dim='time'),
                 alpha=alpha, c='tab:red', lw=lw)
        plt.sca(axes[2])
        plt.plot(zp[ph].z_kpc, zp[ph]['xi'].quantile(0.5, dim='time'),
                 alpha=alpha, c='tab:red', lw=lw)
        plt.sca(axes[3])
        plt.plot(zp[ph].z_kpc, zp[ph]['nebar'].quantile(0.5, dim='time'),
                 alpha=alpha, c='tab:red', lw=lw)
        # Plot warm neutral
        ph = 'w'
        plt.sca(axes[2])
        plt.plot(zp[ph].z_kpc, zp[ph]['A'].quantile(0.5, dim='time'),
                 alpha=alpha, c='tab:blue', lw=lw, ls='--')

        plt.sca(axes[0])
        plt.legend([axes[0].get_lines()[-2], l], ['warm','hot'], loc=2)

        plt.suptitle(self.basename, fontsize='large')
        plt.subplots_adjust(wspace=0.5)
        #plt.tight_layout()

        if savname is None:
            savname = osp.join(self.savdir, 'zprof',
                               self.problem_id + '_zprof.png')

        plt.savefig(savname, dpi=200)

        self.logger.info('zprof plot saved to {:s}'.format(savname))

        return plt.gcf()

    def plt_zprof_frac(self, savname=None, force_override=False):
        """Function to draw time-averaged z-profiles of volume and mass fractions
        """

        zpw = self.read_zprof('w')
        zpa = self.read_zprof('whole') # all
        zp2p = self.read_zprof('2p') # 2-phase (c + u + w)
        zph = self.read_zprof('h') # 2-phase (c + u + w)
        zpc = self.read_zprof('c') # cold
        zpu = self.read_zprof('u') # unstable
        hst = self.read_hst()

        # Volume fractions of hot, warm, and warm ionized
        vfh = (zph['A']).mean(dim='time')
        vfw = (zpw['A']).mean(dim='time')
        vfwi = (zpw['xi']).mean(dim='time')
        # Volume fraction of WIM relative to the volume fraction of warm
        vfwi_vfw = (zpw['xi']/zpw['A']).mean(dim='time')
        vfu = (zpu['A']).mean(dim='time')
        vfc = (zpc['A']).mean(dim='time')

        # Mass fractions of hot, warm, warm ionized, and cold
        mfh = (zph['d']/zpa['d']).mean(dim='time')
        mfw = (zpw['d']/zpa['d']).mean(dim='time')
        mfwi = (zpw['ne']/zpa['d']).mean(dim='time')
        mfwi_mfw = (zpw['ne']/zpw['d']).mean(dim='time')
        mfc = (zpc['d']/zpa['d']).mean(dim='time')
        mfu = (zpu['d']/zpa['d']).mean(dim='time')

        # x axis
        x = zpw.z_kpc

        mpl.rcParams['font.size'] = 16
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        # Volume fractions
        plt.sca(axes[0])
        plt.plot(x, vfh, label=r'$f_{\rm V,h}$', c='orchid')
        plt.plot(x, vfw, label=r'$f_{\rm V,w}$', c='tab:orange')
        plt.plot(x, vfwi, label=r'$f_{\rm V,wi}$', c='tab:green')
        #plt.plot(x, vfwi_vfw, label=r'$f_{\rm V,wi}/f_{\rm V,w}$', c='tab:red')
        #plt.plot(x, vfc, label=r'$f_{\rm V,c}$', c='tab:blue')
        #plt.plot(x, vfu, label=r'$f_{\rm V,u}$', c='salmon')
        plt.legend(loc=2)
        plt.xlabel(r'$z\;[{\rm kpc}]$')
        plt.ylabel('volume fraction')
        plt.locator_params(nbins=10)
        plt.grid()

        # Mass fractions
        plt.sca(axes[1])
        plt.plot(x, mfh, label=r'$f_{\rm M,h}$', c='orchid')
        plt.plot(x, mfw, label=r'$f_{\rm M,w}$', c='tab:orange')
        plt.plot(x, mfwi, label=r'$f_{\rm M,wi}$', c='tab:green')
        #plt.plot(x, mfwi_mfw, label=r'$f_{\rm M,wi}/f_{\rm M,w}$', c='tab:red')
        #plt.plot(x, mfc, label=r'$f_{\rm M,c}$', c='tab:blue')
        #plt.plot(x, mfu, label=r'$f_{\rm M,u}$', c='salmon')
        plt.legend(loc=2)
        plt.xlabel(r'$z\;[{\rm kpc}]$')
        plt.ylabel('mass fraction')
        plt.locator_params(nbins=10)
        plt.grid()

        plt.suptitle(self.basename, fontsize='large')
        plt.subplots_adjust(wspace=0.4)
        # #plt.tight_layout()

        if savname is None:
            savname = osp.join(self.savdir, 'zprof',
                               self.problem_id + '_zprof_frac.png')

        plt.savefig(savname, dpi=200)

        self.logger.info('zprof frac plot saved to {:s}'.format(savname))

        return plt.gcf()

def plt_zprof_var(ax, zp, v, xlim, ylim, ylog, ylabel, alpha=0.02):
    """Function to draw median, 10/25/75/90 percentile ranges."""

    if alpha is None:
        alpha = 0.02*(500.0/zp[v].shape[1])**0.5

    plt.sca(ax)
    plt.plot(zp.z_kpc, zp[v], alpha=alpha, c='grey')
    plt.plot(zp.z_kpc, zp[v].quantile(0.5, dim='time'), c='tab:blue')
    plt.fill_between(zp.z_kpc,
                     zp[v].quantile(0.25, dim='time'),
                     zp[v].quantile(0.75, dim='time'),
                     alpha=0.5, color='tab:blue')
    plt.fill_between(zp.z_kpc,
                     zp[v].quantile(0.10, dim='time'),
                     zp[v].quantile(0.90, dim='time'),
                     alpha=0.20, color='tab:orange')

    if ylog:
        plt.yscale('log')
    else:
        plt.yscale('linear')

    plt.xlabel('z [kpc]')
    plt.ylabel(ylabel)
    plt.xlim(xlim)
    plt.ylim(ylim)
    #plt.suptitle(v)
