#!/usr/bin/env python

import sys, os
sys.path.insert(0, '/home/jk11/athena-tigress/python')
import pyathena as pa
import pyathena.tigradpy as tp

import matplotlib.pyplot as plt

class PltHst:
    
    def plt_hst(self, savname=None, force_override=False):
        """Function to draw time evolution of Sigma_SFR, escape fraction, etc.
        """
        hst = self.read_hst(force_override=force_override)
        fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True,
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
        # # H absorption
        # plt.plot(hst.time, hst.Qiphot/hst.Qi, 'b-')
        # plt.plot(hst.time, hst.Qiphot.cumsum()/hst.Qi.cumsum(), 'b--')
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

        for ax in axes:
            ax.set_xlim(hst.time.iloc[0], hst.time.iloc[-1])

        plt.sca(axes[-1])
        plt.xlabel('time [Myr]')

        if savname is None:
            savname = os.path.join(self.savdir, 'hst',
                                   self.problem_id + '_hst.png')
            
        plt.savefig(savname, dpi=200)
        self.logger.info('History plot saved to {:s}'.format(savname))
