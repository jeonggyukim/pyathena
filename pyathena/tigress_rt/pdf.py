# pdf.py

import os
import os.path as osp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as au
import astropy.constants as ac
from matplotlib.colors import Normalize, LogNorm

from ..classic.utils import texteffect
from ..plt_tools.cmap_custom import get_my_cmap
from ..util.scp_to_pc import scp_to_pc
from ..load_sim import LoadSim
from ..plt_tools.plt_starpar import scatter_sp

class PDF:
    
    bins=dict(nH=np.logspace(-5,3,81),
              nHI=np.logspace(-2,5,71),
              nH2=np.logspace(-2,5,71),
              nHII=np.logspace(-5,3,101),
              T=np.logspace(1,8,141),
              pok=np.logspace(0,7,71),
              chi_PE=np.logspace(-3,4,71),
              chi_FUV=np.logspace(-3,4,71),
              Lambda_cool=np.logspace(-30,-20,101),
              xi_CR=np.logspace(-17,-15,41)
    )
    
    @LoadSim.Decorators.check_pickle
    def read_pdf2d(self, num,
                   bin_fields=None, bins=None, prefix='pdf2d',
                   savdir=None, force_override=False):
        if self.par['configure']['radps'] == 'ON':
            bin_fields_def = [['nH', 'pok'], ['nH', 'T'], ['nH','chi_FUV'],
                              ['T','Lambda_cool'], ['nH','xi_CR']]
        else:
            bin_fields_def = [['nH', 'pok'], ['nH', 'T']]
        if bin_fields is None:
            bin_fields = bin_fields_def

        ds = self.load_vtk(num=num)
        res = dict()
        
        for bf in bin_fields:
            k = '-'.join(bf)
            res[k] = dict()
            dd = ds.get_field(bf)
            xdat = dd[bf[0]].data.flatten()
            ydat = dd[bf[1]].data.flatten()
            # Volume weighted hist
            weights = None
            H, xe, ye = np.histogram2d(xdat, ydat, (self.bins[bf[0]], self.bins[bf[1]]),
                                       weights=weights)
            res[k]['H'] = H
            res[k]['xe'] = xe
            res[k]['ye'] = ye
            
            # Density weighted hist
            weights = (ds.get_field('nH'))['nH'].data.flatten()
            Hw, xe, ye = np.histogram2d(xdat, ydat, (self.bins[bf[0]], self.bins[bf[1]]),
                                        weights=weights)
            res[k]['Hw'] = Hw

        return res

    def plt_pdf2d(self, ax, dat, bf='nH-pok',
                  cmap='cubehelix_r',
                  norm=mpl.colors.LogNorm(1e-6,2e-2),
                  kwargs=dict(alpha=1.0, edgecolor='face', linewidth=0, rasterized=True),
                  weighted=True,
                  xscale='log', yscale='log'):
        
        if weighted:
            hist = 'Hw'
        else:
            hist = 'H'

        ax.pcolormesh(dat[bf]['xe'], dat[bf]['ye'], dat[bf][hist].T/dat[bf][hist].sum(),
                      norm=norm, cmap=cmap, **kwargs)

        kx, ky = bf.split('-')
        ax.set(xscale=xscale, yscale=yscale,
               xlabel=self.dfi[kx]['label'], ylabel=self.dfi[ky]['label'])
            
    def plt_pdf2d_all(self, num, suptitle=None, savdir=None, force_override=False, savefig=True):

        if savdir is None:
            savdir = self.savdir

        s = self
        ds = s.load_vtk(num)
        pdf = s.read_pdf2d(num, force_override=force_override)
        prj = s.read_prj(num, force_override=force_override)
        slc = s.read_slc(num, force_override=force_override)
        hst = s.read_hst(savdir=savdir, force_override=force_override)
        sp = s.load_starpar_vtk(num)
        zpa = s.read_zprof(['whole','2p','h'], savdir=savdir, force_override=force_override)

        fig, axes = plt.subplots(3,4,figsize=(20,15), constrained_layout=True)

        # gs = axes[0, -1].get_gridspec()
        # for ax in axes[0:2, -1]:
        #     ax.remove()
        # ax = fig.add_subplot(gs[0:2, -1])

        s.plt_pdf2d(axes[0,0], pdf, 'nH-pok', weighted=False)
        s.plt_pdf2d(axes[1,0], pdf, 'nH-pok', weighted=True)
        s.plt_pdf2d(axes[0,1], pdf, 'nH-chi_FUV', weighted=False)
        s.plt_pdf2d(axes[1,1], pdf, 'nH-chi_FUV', weighted=True)
        s.plt_pdf2d(axes[0,2], pdf, 'T-Lambda_cool', weighted=False)
        s.plt_pdf2d(axes[1,2], pdf, 'T-Lambda_cool', weighted=True)
        s.plt_pdf2d(axes[0,3], pdf, 'nH-xi_CR', weighted=False)
        s.plt_pdf2d(axes[1,3], pdf, 'nH-xi_CR', weighted=True)

        ax = axes[2,0]
        s.plt_proj(ax, prj, 'z', 'Sigma_gas')
        scatter_sp(sp, ax, 'z', kind='prj', kpc=False, norm_factor=5.0, agemax=20.0)
        ax.axes.xaxis.set_visible(False) ; ax.axes.yaxis.set_visible(False)
        ax.set(xlim=(ds.domain['le'][0], ds.domain['re'][0]),
               ylim=(ds.domain['le'][1], ds.domain['re'][1]))

        ax = axes[2,1]
        s.plt_slice(ax, slc, 'z', 'chi_FUV', norm=LogNorm(1e-1,1e2))
        scatter_sp(sp, ax, 'z', kind='slc', dist_max=50.0, kpc=False, norm_factor=5.0, agemax=20.0)
        ax.axes.xaxis.set_visible(False) ; ax.axes.yaxis.set_visible(False)
        ax.set(xlim=(-512,512),ylim=(-512,512))

        ax = axes[2,2]
        for ph,color in zip(('whole','2p','h'),('grey','b','r')):
            zp = zpa[ph]
            if ph == '2p':
                ax.semilogy(zp.z, zp['xe'][:,num]*zp['d'][:,num], ls=':', label=ph+'_e', c=color)
            ax.semilogy(zp.z, zp['d'][:,num], ls='-', label=ph, c=color)
        
        ax.set(xlabel='z [kpc]', ylabel=r'$\langle n_{\rm H}\rangle\;[{\rm cm}^{-3}]$',
               ylim=(1e-5,5e0))
        ax.legend(loc=1)

        # axes[2,2].remove()
        # gs = fig.add_gridspec(3, 8)
        # ax1 = fig.add_subplot(gs[2, 4])
        # ax2 = fig.add_subplot(gs[2, 5])

        # ax = ax1
        # s.plt_proj(ax, prj, 'y', 'Sigma_gas')
        # pa.scatter_sp(sp, ax, 'y', kind='prj', kpc=False, norm_factor=20.0, agemax=20.0)
        # ax.axes.xaxis.set_visible(False) ; ax.axes.yaxis.set_visible(False)

        # ax = ax2
        # s.plt_slice(ax, slc, 'y', 'chi_FUV', norm=LogNorm(1e-1,1e2))
        # pa.scatter_sp(sp, ax, 'y', kind='slc', kpc=False, norm_factor=20.0, agemax=20.0)
        # ax.axes.xaxis.set_visible(False) ; ax.axes.yaxis.set_visible(False)

        ax = axes[2,3]
        ax.semilogy(hst['time_code'],hst['sfr10'])
        ax.semilogy(hst['time_code'],hst['sfr40'])
        ax.axvline(s.domain['time'], color='grey', lw=0.75)
        ax.set(xlabel='time [code]', ylabel=r'$\Sigma_{\rm SFR}$', ylim=(1e-3,1e0))

        if suptitle is None:
            suptitle = self.basename

        fig.suptitle(suptitle + ' t=' + str(int(ds.domain['time'])), x=0.5, y=1.02,
                     va='center', ha='center', **texteffect(fontsize='xx-large'))

        if savefig:
            savdir = osp.join(savdir, 'pdf2d')
            if not osp.exists(savdir):
                os.makedirs(savdir)

            savname = osp.join(savdir, '{0:s}_{1:04d}_pdf2d.png'.format(self.basename, num))
            plt.savefig(savname, dpi=200, bbox_inches='tight')

        return fig
