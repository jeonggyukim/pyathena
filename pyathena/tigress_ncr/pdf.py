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
from ..plt_tools.cmap import cmap_apply_alpha
from ..util.scp_to_pc import scp_to_pc
from ..load_sim import LoadSim
from ..plt_tools.plt_starpar import scatter_sp

bins_def=dict(
    nH=np.logspace(-5,4,181),
    nHI=np.logspace(-2,5,141),
    nH2=np.logspace(-2,5,141),
    nHII=np.logspace(-5,3,201),
    xH2=np.linspace(0,0.5,101),
    xHI=np.linspace(0,1.0,101),
    xHII=np.linspace(0,1.0,101),
    xe=np.logspace(-5,np.log10(2),201),
    T=np.logspace(1,8,281),
    pok=np.logspace(0,7,141),
    chi_PE=np.logspace(-3,4,141),
    chi_H2=np.logspace(-6,4,201),
    chi_FUV=np.logspace(-3,4,141),
    Erad_LyC=np.logspace(-18,-10,161),
    Lambda_cool=np.logspace(-30,-20,201),
    xi_CR=np.logspace(-17,-14,121)
)

class PDF:

    @LoadSim.Decorators.check_pickle
    def read_pdf2d_avg(self, nums=None, savdir=None, force_override=False):
        """Take sum of all pdf2d
        """

        if nums is None:
            nums = self.nums

        rr = dict()
        print('[read_pdf2d_avg]:', end=' ')
        for i,num in enumerate(nums):
            print(num, end=' ')
            r = self.read_pdf2d(num, force_override=False)
            if i == 0:
                for k in r.keys():
                    if k == 'time_code':
                        rr[k] = []
                    else:
                        rr[k] = dict()
                        rr[k]['xe'] = r[k]['xe']
                        rr[k]['ye'] = r[k]['ye']
                        rr[k]['H'] = np.zeros_like(r[k]['H'])
                        rr[k]['Hw'] = np.zeros_like(r[k]['Hw'])

            for k in r.keys():
                if k == 'time_code':
                    rr[k].append(r[k])
                else:
                    rr[k]['H'] += r[k]['H']
                    rr[k]['Hw'] += r[k]['Hw']

        return rr

    @LoadSim.Decorators.check_pickle
    def read_pdf2d(self, num,
                   bin_fields=None,
                   weight_fields=None,
                   bins=None, prefix='pdf2d',
                   savdir=None, force_override=False):

        bin_fields_def = [['nH', 'pok'], ['nH', 'pok'], ['nH', 'pok'], ['nH', 'pok'],
                          ['nH', 'T']]
        weight_fields_def = ['nH', '2nH2', 'nHI', 'nHII',
                             'nH']
        if self.par['configure']['radps'] == 'ON':
            bin_fields_def += [['T','Lambda_cool'], ['nH','xH2'],
                               ['T','xHII'], ['T', 'xHI']]
            weight_fields_def += ['cool_rate', 'nH', 'nH', 'nH']
            if (self.par['cooling']['iCR_attenuation']):
                bin_fields_def += [['nH','xi_CR']]
                weight_fields_def += ['nH']
            if (self.par['cooling']['iPEheating'] == 1):
                bin_fields_def += [['nH','chi_FUV']]
                weight_fields_def += ['nH']
            if (self.par['radps']['iPhotDiss'] == 1):
                bin_fields_def += [['nH','chi_H2']]
                weight_fields_def += ['nH']
            if (self.par['radps']['iPhotIon'] == 1):
                bin_fields_def += [['nH','Erad_LyC']]
                weight_fields_def += ['nH']

        if bin_fields is None:
            bin_fields = bin_fields_def
            weight_fields = weight_fields_def

        ds = self.load_vtk(num=num)
        res = dict()
        fields = np.unique(np.append(np.unique(bin_fields),
                                     np.unique(weight_fields +
                                               ['xHI','xH2','xHII'])))
        dd = ds.get_field(fields)
        dd = dd.stack(xyz=['x','y','z']).dropna(dim='xyz')
        for bf,wf in zip(bin_fields,weight_fields):
            k = '-'.join(bf)
            res[k] = dict()
            xdat = dd[bf[0]]
            ydat = dd[bf[1]]
            xbins = self.bins[bf[0]]
            ybins = self.bins[bf[1]]
            weights = dd[wf]
            # Unweighted hist (volume-weighted)
            H, xe, ye = np.histogram2d(xdat, ydat, (xbins, ybins), weights=None)
            # Weighted hist
            Hw, xe, ye = np.histogram2d(xdat, ydat, (xbins, ybins),
                                        weights=weights)
            res[k]['Hw'] = Hw
            res[k]['H'] = H
            res[k]['xe'] = xe
            res[k]['ye'] = ye

        # # We don't need this if weights are set as nHI, 2nH2, or nHII
        # # nH-T weighted by H2, HI, and HII masses
        # k = 'nH-T'
        # xdat = dd['nH']
        # ydat = dd['T']
        # xbins = self.bins['nH']
        # ybins = self.bins['T']
        # weights = 2.0*dd['xH2']*dd['nH']
        # Hw, xe, ye = np.histogram2d(xdat, ydat, (xbins, ybins), weights=weights)
        # res[k]['MH2'] = Hw
        # weights = dd['xHI']*dd['nH']
        # Hw, xe, ye = np.histogram2d(xdat, ydat, (xbins, ybins), weights=weights)
        # res[k]['MHI'] = Hw
        # weights = dd['xHII']*dd['nH']
        # Hw, xe, ye = np.histogram2d(xdat, ydat, (xbins, ybins), weights=weights)
        # res[k]['MHII'] = Hw

        res['time_code'] = ds.domain['time']

        return res

    def plt_pdf2d(self, ax, dat, bf='nH-pok',
                  cmap='cubehelix_r',
                  norm=mpl.colors.LogNorm(1e-6,2e-2),
                  kwargs=dict(alpha=1.0, edgecolor='face', linewidth=0, rasterized=True),
                  weighted=True, wfield=None,
                  xscale='log', yscale='log'):

        if weighted:
            hist = 'Hw'
        else:
            hist = 'H'

        if wfield is not None:
            hist = wfield
            ax.annotate(wfield,(0.05,0.95),xycoords='axes fraction',ha='left',va='top')

        try:
            pdf = dat[bf][hist].T/dat[bf][hist].sum()

            ax.pcolormesh(dat[bf]['xe'], dat[bf]['ye'], pdf,
                          norm=norm, cmap=cmap, **kwargs)

            kx, ky = bf.split('-')
            ax.set(xscale=xscale, yscale=yscale,
                   xlabel=self.dfi[kx]['label'], ylabel=self.dfi[ky]['label'])
        except KeyError:
            pass

    def plt_pdf2d_all(self, num, suptitle=None, savdir=None,
                      plt_zprof=True, savdir_pkl=None,
                      force_override=False, savefig=True):

        if savdir is None:
            savdir = self.savdir

        s = self
        ds = s.load_vtk(num)
        pdf = s.read_pdf2d(num, savdir=savdir_pkl, force_override=force_override)
        prj = s.read_prj(num, savdir=savdir_pkl, force_override=force_override)
        slc = s.read_slc(num, savdir=savdir_pkl, force_override=force_override)
        hst = s.read_hst(savdir=savdir, force_override=force_override)
        sp = s.load_starpar_vtk(num)
        if plt_zprof:
            zpa = s.read_zprof(['whole','2p','h'], savdir=savdir, force_override=force_override)

        fig, axes = plt.subplots(3,4,figsize=(20,15), constrained_layout=True)

        # gs = axes[0, -1].get_gridspec()
        # for ax in axes[0:2, -1]:
        #     ax.remove()
        # ax = fig.add_subplot(gs[0:2, -1])

        #s.plt_pdf2d(axes[0,0], pdf, 'nH-pok', weighted=False)
        s.plt_pdf2d(axes[0,0], pdf, 'nH-pok', weighted=True)
        #s.plt_pdf2d(axes[0,1], pdf, 'nH-chi_FUV', weighted=False)
        s.plt_pdf2d(axes[0,1], pdf, 'nH-chi_FUV', weighted=True)
        #s.plt_pdf2d(axes[0,2], pdf, 'T-Lambda_cool', weighted=False)
        s.plt_pdf2d(axes[0,2], pdf, 'T-Lambda_cool', weighted=True)
        #s.plt_pdf2d(axes[0,3], pdf, 'nH-xi_CR', weighted=False)
        s.plt_pdf2d(axes[0,3], pdf, 'nH-xi_CR', weighted=True)
        s.plt_pdf2d(axes[1,0], pdf, 'nH-T', weighted=True)
        s.plt_pdf2d(axes[1,1], pdf, 'nH-T', wfield = 'MH2')
        s.plt_pdf2d(axes[1,2], pdf, 'nH-T', wfield = 'MHI')
        s.plt_pdf2d(axes[1,3], pdf, 'nH-T', wfield = 'MHII')
        s.plt_pdf2d(axes[2,2], pdf, 'nH-chi_H2', weighted=False)

        ax = axes[2,0]
        # s.plt_proj(ax, prj, 'z', 'Sigma_gas')
        ax.imshow(prj['z']['Sigma_gas'], cmap='pink_r',
                  extent=prj['extent']['z'], norm=mpl.colors.LogNorm(),
                  origin='lower', interpolation='none')
        scatter_sp(sp, ax, 'z', kind='prj', kpc=False, norm_factor=5.0, agemax=20.0)
        ax.axis('off')
        #ax.axes.xaxis.set_visible(False) ; ax.axes.yaxis.set_visible(False)
        ax.set(xlim=(ds.domain['le'][0], ds.domain['re'][0]),
               ylim=(ds.domain['le'][1], ds.domain['re'][1]))

        ax = axes[2,1]
        s.plt_slice(ax, slc, 'z', 'chi_FUV', norm=LogNorm(1e-1,1e2))
        #scatter_sp(sp, ax, 'z', kind='slc', dist_max=50.0, kpc=False, norm_factor=5.0, agemax=20.0)
        ax.axis('off')
        #ax.axes.xaxis.set_visible(False) ; ax.axes.yaxis.set_visible(False)
        ax.set(xlim=(ds.domain['le'][0], ds.domain['re'][0]),
               ylim=(ds.domain['le'][1], ds.domain['re'][1]))

        if plt_zprof:
            ax = axes[2,2]
            for ph,color in zip(('whole','2p','h'),('grey','b','r')):
                zp = zpa[ph]
                if ph == '2p':
                    ax.semilogy(zp.z, zp['xe'][:,num]*zp['d'][:,num],
                                ls=':', label=ph+'_e', c=color)
                ax.semilogy(zp.z, zp['d'][:,num], ls='-', label=ph, c=color)
                ax.set(xlabel='z [kpc]', ylabel=r'$\langle n_{\rm H}\rangle\;[{\rm cm}^{-3}]$',
                       ylim=(1e-5,5e0))
                ax.legend(loc=1)

        # axes[2,2].remove()
        # gs = fig.add_gridspec(3, 8)
        # ax1 = fig.add_subplot(gs[2, 4])
        # ax2 = fig.add_subplot(gs[2, 5])

        # ax = ax1
        s.plt_proj(ax, prj, 'y', 'Sigma_gas')
        scatter_sp(sp, ax, 'y', kind='prj', kpc=False, norm_factor=20.0, agemax=20.0)
        ax.axes.xaxis.set_visible(False) ; ax.axes.yaxis.set_visible(False)

        # ax = ax2
        s.plt_slice(ax, slc, 'y', 'chi_FUV', norm=LogNorm(1e-1,1e2))
        scatter_sp(sp, ax, 'y', kind='slc', kpc=False, norm_factor=20.0, agemax=20.0)
        ax.axes.xaxis.set_visible(False) ; ax.axes.yaxis.set_visible(False)

        ax = axes[2,3]
        ax.semilogy(hst['time_code'],hst['sfr10'])
        ax.semilogy(hst['time_code'],hst['sfr40'])
        ax.axvline(s.domain['time'], color='grey', lw=0.75)
        ax.set(xlabel='time [code]', ylabel=r'$\Sigma_{\rm SFR}$', ylim=(1e-3,None))

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
            plt.close()

        return fig
