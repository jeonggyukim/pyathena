# pdf.py

import os
import os.path as osp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as au
import astropy.constants as ac
from matplotlib.colors import Normalize, LogNorm
import xarray as xr

from ..classic.utils import texteffect
from ..plt_tools.cmap import cmap_apply_alpha
from ..util.scp_to_pc import scp_to_pc
from ..load_sim import LoadSim
from ..plt_tools.plt_starpar import scatter_sp
from ..classic.cooling import coolftn
from ..io.read_hst import read_hst
from .get_cooling import set_bins_default

class PDF:
    bins,nologs = set_bins_default()

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
                          ['nH', 'T'],['nH', 'T'],['nH', 'T'],['nH', 'T']]
        weight_fields_def = ['nH', '2nH2', 'nHI', 'nHII',
                             'nH', '2nH2', 'nHI', 'nHII']
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
            if not (k in res):
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
            res[k][wf] = Hw
            res[k]['vol'] = H
            res[k]['xe'] = xe
            res[k]['ye'] = ye

        res['time_code'] = ds.domain['time']

        return res

    def plt_pdf2d(self, ax, dat, bf='nH-pok',
                  cmap='cubehelix_r',
                  norm=mpl.colors.LogNorm(1e-6,2e-2),
                  kwargs=dict(alpha=1.0, edgecolor='face', linewidth=0, rasterized=True),
                  weighted=True, wfield=None,
                  xscale='log', yscale='log'):

        if weighted:
            hist = 'vol'
        else:
            hist = 'nH'

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
        s.plt_pdf2d(axes[0,0], pdf, 'nH-pok', wfield = 'vol')
        s.plt_pdf2d(axes[0,1], pdf, 'nH-pok', wfield = 'nH')
        s.plt_pdf2d(axes[0,2], pdf, 'nH-chi_FUV', wfield = 'vol')
        s.plt_pdf2d(axes[0,3], pdf, 'nH-xi_CR', wfield = 'nH')
        s.plt_pdf2d(axes[1,0], pdf, 'nH-T', wfield = 'nH')
        s.plt_pdf2d(axes[1,1], pdf, 'nH-T', wfield = '2nH2')
        s.plt_pdf2d(axes[1,2], pdf, 'nH-T', wfield = 'nHI')
        s.plt_pdf2d(axes[1,3], pdf, 'nH-T', wfield = 'nHII')
        s.plt_pdf2d(axes[2,2], pdf, 'nH-chi_H2', wfield = 'vol')

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

    def load_one_jointpdf(self,xf,yf,wf,ds=None,num=None,ivtk=None,zrange=(0,300),
            force_override=False):
        '''Load joind pdfs of a variety of qunatities for a given zrange

        Parameters
        ----------
        xf : str
            x field name
        yf : str
            y field name
        wf : str
            weight field name
        num : int
            vtk snapshot number
        ivtk : int
            vtk snapshot index
        zrange : tuple (zmin,zmax)
            range of |z| over which pdfs are calculated

        '''
        zmin,zmax = zrange
        savdir = '{}/jointpdf_z{:02d}-{:02d}/{}-{}-{}'.format(self.savdir,int(zmin/100),int(zmax/100),xf,yf,wf)
        if not os.path.isdir(savdir): os.makedirs(savdir)
        fbase = os.path.basename(self.fvtk)
        fpdf = os.path.join(savdir,fbase.replace('.vtk','.pdf.nc'))
        if not force_override and osp.exists(fpdf) and osp.getmtime(fpdf) > osp.getmtime(self.fvtk):
            self.logger.info('[jointpdf]: Reading Joint PDFs of {} and {} weigthed by {} at z in +-({},{})'.format(xf,yf,wf,zmin,zmax))
            pdf = xr.open_dataarray(fpdf)
        else:
            self.logger.info('[jointpdf]: Creating Joint PDFs of {} and {} weigthed by {} at z in +-({},{})'.format(xf,yf,wf,zmin,zmax))
            pdf = self.one_jointpdf(ds,xf,yf,wf,zmin=zmin,zmax=zmax)
            pdf.to_netcdf(fpdf)
        pdf.close()
        return pdf

    def one_jointpdf(self,ds,xf,yf,wf,zmin=0,zmax=300):
        if wf is None:
            fields = {xf,yf}
        else:
            fields = {xf,yf,wf}

        self.logger.info('[jointpdf] reading {}'.format(fields))

        data = ds.get_field(list(fields))

        data_zcut = xr.concat([data.sel(z=slice(-zmax,-zmin)),
                               data.sel(z=slice(zmin,zmax))],dim='z')

        x = data_zcut[xf].data.flatten()
        y = data_zcut[yf].data.flatten()

        if wf is None:
            h=np.histogram2d(x,y,bins=[self.bins[xf],self.bins[yf]])
        else:
            w = data_zcut[wf].data.flatten()
            h=np.histogram2d(x,y,bins=[self.bins[xf],self.bins[yf]],weights=w)
        xe = h[1]
        ye = h[2]
        if not (xf in self.nologs):
            xe = np.log10(xe)
            xf = 'log_'+xf
        if not (yf in self.nologs):
            ye = np.log10(ye)
            yf = 'log_'+yf
        xc = 0.5*(xe[1:]+xe[:-1])
        yc = 0.5*(ye[1:]+ye[:-1])
        dx = xe[1]-xe[0]
        dy = ye[1]-ye[0]
        pdf = xr.DataArray(h[0].T/dx/dy,coords=[yc,xc],dims=[yf,xf])
        pdf.name = wf

        return pdf

def plot_pair(pdf,wf='vol',fields=None):
    if fields is None: fields = list(pdf.dims.keys())
    nf = len(fields)
    fig, axes = plt.subplots(nf,nf,figsize=(4*nf,3*nf))

    # 1D histogram
    for i,xf in enumerate(fields):
        plt.sca(axes[i,i])
        try:
            yf = fields[i+1]
            key = '{}-{}-{}'.format(xf,yf,wf)
        except:
            yf = fields[0]
            key = '{}-{}-{}'.format(yf,xf,wf)
        xc = pdf[xf]
        yc = pdf[yf]
        dy = yc[1]-yc[0]
        plt.step(xc,pdf[key].sum(dim=yf)*dy,where='mid')

        plt.yscale('log')
        plt.ylabel('d{}/dlog{}'.format(wf[0].upper(),xf))
        plt.xlabel(xf)

    # 2D histogram in bottom-left
    keylist=[]
    axlist=[]
    for i,xf in enumerate(fields):
        for j,yf in enumerate(fields):
            key ='{}-{}-{}'.format(xf,yf,wf)
            if key in pdf:
                keylist.append(key)
                axlist.append(axes[j,i])
            else:
                if i != j:
                    axes[j,i].axis('off')

    for key,ax in zip(keylist,axlist):
        xf,yf,wf = key.split('-')
        xc = pdf[xf]
        yc = pdf[yf]
        dx = xc[1]-xc[0]
        dy = yc[1]-yc[0]
        xmin = xc.min() - 0.5*dx
        xmax = xc.max() - 0.5*dx
        ymin = yc.min() - 0.5*dy
        ymax = yc.max() - 0.5*dy
        plt.sca(ax)
        plt.imshow(pdf[key],norm=LogNorm(),extent=[xmin,xmax,ymin,ymax],
                   origin='lower',interpolation='nearest')
        ax.set_aspect('auto')
        plt.xlabel(xf)
        plt.ylabel(yf)


