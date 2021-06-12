# slc_prj.py

import os
import os.path as osp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import astropy.units as au
import astropy.constants as ac
from matplotlib.colors import Normalize, LogNorm
from mpl_toolkits.axes_grid1 import ImageGrid

from ..load_sim import LoadSim
from ..io.read_starpar_vtk import read_starpar_vtk
from ..plt_tools.cmap_shift import cmap_shift
from ..plt_tools.plt_starpar import scatter_sp
from ..classic.utils import texteffect

cmap_def = dict(
    Sigma_gas=plt.cm.pink_r,
    Sigma_H2=plt.cm.pink_r,
    EM=plt.cm.plasma,
    nH=plt.cm.Spectral_r,
    T=cmap_shift(mpl.cm.RdYlBu_r, midpoint=3./7.),
    vz=plt.cm.bwr,
    chi_FUV=plt.cm.viridis,
    Erad_LyC=plt.cm.viridis,
    xi_CR=plt.cm.viridis,
    Bmag=plt.cm.cividis,
)

norm_def = dict(
    Sigma_gas=LogNorm(1e-2,1e2),
    Sigma_H2=LogNorm(1e-2,1e2),
    EM=LogNorm(1e0,1e5),
    nH=LogNorm(1e-4,1e3),
    T=LogNorm(1e1,1e7),
    vz=Normalize(-200,200),
    chi_FUV=LogNorm(1e-2,1e2),
    Erad_LyC=LogNorm(1e-16,5e-13),
    xi_CR=LogNorm(5e-17,1e-15),
    Bmag=LogNorm(1.e-2,1.e2)
)

class SliceProj:

    @staticmethod
    def _get_extent(domain):

        r = dict()
        r['x'] = (domain['le'][1], domain['re'][1],
                  domain['le'][2], domain['re'][2])
        r['y'] = (domain['le'][0], domain['re'][0],
                  domain['le'][2], domain['re'][2])
        r['z'] = (domain['le'][0], domain['re'][0],
                  domain['le'][1], domain['re'][1])

        return r

    @LoadSim.Decorators.check_pickle
    def read_slc(self, num, axes=['x', 'y', 'z'], fields=None, prefix='slc',
                 savdir=None, force_override=False):

        fields_def = ['nH', 'nH2', 'vz', 'T', 'cs', 'vx', 'vy', 'vz', 'pok']
        if self.par['configure']['radps'] == 'ON':
            if (self.par['cooling']['iCR_attenuation']):
                fields_def += ['xi_CR']
            if self.par['radps']['iPhotIon'] == 1:
                fields_def += ['Erad_LyC']
            if self.par['cooling']['iPEheating'] == 1:
                fields_def += ['chi_FUV']
        if self.par['configure']['gas'] == 'mhd':
            fields_def += ['Bx','By','Bz','Bmag']

        fields = fields_def
        axes = np.atleast_1d(axes)

        ds = self.load_vtk(num=num)
        res = dict()
        res['extent'] = self._get_extent(ds.domain)

        for ax in axes:
            dat = ds.get_slice(ax, fields, pos='c', method='nearest')
            res[ax] = dict()
            for f in fields:
                res[ax][f] = dat[f].data

        for zpos,zlab in zip([-1000,-500,500,1000],['zn10','zn05','zp05','zp10']):
            dat = ds.get_slice('z', fields, pos=zpos, method='nearest')
            res[zlab] = dict()
            for f in fields:
                res[ax][f] = dat[f].data

        return res

    @LoadSim.Decorators.check_pickle
    def read_prj(self, num, axes=['x', 'y', 'z'], prefix='prj',
                 savdir=None, force_override=False):

        axtoi = dict(x=0, y=1, z=2)
        fields = ['nH', 'nH2', 'nesq']
        axes = np.atleast_1d(axes)

        ds = self.load_vtk(num=num)
        dat = ds.get_field(fields, as_xarray=True)

        res = dict()
        res['extent'] = self._get_extent(ds.domain)

        for ax in axes:
            i = axtoi[ax]
            dx = ds.domain['dx'][i]*self.u.length
            conv_Sigma = (dx*self.u.muH*ac.u.cgs/au.cm**3).to('Msun/pc**2')
            conv_EM = (dx*au.cm**-6).to('pc cm-6')

            res[ax] = dict()
            res[ax]['Sigma_gas'] = (np.sum(dat['nH'], axis=2-i)*conv_Sigma).data
            res[ax]['Sigma_H2'] = (2.0*np.sum(dat['nH2'], axis=2-i)*conv_Sigma).data
            res[ax]['Sigma_HI'] = res[ax]['Sigma_gas'] - res[ax]['Sigma_H2']
            res[ax]['EM'] = (np.sum(dat['nesq'], axis=2-i)*conv_EM).data

        return res

    @staticmethod
    def plt_slice(ax, slc, axis='z', field='density', cmap=None, norm=None):
        try:
            if cmap is None:
                cmap = cmap_def[field]

            if norm is None:
                norm = mpl.colors.LogNorm()
            elif norm is 'linear':
                norm = mpl.colors.Normalize()

            ax.imshow(slc[axis][field], cmap=cmap,
                      extent=slc['extent'][axis], norm=norm, origin='lower', interpolation='none')
        except KeyError:
            pass

    @staticmethod
    def plt_proj(ax, prj, axis='z', field='Sigma_gas',
                 cmap=None, norm=None, vmin=None, vmax=None):
        try:
            vminmax = dict(Sigma_gas=(1e-2,1e2))
            cmap_def = dict(Sigma_gas='pink_r')

            if cmap is None:
                try:
                    cmap = cmap_def[field]
                except KeyError:
                    cmap = plt.cm.viridis
            if vmin is None or vmax is None:
                vmin = vminmax[field][0]
                vmax = vminmax[field][1]

            if norm is None or 'log':
                norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
            elif norm is 'linear':
                norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

            ax.imshow(prj[axis][field], cmap=cmap, extent=prj['extent'][axis],
                      norm=norm, origin='lower', interpolation='none')
        except KeyError:
            pass

    def plt_snapshot(self, num,
                     fields_xy=('Sigma_gas', 'Sigma_H2', 'EM', 'nH', 'T', 'chi_FUV'),
                     fields_xz=('Sigma_gas', 'Sigma_H2', 'EM', 'nH', 'T', 'vz', 'Bmag'),
                     #fields_xy=('Sigma_gas', 'EM', 'xi_CR', 'nH', 'chi_FUV', 'Erad_LyC'),
                     #fields_xz=('Sigma_gas', 'EM', 'nH', 'chi_FUV', 'Erad_LyC', 'xi_CR'),
                     norm_factor=5.0, agemax=20.0, agemax_sn=40.0, runaway=False,
                     suptitle=None, savdir_pkl=None, savdir=None, force_override=False, savefig=True):
        """Plot 12-panel projection, slice plots in the z and y directions

        Parameters
        ----------
        num : int
            vtk snapshot number
        fields_xy: list of str
            field names for z projections and slices
        fields_xz: list of str
            Field names for y projections and slices
        norm_factor : float
            Normalization factor for starpar size. The smaller the value the
            bigger the size)
        agemax : float
            Maximum age of source particles
        agemax_sn : float
            Maximum age of sn particles
        """

        label = dict(Sigma_gas=r'$\Sigma$',
                     Sigma_H2=r'$\Sigma_{\rm H_2}$',
                     EM=r'${\rm EM}$',
                     nH=r'$n_{\rm H}$',
                     T=r'$T$',
                     vz=r'$v_z$',
                     chi_FUV=r'$\mathcal{E}_{\rm FUV}$',
                     Erad_LyC=r'$\mathcal{E}_{\rm LyC}$',
                     xi_CR=r'$\xi_{\rm CR}$',
                     Bmag=r'$|B|$'
        )

        kind = dict(Sigma_gas='prj', Sigma_H2='prj', EM='prj',
                    nH='slc', T='slc', vz='slc', chi_FUV='slc',
                    Erad_LyC='slc', xi_CR='slc', Bmag='slc')
        nxy = len(fields_xy)
        nxz = len(fields_xz)
        ds = self.load_vtk(num=num)
        LzoLx = ds.domain['Lx'][2]/ds.domain['Lx'][0]
        xwidth = 3
        ysize = LzoLx*xwidth
        xsize = ysize/nxy*4 + nxz*xwidth
        x1 = 0.90*(ysize*4/nxy/xsize)
        x2 = 0.90*(nxz*xwidth/xsize)

        fig = plt.figure(figsize=(xsize, ysize))#, constrained_layout=True)
        g1 = ImageGrid(fig, [0.02, 0.05, x1, 0.94], (nxy//2, 2), axes_pad=0.1,
                       aspect=True, share_all=True, direction='column')
        g2 = ImageGrid(fig, [x1+0.07, 0.05, x2, 0.94], (1, nxz), axes_pad=0.1,
                       aspect=True, share_all=True)

        dat = dict()
        dat['slc'] = self.read_slc(num, savdir=savdir_pkl, force_override=force_override)
        dat['prj'] = self.read_prj(num, savdir=savdir_pkl, force_override=force_override)
        sp = self.load_starpar_vtk(num)

        extent = dat['prj']['extent']['z']
        for i, (ax, f) in enumerate(zip(g1, fields_xy)):
            ax.set_aspect(ds.domain['Lx'][1]/ds.domain['Lx'][0])
            self.plt_slice(ax, dat[kind[f]], 'z', f, cmap=cmap_def[f], norm=norm_def[f])

            if i == 0:
                scatter_sp(sp, ax, 'z', kind='prj', kpc=False,
                           norm_factor=norm_factor, agemax=agemax, agemax_sn=agemax_sn,
                           runaway=runaway, cmap=plt.cm.cool_r)
            ax.set(xlim=(extent[0], extent[1]), ylim=(extent[2], extent[3]))
            ax.text(0.5, 0.92, label[f], **texteffect(fontsize='x-large'),
                    ha='center', transform=ax.transAxes)
            if i == 2:
                ax.set(xlabel='x [pc]', ylabel='y [pc]')
            else:
                ax.axes.get_xaxis().set_visible(False)
                ax.axes.get_yaxis().set_visible(False)

        extent = dat['prj']['extent']['y']
        for i, (ax, f) in enumerate(zip(g2, fields_xz)):
            ax.set_aspect(ds.domain['Lx'][2]/ds.domain['Lx'][0])
            self.plt_slice(ax, dat[kind[f]], 'y', f, cmap=cmap_def[f], norm=norm_def[f])
            if i == 0:
                scatter_sp(sp, ax, 'y', kind='prj', kpc=False,
                           norm_factor=norm_factor, agemax=agemax,
                           cmap=plt.cm.cool_r)
            ax.set(xlim=(extent[0], extent[1]), ylim=(extent[2], extent[3]))
            ax.text(0.5, 0.97, label[f], **texteffect(fontsize='x-large'),
                    ha='center', transform=ax.transAxes)
            if i == 0:
                ax.set(xlabel='x [pc]', ylabel='z [pc]')
            else:
                ax.axes.get_xaxis().set_visible(False)
                ax.axes.get_yaxis().set_visible(False)

        if suptitle is None:
            suptitle = self.basename
        # fig.suptitle(suptitle + ' t=' + str(int(ds.domain['time'])), x=0.4, y=1.02,
        #              va='center', ha='center', **texteffect(fontsize='xx-large'))
        fig.suptitle('Model: {0:s}  time='.format(suptitle) + str(int(ds.domain['time'])), x=0.4, y=1.02,
                     va='center', ha='center', **texteffect(fontsize='xx-large'))
        # plt.subplots_adjust(top=0.95)

        if savefig:
            if savdir is None:
                savdir = osp.join(self.savdir, 'snapshot')
            if not osp.exists(savdir):
                os.makedirs(savdir)

            savname = osp.join(savdir, '{0:s}_{1:04d}.png'.format(self.basename, num))
            plt.savefig(savname, dpi=200, bbox_inches='tight')

        return fig
