# slc_prj.py

import os
import os.path as osp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
import astropy.units as au
import astropy.constants as ac

from ..load_sim import LoadSim
from ..io.read_starpar_vtk import read_starpar_vtk
from ..plt_tools.cmap_shift import cmap_shift
from ..classic.plot_tools.scatter_sp import scatter_sp
from ..classic.utils import texteffect

class SliceProj:

    @staticmethod
    def get_extent(domain):

        r = dict()
        r['x'] = (domain['le'][1], domain['re'][1],
                  domain['le'][2], domain['re'][2])
        r['y'] = (domain['le'][0], domain['re'][0],
                  domain['le'][2], domain['re'][2])
        r['z'] = (domain['le'][0], domain['re'][0],
                  domain['le'][1], domain['re'][1])

        return r

    @LoadSim.Decorators.check_pickle
    def read_slc(self, num, axes=['x', 'y', 'z'],
                 fields=['nH', 'nH2', 'nHI', 'nHII', 'T', 'nHn', 'chi_PE',
                         'Erad_FUV', 'Erad_LyC'], prefix='slc',
                 savdir=None, force_override=False):

        axes = np.atleast_1d(axes)
        ds = self.load_vtk(num=num)
        res = dict()
        res['extent'] = self.get_extent(ds.domain)
        res['time'] = ds.domain['time']

        for ax in axes:
            dat = ds.get_slice(ax, fields, pos='c', method='nearest')
            res[ax] = dict()
            for f in fields:
                # if 'velocity' in f:
                #     for k in ('3',):
                #         res[ax][f+k] = dat[f+k].data
                # else:
                res[ax][f] = dat[f].data

        return res

    @LoadSim.Decorators.check_pickle
    def read_prj(self, num, axes=['x', 'y', 'z'],
                 fields=['density', 'xHI', 'xH2', 'xHII', 'nesq'],
                 prefix='prj',
                 savdir=None, force_override=False):

        axtoi = dict(x=0, y=1, z=2)
        axes = np.atleast_1d(axes)

        ds = self.load_vtk(num=num)
        dat = ds.get_field(fields, as_xarray=True)
        res = dict()
        res['extent'] = self.get_extent(ds.domain)
        res['time'] = ds.domain['time']

        for ax in axes:
            i = axtoi[ax]
            dx = ds.domain['dx'][i]*self.u.length
            conv_Sigma = (dx*self.u.muH*ac.u.cgs/au.cm**3).to('Msun/pc**2')
            conv_EM = (dx*au.cm**-6).to('pc cm-6')

            res[ax] = dict()
            res[ax]['Sigma'] = (np.sum(dat['density'], axis=2-i)*conv_Sigma).data
            if 'xH2' in fields:
                res[ax]['Sigma_H2'] = (np.sum(2.0*dat['density']*dat['xH2'],
                                              axis=2-i)*conv_Sigma).data
            if 'xHI' in fields:
                res[ax]['Sigma_HI'] = (np.sum(dat['density']*dat['xHI'],
                                              axis=2-i)*conv_Sigma).data
            if 'xHII' in fields:
                res[ax]['Sigma_HII'] = (np.sum(dat['density']*dat['xHII'],
                                               axis=2-i)*conv_Sigma).data
            if 'nesq' in fields:
                res[ax]['EM'] = (np.sum(dat['nesq'], axis=2-i)*conv_EM).data

            if 'specific_scalar[1]' in fields:
                res[ax]['Sigma_scalar1'] = (np.sum(dat['density']*dat['specific_scalar[1]'],
                                                   axis=2-i)*conv_Sigma).data
            if 'specific_scalar[2]' in fields:
                res[ax]['Sigma_scalar2'] = (np.sum(dat['density']*dat['specific_scalar[2]'],
                                                   axis=2-i)*conv_Sigma).data


        return res

    @staticmethod
    def plt_imshow(ax, dat, dim='z', field='Sigma', cmap='viridis',
                   norm=mpl.colors.LogNorm()):
        im = ax.imshow(dat[dim][field], cmap=cmap, extent=dat['extent'][dim],
                       norm=norm, origin='lower', interpolation='none')
        return im

    def plt_snapshot(self, num, savefig=True):

        d = self.read_prj(num, force_override=False)
        sp = self.load_starpar_vtk(num)
        nr = 3
        nc = 4
        fig, axes = plt.subplots(nr, nc, figsize=(16, 12.5), # constrained_layout=True,
                                 gridspec_kw=dict(hspace=0.0, wspace=0.0))

        norm = LogNorm(1e-1,1e3)
        norm_EM = LogNorm(3e1,3e5)
        im1 = []
        im2 = []
        im3 = []
        im4 = []
        for i, axis in enumerate(('x','y','z')):
            extent = d['extent'][axis]
            im1.append(axes[i, 0].imshow(d[axis]['Sigma'], norm=norm,
                                         extent=extent, origin='lower'))
            im2.append(axes[i, 1].imshow(d[axis]['Sigma_H2'], norm=norm,
                                         extent=extent, origin='lower'))
            im3.append(axes[i, 2].imshow(d[axis]['Sigma_HI'], norm=norm,
                                         extent=extent, origin='lower'))
            im4.append(axes[i, 3].imshow(d[axis]['EM'], norm=norm_EM,
                                         extent=extent, origin='lower', cmap='plasma'))

            # Overplot starpar
            if not sp.empty:
                scatter_sp(sp, axes[i, 0], axis=axis, type='proj', kpc=False,
                           norm_factor=4.0, agemax=10.0)
                scatter_sp(sp, axes[i, 3], axis=axis, type='proj', kpc=False,
                           norm_factor=4.0, agemax=10.0)

        for ax in axes.flatten():
            plt.axis('on')
            ax.set_xticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')

        # Add colorbars
        labels = [r'$\Sigma_{\rm gas}\;[{M_{\odot}\,{\rm pc}^{-2}}]$',
                  r'$\Sigma_{\rm H_2}\;[{M_{\odot}\,{\rm pc}^{-2}}]$',
                  r'$\Sigma_{\rm H\,I}\;[{M_{\odot}\,{\rm pc}^{-2}}]$',
                  r'${\rm EM}\;[{\rm pc}\,{\rm cm}^{-6}]$']

        for j,im,label in zip(range(nc),(im1,im2,im3,im4),labels):
            bbox_ax_top = axes[0,j].get_position()
            cax = fig.add_axes([bbox_ax_top.x0+0.01, bbox_ax_top.y1+0.01,
                                bbox_ax_top.x1-bbox_ax_top.x0-0.02, 0.015])
            cbar = plt.colorbar(im[0], cax=cax, orientation='horizontal')
            cbar.set_label(label=label, fontsize='small')
            cbar.ax.xaxis.set_ticks_position('top')
            cbar.ax.xaxis.set_label_position('top')
            cbar_yticks = plt.getp(cbar.ax.axes, 'xticklabels')
            plt.setp(cbar_yticks, color='k', fontsize='x-small')
            #cbar.ax.set_yticks(arange(vmin, vmax, 2), size='small')

        plt.subplots_adjust(wspace=None, hspace=None)
        plt.suptitle(self.basename + '  t={0:4.1f}'.format(sp.time))

        if savefig:
            savdir = osp.join(self.savdir, 'snapshots')
            # savdir = osp.join('/tigress/jk11/figures/GMC', self.basename, 'snapshots')
            if not osp.exists(savdir):
                os.makedirs(savdir)

            savname = osp.join(savdir, '{0:s}_{1:04d}.png'.format(self.basename, num))
            plt.savefig(savname, dpi=200, bbox_inches='tight')

        return fig
