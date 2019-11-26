# read_slc_prj.py

import os
import os.path as osp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
import astropy.units as au
import astropy.constants as ac

from inspect import getcallargs

import pyathena as pa
from ..load_sim import LoadSim
from ..io.read_starpar_vtk import read_starpar_vtk
from ..plt_tools.cmap_shift import cmap_shift
from ..classic.plot_tools.scatter_sp import scatter_sp
from ..classic.utils import texteffect

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
    def read_slc(self, num, axes=['x', 'y', 'z'], fields=None, dirname='slc',
                 savdir=None, force_override=False):
        
        fields_def = ['density', 'xH2', 'velocity', 'temperature',
                      'CR_ionization_rate', 'heat_ratio']
        fields = fields_def
        axes = np.atleast_1d(axes)

        ds = self.load_vtk(num=num)
        res = dict()
        res['extent'] = self._get_extent(ds.domain)
        
        for ax in axes:
            dat = ds.get_slice(ax, fields, pos='c', method='nearest')
            res[ax] = dict()
            for f in fields:
                if 'velocity' in f:
                    for k in ('3',):
                        res[ax][f+k] = dat[f+k].data
                else:
                    res[ax][f] = dat[f].data

        return res

    @LoadSim.Decorators.check_pickle
    def read_prj(self, num, axes=['x', 'y', 'z'], dirname='prj',
                 savdir=None, force_override=False):

        axtoi = dict(x=0, y=1, z=2)
        fields = ['density', 'xH2', 'temperature', 'heat_ratio']
        axes = np.atleast_1d(axes)

        ds = self.load_vtk(num=num)
        dat = ds.get_field(fields, as_xarray=True)

        res = dict()
        res['extent'] = self._get_extent(ds.domain)

        for ax in axes:
            i = axtoi[ax]
            dx = ds.domain['dx'][i]*self.u.length
            conv_Sigma = (dx*self.u.muH*ac.u.cgs/au.cm**3).to('Msun/pc**2')
            
            res[ax] = dict()
            res[ax]['Sigma_gas'] = (np.sum(dat['density'], axis=2-i)*conv_Sigma).data
            res[ax]['Sigma_H2'] = (np.sum(dat['density']*dat['xH2'], axis=2-i)*conv_Sigma).data
            res[ax]['Sigma_HI'] = res[ax]['Sigma_gas'] - res[ax]['Sigma_H2']
            res[ax]['heat_ratio'] = np.sum(dat['heat_ratio'], axis=2-i)

        return res

    @staticmethod
    def plt_slice(ax, slc, axis='z', field='density', cmap=None, norm=None):

        if cmap is None:
                cmap = 'viridis'

        if norm is None:
            norm = mpl.colors.LogNorm()
        elif norm is 'linear':
            norm = mpl.colors.Normalize()

        ax.imshow(slc[axis][field], cmap=cmap,
                  extent=slc['extent'][axis], norm=norm, origin='lower', interpolation='none')
            
    @staticmethod        
    def plt_proj(ax, prj, axis='z', field='density', cmap=None, norm=None, vmin=None, vmax=None):
        if cmap is None:
            cmap = 'viridis'

        if norm is None or 'log':
            norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
        elif norm is 'linear':
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

        ax.imshow(prj[axis][field], cmap=cmap, extent=prj['extent'][axis],
                  norm=norm, origin='lower', interpolation='none')

    def plt_snapshot(self, num, axis='z', savefig=True):


        cmap = dict(Sigma_gas='pink_r', Sigma_HI='pink_r', Sigma_H2='pink_r', \
                    density='Spectral_r',
                    heat_ratio='viridis',
                    temperature=cmap_shift(mpl.cm.RdYlBu_r, midpoint=3./7.))
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 12),
                                 gridspec_kw=dict(wspace=0.0, hspace=0.0))

        ds = self.load_vtk(num)
        prj = self.read_prj(num, force_override=False)
        slc = self.read_slc(num, force_override=False)
        sp = read_starpar_vtk(self.files['starpar'][num])

        axes = axes.flatten()
        self.plt_proj(axes[0], prj, 'z', 'Sigma_gas', norm='log',
                      cmap=cmap['Sigma_gas'], vmin=0.1, vmax=10e2)
        self.plt_proj(axes[1], prj, 'z', 'Sigma_H2', norm='log',
                      cmap=cmap['Sigma_H2'], vmin=0.1, vmax=10e2)
        self.plt_proj(axes[2], slc, 'z', 'density', norm='log', 
                      cmap=cmap['density'], vmin=1e-4, vmax=1e3)
        self.plt_proj(axes[3], slc, 'z', 'heat_ratio', norm='log',
                      cmap=cmap['heat_ratio'], vmin=0.1, vmax=1e3)

        for ax in (axes[0], axes[1]):
            scatter_sp(sp, ax, type='proj', kpc=False, norm_factor=5.0, agemax=40.0)
            extent = prj['extent']['z']
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])

        for i, (ax, title) in enumerate(zip(axes, 
               (r'$N_{\rm H}$', r'$N_{\rm H_2}$', r'$n_{\rm H}$', r'$G_{\rm 0}$'))):
            if i != 2:
                ax.axes.get_xaxis().set_visible(False)
                ax.axes.get_yaxis().set_visible(False)
            else:
                ax.set_xlabel('x [pc]')
                ax.set_ylabel('y [pc]')

            ax.text(0, 220, title, **texteffect(fontsize='xx-large'), ha='center')

        fig.suptitle('t=' + str(int(ds.domain['time'])),
                     ha='center', fontsize='xx-large')
        plt.subplots_adjust(top=0.97)

        if savefig:
            savdir = osp.join(self.savdir, 'snapshots')
            if not osp.exists(savdir):
                os.makedirs(savdir)

            savname = osp.join(savdir, '{0:s}_{1:04d}.png'.format(self.basename, num))
            plt.savefig(savname, dpi=200)
            
        return fig
            
    def plt_snapshot2(self, num, force_override=False, savefig=True):

        from mpl_toolkits.axes_grid1 import ImageGrid

        ds = self.load_vtk(num=num)
        fig = plt.figure(figsize=(22, 12))

        g1 = ImageGrid(fig, [0.02, 0.05, 0.4, 0.94], (3, 2), axes_pad=0.1,
                       aspect=True, share_all=True)
        for i in range(6):
            g1[i].set_aspect(1)

        g2 = ImageGrid(fig, [0.32, 0.05, 0.85, 0.94], (1, 6), axes_pad=0.1,
                          aspect=True, share_all=True)
        for i in range(6):
            g2[i].set_aspect(ds.domain['Lx'][2]/ds.domain['Lx'][0])

        slc = self.read_slc(num, force_override=force_override)
        prj = self.read_prj(num, force_override=force_override)
        sp = read_starpar_vtk(self.files['starpar'][num])

        self.plt_slice(g1[0], prj, 'z', 'Sigma_gas', cmap='pink_r', norm=LogNorm(5e-1,1e3))
        self.plt_slice(g1[1], prj, 'z', 'Sigma_H2', cmap='pink_r', norm=LogNorm(5e-1,1e3))
        self.plt_slice(g1[2], slc, 'z', 'density', cmap='Spectral_r', norm=LogNorm(1e-3,1e3))
        self.plt_slice(g1[3], slc, 'z', 'temperature', cmap=cmap_shift(mpl.cm.RdYlBu_r, midpoint=3./7.),
                    norm=LogNorm(1e1,1e7))
        self.plt_slice(g1[4], slc, 'z', 'heat_ratio', cmap='viridis', norm=LogNorm(0.01,1e3))
        self.plt_slice(g1[5], slc, 'z', 'CR_ionization_rate', cmap='viridis', norm=LogNorm(1e-16,1e-14))

        self.plt_slice(g2[0], prj, 'y', 'Sigma_gas', cmap='pink_r', norm=LogNorm(5e-1,1e3))
        self.plt_slice(g2[1], prj, 'y', 'Sigma_H2', cmap='pink_r', norm=LogNorm(5e-1,1e3))
        self.plt_slice(g2[2], slc, 'y', 'density', cmap='Spectral_r', norm=LogNorm(1e-3,1e3))
        self.plt_slice(g2[3], slc, 'y', 'velocity3', cmap='bwr', norm=Normalize(-300,300))
        self.plt_slice(g2[4], slc, 'y', 'temperature', cmap=cmap_shift(mpl.cm.RdYlBu_r, midpoint=3./7.),
                    norm=LogNorm(1e1,1e7))
        self.plt_slice(g2[5], slc, 'y', 'heat_ratio', cmap='viridis', norm=LogNorm(0.01,1e3))

        for ax in (g1[0], g1[1]):
            scatter_sp(sp, ax, type='proj', kpc=False, norm_factor=5.0, agemax=20.0)
            extent = prj['extent']['z']
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
        
        titles1 = (r'$N_{\rm H}$', r'$N_{\rm H_2}$', r'$n_{\rm H}$', r'$T$', r'$G_0$', r'CRIR')
        for i, (ax, title) in enumerate(zip(g1, titles1)):
            ax.text(0.5, 0.92, title, **texteffect(fontsize='xx-large'), ha='center', transform=ax.transAxes)
            if i != 4:
                ax.axes.get_xaxis().set_visible(False)
                ax.axes.get_yaxis().set_visible(False)
            else:
                ax.set_xlabel('x [pc]')
                ax.set_ylabel('y [pc]')

        titles2 = (r'$N_{\rm H}$', r'$N_{\rm H_2}$', r'$n_{\rm H}$', r'$v_{z}$', r'$T$', r'$G_0$')
        for i, (ax, title) in enumerate(zip(g2, titles2)):
            ax.text(0.5, 0.97, title, **texteffect(fontsize='xx-large'), ha='center', transform=ax.transAxes)
            if i != 0:
                ax.axes.get_xaxis().set_visible(False)
                ax.axes.get_yaxis().set_visible(False)
            else:
                ax.set_xlabel('x [pc]')
                ax.set_ylabel('z [pc]')

        fig.suptitle('t=' + str(int(ds.domain['time'])), x=0.08, y=0.985,
                     ha='center', **texteffect(fontsize='xx-large'))
        plt.subplots_adjust(top=0.97)

        if savefig:
            savdir = osp.join(self.savdir, 'snapshots2')
            if not osp.exists(savdir):
                os.makedirs(savdir)

            savname = osp.join(savdir, '{0:s}_{1:04d}.png'.format(self.basename, num))
            plt.savefig(savname, dpi=200)
            
        return fig
