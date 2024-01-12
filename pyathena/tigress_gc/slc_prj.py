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
import xarray as xr

from ..load_sim import LoadSim
from ..io.read_starpar_vtk import read_starpar_vtk
from ..plt_tools.cmap_shift import cmap_shift
from ..plt_tools.plt_starpar import scatter_sp
from ..classic.utils import texteffect

cmap_def = dict(
    Sigma_gas=plt.cm.pink_r,
    nH=plt.cm.Spectral_r,
    T=cmap_shift(mpl.cm.RdYlBu_r, midpoint=3./7.),
    vz=plt.cm.bwr,
    Bmag=plt.cm.cividis,
)

norm_def = dict(
    Sigma_gas=LogNorm(1e-2,1e2),
    nH=LogNorm(1e-4,1e3),
    T=LogNorm(1e1,1e7),
    vz=Normalize(-200,200),
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

        fields_def = ['nH', 'vz', 'T', 'cs', 'vx', 'vy', 'vz', 'pok']
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
                res[zlab][f] = dat[f].data

        return res

    @LoadSim.Decorators.check_pickle
    def read_prj(self, num, axes=['x', 'y', 'z'], prefix='prj',
                 savdir=None, force_override=False, id0=True):

        axtoi = dict(x=0, y=1, z=2)
        fields = ['nH']
        axes = np.atleast_1d(axes)

        ds = self.load_vtk(num=num, id0=id0)
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

        return res

    def read_slc_xarray(self, num, axis='zall', force_override=False):
        slc = self.read_slc(num, force_override=force_override)
        if axis == 'zall':
            slc_dset = slc_get_all_z(slc)
        else:
            slc_dset = slc_to_xarray(slc, axis)
        return slc_dset

    @staticmethod
    def plt_slice(ax, slc, axis='z', field='density', cmap=None, norm=None):
        try:
            if cmap is None:
                cmap = cmap_def[field]

            if norm is None:
                norm = mpl.colors.LogNorm()
            elif norm == 'linear':
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

            if norm is None or norm == 'log':
                norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
            elif norm == 'linear':
                norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

            ax.imshow(prj[axis][field], cmap=cmap, extent=prj['extent'][axis],
                      norm=norm, origin='lower', interpolation='none')
        except KeyError:
            pass

    def plt_snapshot(self, num,
                     fields_xy=('Sigma_gas', 'nH', 'T'),
                     fields_xz=('Sigma_gas', 'nH', 'T', 'vz', 'Bmag'),
                     norm_factor=5.0, agemax=20.0, agemax_sn=40.0, runaway=False,
                     suptitle=None, savdir_pkl=None, savdir=None, force_override=False,
                     figsize=(26,12),
                     savefig=True):
        """Plot 12-panel projection, slice plots in the z and y directions

        Parameters
        ----------
        num : int
            vtk snapshot number
        fields_xy: list of str
            Field names for z projections and slices
        fields_xz: list of str
            Field names for y projections and slices
        norm_factor : float
            Normalization factor for starpar size. Smaller norm_factor for bigger size.
        agemax : float
            Maximum age of radiation source particles [Myr]
        agemax_sn : float
            Maximum age of sn particles [Myr]
        runaway : bool
            If True, show runaway star particles
        suptitle : str
            Suptitle for snapshot
        savdir_pkl : str
            Path to which save (from which load) projections and slices
        savdir : str
            Path to which save (from which load) projections and slices
        """

        label = dict(Sigma_gas=r'$\Sigma$',
                     nH=r'$n_{\rm H}$',
                     T=r'$T$',
                     vz=r'$v_z$',
                     Bmag=r'$|B|$'
        )

        kind = dict(Sigma_gas='prj', nH='slc', T='slc', vz='slc', Bmag='slc')
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


def slc_to_xarray(slc,axis='z'):
    dset = xr.Dataset()
    for f in slc[axis].keys():
        x0,x1,y0,y1=slc['extent'][axis[0]]

        Ny,Nx=slc[axis][f].shape

        xfc = np.linspace(x0,x1,Nx+1)
        yfc = np.linspace(y0,y1,Ny+1)
        xcc = 0.5*(xfc[1:] + xfc[:-1])
        ycc = 0.5*(yfc[1:] + yfc[:-1])

        dims = dict(z=['y','x'],x=['z','y'],y=['z','x'])

        dset[f] = xr.DataArray(slc[axis][f],coords=[ycc,xcc],dims=dims[axis[0]])
    return dset

def slc_get_all_z(slc):
    dlist = []
    for k in slc.keys():
        if k.startswith('z'):
            slc_dset = slc_to_xarray(slc,k)
            if len(k) == 1:
                z0=0.
            elif k[1] == 'n':
                z0 = float(k[2:])*(-100)
            elif k[1] == 'p':
                z0 = float(k[2:])*(100)
            else:
                raise KeyError
            slc_dset = slc_dset.assign_coords(z=z0)
            dlist.append(slc_dset)
        else:
            pass
    return xr.concat(dlist,dim='z').sortby('z')
