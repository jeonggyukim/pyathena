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

cmap_def = dict(
    Sigma_gas=plt.cm.pink_r,
)

norm_def = dict(
    Sigma_gas=(1e-1, 2e2),
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
    def read_prj(self, num, axes=['x', 'y', 'z'], prefix='prj',
                 savdir=None, force_override=False):
        axtoi = dict(x=0, y=1, z=2)
        fields = ['dens', 'mom1', 'mom2', 'mom3']
        axes = np.atleast_1d(axes)

        ds = self.load_hdf5(num=num, quantities=fields)

        res = dict()
        res['extent'] = self._get_extent(self.domain)

        for ax in axes:
            i = axtoi[ax]
            dx = self.domain['dx'][i]
            mom_los = ds[f'mom{i+1}']

            res[ax] = dict()
            dcol = (ds.dens*dx).sum(ax)
            res[ax]['Sigma_gas'] = dcol


            for ncrit in [50, 100]:
                d = ds.where((ds.dens > ncrit), other=np.nan)
                dens = ds.dens.copy(deep=False)
                vel = d[f'mom{i+1}']/dens
                vel_los = vel.weighted(dens).mean(ax)
                vdisp_los = np.sqrt((vel**2).weighted(dens).mean(ax) - vel_los**2)

                res[ax][f'vel_nc{ncrit}'] = vel_los
                res[ax][f'veldisp_nc{ncrit}'] = vdisp_los

        return res

    @staticmethod
    def plt_proj(prj, axis='z', field='Sigma_gas', ax=None,
                 cmap=None, norm=None, vmin=None, vmax=None):
        try:
            if cmap is None:
                try:
                    cmap = cmap_def[field]
                except KeyError:
                    cmap = plt.cm.viridis
            if vmin is None or vmax is None:
                vmin = norm_def[field][0]
                vmax = norm_def[field][1]

            if norm is None or norm == 'log':
                norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
            elif norm == 'linear':
                norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

            if ax is not None:
                plt.sca(ax)
            plt.imshow(prj[axis][field], cmap=cmap, extent=prj['extent'][axis],
                      norm=norm, origin='lower', interpolation='none')
        except KeyError:
            pass
