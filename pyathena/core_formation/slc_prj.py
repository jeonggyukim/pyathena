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
    density=plt.cm.Spectral_r,
)

class SliceProj:

    @staticmethod
    def plt_proj(num, axis='z', field='Sigma_gas',
                 cmap=None, norm=None, vmin=None, vmax=None):
        try:
            vminmax = dict(Sigma_gas=(1e-2,1e2))

            if cmap is None:
                try:
                    cmap = cmap_def[field]
                except KeyError:
                    cmap = plt.cm.viridis
            if vmin is None or vmax is None:
                vmin = vminmax[field][0]
                vmax = vminmax[field][1]

            prj = yt.ProjectionPlot(ds, 'z', ('athena_pp', 'density'))
            prj.set_cmap(("athena_pp", "density"), cmap)
            prj.set_zlim(("athena_pp", "density"), zmin=vmin, zmax=vmax)
            if norm is None or norm == 'log':
                prj.set_log(("athena_pp", "density"), True)
            elif norm == 'linear':
                prj.set_log(("athena_pp", "density"), False)
            return prj
        except KeyError:
            pass
