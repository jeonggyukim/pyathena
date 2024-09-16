import os
import os.path as osp
from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr
import pickle

from ..load_sim import LoadSim
from ..util.units import Units
from .hst import Hst
from .slc_prj import SliceProj
from pyathena.tigress_gc import config, tools

class LoadSimTIGRESSGC(LoadSim, Hst, SliceProj):
    """LoadSim class for analyzing TIGRESS-GC simulations.
    """

    def __init__(self, basedir, savdir=None, load_method='pyathena',
                 verbose=False, force_override=False):
        """The constructor for LoadSimTIGRESSGC class

        Parameters
        ----------
        basedir : str
            Name of the directory where all data is stored
        savdir : str
            Name of the directory where pickled data and figures will be saved.
            Default value is basedir.
        load_method : str
            Load vtk using 'pyathena' or 'yt'. Default value is 'pyathena'.
            If None, savdir=basedir. Default value is None.
        verbose : bool or str or int
            Print verbose messages using logger. If True/False, set logger
            level to 'DEBUG'/'WARNING'. If string, it should be one of the string
            representation of python logging package:
            ('NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
            Numerical values from 0 ('NOTSET') to 50 ('CRITICAL') are also
            accepted.
        """

        super(LoadSimTIGRESSGC,self).__init__(basedir, savdir=savdir,
                                               load_method=load_method, verbose=verbose)

        # Set unit and domain
        try:
            muH = self.par['problem']['muH']
        except KeyError:
            pass
        self.muH = config.muH
        u = Units(muH=self.muH)
        self.u = u
        self.dx, self.dy, self.dz = self.domain['dx']

        try:
            savdir = Path(self.savdir, 'GRID')
            self.nodes = self._load_nodes(savdir=savdir, force_override=force_override)
        except FileNotFoundError:
            self.logger.warning("Failed to load node information")
            pass

    def load_dendro(self, num):
        """Load pickled dendrogram object

        Parameters
        ----------
        num : int
            Snapshot number.
        """
        fname = Path(self.savdir, 'GRID',
                     'dendrogram.{:04d}.p'.format(num))

        with open(fname, 'rb') as handle:
            return pickle.load(handle)

    @LoadSim.Decorators.check_pickle
    def _load_nodes(self, prefix='nodes', savdir=None, force_override=False):
        """Load pickled linewidth-sized rel."""

        res = []
        for num in self.nums[config.NUM_START:]:
            fname = Path(self.savdir, 'linewidth_size',
                         'grid_dendro.{:04d}.p'.format(num))
            with open(fname, 'rb') as handle:
                res.append(pickle.load(handle))
        if len(res) > 0:
            res = pd.DataFrame(res).set_index('num').sort_index()
        return res

    @LoadSim.Decorators.check_pickle
    def load_prfm(self, prefix='prfm_quantities', savdir=None,
                  force_override=False):
        """Load prfm quantities"""
        prfm = []
        for num in self.nums[config.NUM_START:]:
            fname = Path(self.basedir, 'prfm_quantities',
                        'prfm.{:04}.nc'.format(num))
            time = self.load_vtk(num).domain['time']*self.u.Myr
            ds = xr.open_dataset(fname, engine='netcdf4').expand_dims(dict(t=[time]))
            prfm.append(ds)
        prfm = xr.concat(prfm, 't')
        tools.add_derived_fields(self, prfm, 'R')
        prfm = prfm.where(prfm.R < config.Rmax[self.basename.split('_')[0]], other=np.nan)
        return prfm

    def load_radial_profiles(self):
        """Load radial profiles"""
        rprofs = []
        for num in self.nums:
            fname = Path(self.basedir, 'azimuthal_averages_warmcold',
                         'gc_azimuthal_average.{:04}.nc'.format(num))
            time = self.load_vtk(num).domain['time']*self.u.Myr
            ds = xr.open_dataset(fname, engine='netcdf4').expand_dims(dict(t=[time]))
            rprofs.append(ds)
        rprofs = xr.concat(rprofs, 't')
        return rprofs

class LoadSimTIGRESSGCAll(object):
    """Class to load multiple simulations"""
    def __init__(self, models=None):

        # Default models
        if models is None:
            models = dict()
        self.models = []
        self.basedirs = dict()
        self.simdict = dict()

        for mdl, basedir in models.items():
            if not osp.exists(basedir):
                print('[LoadSimTIGRESSGCAll]: Model {0:s} doesn\'t exist: {1:s}'.format(
                    mdl,basedir))
            else:
                self.models.append(mdl)
                self.basedirs[mdl] = basedir


    def set_model(self, model, savdir=None, load_method='pyathena',
                  verbose=False, reset=False, force_override=False):
        self.model = model
        if reset or force_override:
            self.sim = LoadSimTIGRESSGC(self.basedirs[model], savdir=savdir,
                                        load_method=load_method, verbose=verbose,
                                        force_override=force_override)
            self.simdict[model] = self.sim
        else:
            try:
                self.sim = self.simdict[model]
            except KeyError:
                self.sim = LoadSimTIGRESSGC(self.basedirs[model], savdir=savdir,
                                            load_method=load_method, verbose=verbose,
                                            force_override=force_override)
                self.simdict[model] = self.sim

        return self.sim
