import os
import os.path as osp
import pathlib
import pandas as pd
import numpy as np
import xarray as xr

from ..load_sim import LoadSim
from ..util.units import Units
from .hst import Hst
from .slc_prj import SliceProj
from pyathena.tigress_gc import config, tools

class LoadSimTIGRESSGC(LoadSim, Hst, SliceProj):
    """LoadSim class for analyzing TIGRESS-GC simulations.
    """

    def __init__(self, basedir, savdir=None, load_method='pyathena', verbose=False):
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
        self.domain = self._get_domain_from_par(self.par)
        self.dx, self.dy, self.dz = self.domain['dx']
        try:
            rprof = xr.open_dataset('{}/radial_profile.nc'.format(self.basedir), engine='netcdf4')
            Rring = self.par['problem']['Rring']
            rprof.coords['eta'] = np.sqrt((rprof.R - Rring)**2 + rprof.z**2)
            eta0 = 200
            for ax in (1,2,3):
                rprof['B'+str(ax)] *= u.muG
                rprof['B_squared'+str(ax)] *= u.muG**2
                rprof['mass_flux'+str(ax)] *= u.Msun/u.pc**2/u.Myr/1e6 # Msun / pc^2 / yr
            rprof['mdot_in'] *= u.Msun/u.Myr/1e6     # Msun / yr
            rprof['mdot_in_mid'] *= u.Msun/u.Myr/1e6
            rprof['mdot_Reynolds'] *= u.Msun/u.Myr/1e6
            rprof['mdot_Maxwell'] *= u.Msun/u.Myr/1e6
            rprof['mdot_dLdt'] *= u.Msun/u.Myr/1e6
            rprof['mdot_coriolis'] *= u.Msun/u.Myr/1e6
            rprof['surface_density'] *= u.Msun / u.pc**2
            rprof['pressure'] *= u.pok
            rprof['turbulent_pressure'] *= u.pok
            rprof['Breg1'] = (rprof.B1.where(rprof.eta<eta0)).weighted(rprof.density).mean(dim=['R','z'])
            rprof['Btrb1'] = np.sqrt(rprof.B_squared1 - rprof.B1**2).where(rprof.eta<eta0).weighted(rprof.density).mean(dim=['R','z'])
            rprof['Breg2'] = (rprof.B2.where(rprof.eta<eta0)).weighted(rprof.density).mean(dim=['R','z'])
            rprof['Btrb2'] = np.sqrt(rprof.B_squared2 - rprof.B2**2).where(rprof.eta<eta0).weighted(rprof.density).mean(dim=['R','z'])
            rprof['Breg3'] = (rprof.B3.where(rprof.eta<eta0)).weighted(rprof.density).mean(dim=['R','z'])
            rprof['Btrb3'] = np.sqrt(rprof.B_squared3 - rprof.B3**2).where(rprof.eta<eta0).weighted(rprof.density).mean(dim=['R','z'])
            rprof['Breg'] = np.sqrt(rprof.Breg1**2 + rprof.Breg2**2 + rprof.Breg3**2)
            rprof['Btrb'] = np.sqrt(rprof.Btrb1**2 + rprof.Btrb2**2 + rprof.Btrb3**2)
            self.rprof = rprof
        except:
            self.rprof = None

    @LoadSim.Decorators.check_pickle
    def load_prfm(self, prefix='prfm_quantities', savdir=None,
                  force_override=False):
        """Load prfm quantities

        Parameters
        ----------
        num : int
            Snapshot number.
        """
        prfm = []
        for num in self.nums[config.NUM_START:]:
            fname = pathlib.Path(self.basedir, 'prfm_quantities',
                                 'prfm.{:04}.nc'.format(num))
            time = self.load_vtk(num).domain['time']*self.u.Myr
            ds = xr.open_dataset(fname, engine='netcdf4').expand_dims(dict(t=[time]))
            prfm.append(ds)
        prfm = xr.concat(prfm, 't')
        tools.add_derived_fields(self, prfm, 'R')
        prfm = prfm.where(prfm.R < config.Rmax[self.basename.split('_')[0]], other=np.nan)
        return prfm


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


    def set_model(self, model, savdir=None, load_method='pyathena', verbose=False):
        self.model = model
        try:
            self.sim = self.simdict[model]
        except KeyError:
            self.sim = LoadSimTIGRESSGC(self.basedirs[model], savdir=savdir,
                                        load_method=load_method, verbose=verbose)
            self.simdict[model] = self.sim

        return self.sim
