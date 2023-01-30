import os
import os.path as osp
import pandas as pd
import numpy as np

from ..load_sim import LoadSim
from .hst import Hst
from .slc_prj import SliceProj
from .tools import Tools
from .pdf import LognormalPDF

class LoadSimCoreFormation(LoadSim, Hst, SliceProj, Tools, LognormalPDF):
    """LoadSim class for analyzing core collapse simulations."""

    def __init__(self, basedir=None, savdir=None, load_method='yt', verbose=False):
        """The constructor for LoadSimCoreFormation class

        Parameters
        ----------
        basedir : str
            Name of the directory where all data is stored
        savdir : str
            Name of the directory where pickled data and figures will be saved.
            Default value is basedir.
        load_method : str
            Load hdf5 using 'pyathena' or 'yt'. Default value is 'yt'.
        verbose : bool or str or int
            Print verbose messages using logger. If True/False, set logger
            level to 'DEBUG'/'WARNING'. If string, it should be one of the string
            representation of python logging package:
            ('NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
            Numerical values from 0 ('NOTSET') to 50 ('CRITICAL') are also
            accepted.
        """

        if basedir is not None:
            super().__init__(basedir, savdir=savdir, load_method=load_method,
                             verbose=verbose)
            # Set domain
            self.domain = self._get_domain_from_par(self.par)
            LognormalPDF.__init__(self, self.par['problem']['Mach'])

        # Set unit system
        # [L] = L_{J,0}, [M] = M_{J,0}, [V] = c_s
        self.rho0 = 1.0
        self.cs = 1.0
        self.G = np.pi
        self.tff = np.sqrt(3/32)

class LoadSimCoreFormationAll(object):
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
                print('[LoadSimCoreFormationAll]: Model {0:s} doesn\'t exist: {1:s}'.format(
                    mdl,basedir))
            else:
                self.models.append(mdl)
                self.basedirs[mdl] = basedir

    def set_model(self, model, savdir=None, load_method='yt', verbose=False):
        self.model = model
        try:
            self.sim = self.simdict[model]
        except KeyError:
            self.sim = LoadSimCoreFormation(self.basedirs[model], savdir=savdir,
                                           load_method=load_method, verbose=verbose)
            self.simdict[model] = self.sim

        return self.sim
