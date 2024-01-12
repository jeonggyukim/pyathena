import os
import os.path as osp
import pandas as pd
import numpy as np

from ..load_sim import LoadSim
from ..util.units import Units
from .pdf import PDF
from .hist2d import Hist2d
from .h2 import H2
from .hst import Hst
from .zprof import Zprof
from .slc_prj import SliceProj
from .starpar import StarPar
from .snapshot_HIH2EM import Snapshot_HIH2EM
from .profile_1d import Profile1D

class LoadSimTIGRESSNCR(LoadSim, Hst, Zprof, SliceProj,
                        StarPar, PDF, Hist2d, H2, Profile1D, Snapshot_HIH2EM):
    """LoadSim class for analyzing TIGRESS-RT simulations.
    """

    def __init__(self, basedir, savdir=None, load_method='pyathena',
                 muH = 1.4271, verbose=False):
        """The constructor for LoadSimTIGRESSNCR class

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

        super(LoadSimTIGRESSNCR,self).__init__(basedir, savdir=savdir,
                                               load_method=load_method, verbose=verbose)

        # Set unit and domain
        try:
            muH = self.par['problem']['muH']
        except KeyError:
            pass
        self.muH = muH
        self.u = Units(muH=muH)
        self.domain = self._get_domain_from_par(self.par)

    def test_newcool(self):
        try:
            if self.par['configure']['new_cooling'] == 'ON':
                newcool = True
            else:
                newcool = False
        except KeyError:
            newcool = False
        return newcool

    def calc_deltay(self, time):
        """
        Function to calculate the y-offset at radial edges of the domain

        Parameters
        ----------
        time : float
            simulation time in code unit

        Returns
        -------
        Delta y = (q*Omega0*Lx*t) mod (Ly)
        """

        par = self.par
        domain = self.domain
        u = self.u

        # Compute Delta y = (q*Omega0*Lx*t) mod (Ly)
        qshear = par['problem']['qshear']
        Omega0 = par['problem']['Omega']*(u.kms*u.pc)
        deltay = np.fmod(qshear*Omega0*domain['Lx'][0]*time, domain['Lx'][1])

        return deltay


class LoadSimTIGRESSNCRAll(object):
    """Class to load multiple simulations"""
    def __init__(self, models=None, muH=None):

        # Default models
        if models is None:
            models = dict()
        if muH is None:
            muH = dict()
            for mdl in models:
                muH[mdl] = 1.4271
        self.models = []
        self.basedirs = dict()
        self.muH = dict()
        self.simdict = dict()

        for mdl, basedir in models.items():
            if not osp.exists(basedir):
                print('[LoadSimTIGRESSNCRAll]: Model {0:s} doesn\'t exist: {1:s}'.format(
                    mdl,basedir))
            else:
                self.models.append(mdl)
                self.basedirs[mdl] = basedir
                if mdl in muH:
                    self.muH[mdl] = muH[mdl]
                else:
                    print('[LoadSimTIGRESSNCRAll]: muH for {0:s} has to be set'.format(
                          mdl))

    def set_model(self, model, savdir=None, load_method='pyathena', verbose=False):
        self.model = model
        try:
            self.sim = self.simdict[model]
        except KeyError:
            self.sim = LoadSimTIGRESSNCR(self.basedirs[model], savdir=savdir,
                                         muH=self.muH[model],
                                         load_method=load_method, verbose=verbose)
            self.simdict[model] = self.sim

        return self.sim
