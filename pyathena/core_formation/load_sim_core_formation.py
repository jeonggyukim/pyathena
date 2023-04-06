import os.path as osp
import pandas as pd
import numpy as np
from pathlib import Path
import pickle

from pyathena.load_sim import LoadSim
from pyathena.io.timing_reader import TimingReader
from pyathena.core_formation.hst import Hst
from pyathena.core_formation.slc_prj import SliceProj
from pyathena.core_formation.pdf import LognormalPDF
from pyathena.core_formation.tes import TES

class LoadSimCoreFormation(LoadSim, Hst, SliceProj, LognormalPDF, TimingReader):
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

        # Set unit system
        # [L] = L_{J,0}, [M] = M_{J,0}, [V] = c_s
        self.rho0 = 1.0
        self.cs = 1.0
        self.G = np.pi
        self.tff = np.sqrt(3/32)

        if basedir is not None:
            super().__init__(basedir, savdir=savdir, load_method=load_method,
                             units=None, verbose=verbose)
            LognormalPDF.__init__(self, self.par['problem']['Mach'])
            TimingReader.__init__(self, self.basedir, self.problem_id)

            # Set domain
            self.domain = self._get_domain_from_par(self.par)

            # find collapse time and the snapshot numbers at the time of collapse
            self.dt_output = {}
            for k, v in self.par.items():
                if k.startswith('output'):
                    self.dt_output[v['file_type']] = v['dt']
            self.tcolls, self.nums_tcoll = {}, {}
            # mass and position of sink particle at the time of creation
            self.mp0, self.xp0, self.yp0, self.zp0 = {}, {}, {}, {}
            self.vpx0, self.vpy0, self.vpz0 = {}, {}, {}
            for pid in self.pids:
                phst = self.load_parhst(pid)
                phst0 = phst.iloc[0]
                tcoll = phst0.time
                self.mp0[pid] = phst0.mass
                self.xp0[pid] = phst0.x1
                self.yp0[pid] = phst0.x2
                self.zp0[pid] = phst0.x3
                self.vpx0[pid] = phst0.v1
                self.vpy0[pid] = phst0.v2
                self.vpz0[pid] = phst0.v3
                self.tcolls[pid] = tcoll
                self.nums_tcoll[pid] = np.floor(tcoll / self.dt_output['hdf5']).astype('int')


    def load_fiso_dicts(self, num):
        fname = Path(self.basedir, 'fiso.{:05d}.p'.format(num))
        with open(fname, 'rb') as handle:
            self.fiso_dicts = pickle.load(handle)
        return self.fiso_dicts


    def load_tcoll_cores(self):
        fname = Path(self.basedir, 'tcoll_cores.p')
        with open(fname, 'rb') as handle:
            self.tcoll_cores = pickle.load(handle)
        return self.tcoll_cores


    def get_tJeans(self, lmb, rho=None):
        """e-folding time of the fastest growing mode of the Jeans instability
        lmb = wavelength of the mode
        """
        if rho is None:
            rho = self.rho0
        tJeans = 1/np.sqrt(4*np.pi*self.G*rho)*lmb/np.sqrt(lmb**2 - 1)
        return tJeans

    def get_tcr(self, lscale, dv):
        """crossing time for a length scale lscale and velocity dv"""
        tcr = lscale/dv
        return tcr

    def get_Lbox(self, Mach):
        """Return box size at which t_cr = t_Jeans,
        where t_cr = (Lbox/2)/Mach
        """
        dv = Mach*self.cs
        Lbox = np.sqrt(1 + dv**2/(np.pi*self.G*self.rho0))
        return Lbox

    def get_sonic(self, Mach, p=0.5):
        """returns sonic scale for periodic box with Mach number Mach
        assume linewidth-size relation v ~ R^p
        """
        if Mach==0:
            return np.inf
        Lbox = self.get_Lbox(Mach)
        lambda_s = Lbox*Mach**(-1/p)
        return lambda_s

    def get_RLP(self, M):
        """Returns the LP radius enclosing  mass M"""
        RLP = self.G*M/8.86/self.cs**2
        return RLP

    def get_rhoLP(self, r):
        """Larson-Penston asymptotic solution in dimensionless units"""
        rhoLP = 8.86*self.cs**2/(4*np.pi*self.G*r**2)
        return rhoLP

    def get_critical_TES(self, rhoe, lmb_sonic, p=0.5):
        """
        Calculate critical turbulent equilibrium sphere

        Description
        -----------
        Critical mass of turbulent equilibrium sphere is given by
            M_crit = M_{J,e}m_crit(xi_s)
        where m_crit is the dimensionless critical mass and M_{J,e}
        is the Jeans mass at the edge density rho_e.
        This function assumes unit system:
            [L] = L_{J,0}, [M] = M_{J,0}

        Parameters
        ----------
        rhoe : edge density
        lmb_sonic : sonic radius
        p (optional) : power law index for the linewidth-size relation

        Returns
        -------
        rhoc : central density
        R : radius of the critical TES
        M : mass of the critical TES
        """
        LJ_e = 1.0*(rhoe/self.rho0)**-0.5
        MJ_e = 1.0*(rhoe/self.rho0)**-0.5
        xi_s = lmb_sonic / LJ_e
        tes = TES(p, xi_s)
        rat, xi0, m = tes.get_crit()
        rhoc = rat*rhoe
        R = LJ_e*xi0
        M = MJ_e*m
        return rhoc, R, M

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
