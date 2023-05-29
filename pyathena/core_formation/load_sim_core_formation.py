import os.path as osp
import pandas as pd
import xarray as xr
import numpy as np
import pathlib
import pickle

from pyathena.load_sim import LoadSim
from pyathena.io.timing_reader import TimingReader
from pyathena.core_formation.hst import Hst
from pyathena.core_formation.tools import LognormalPDF
from pyathena.core_formation.tes import TESe
from pyathena.core_formation import tools


class LoadSimCoreFormation(LoadSim, Hst, LognormalPDF, TimingReader):
    """LoadSim class for analyzing core collapse simulations."""

    def __init__(self, basedir_or_Mach=None, savdir=None,
                 load_method='pyathena', verbose=False):
        """The constructor for LoadSimCoreFormation class

        Parameters
        ----------
        basedir_or_Mach : str or float
            Path to the directory where all data is stored;
            Alternatively, Mach number
        savdir : str
            Name of the directory where pickled data and figures will be saved.
            Default value is basedir.
        load_method : str
            Load hdf5 using 'pyathena' or 'yt'. Default value is 'pyathena'.
        verbose : bool or str or int
            Print verbose messages using logger. If True/False, set logger
            level to 'DEBUG'/'WARNING'. If string, it should be one of the
            string representation of python logging package:
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

        if isinstance(basedir_or_Mach, (pathlib.PosixPath, str)):
            basedir = basedir_or_Mach
            super().__init__(basedir, savdir=savdir, load_method=load_method,
                             units=None, verbose=verbose)
            self.Mach = self.par['problem']['Mach']

            LognormalPDF.__init__(self, self.Mach)
            TimingReader.__init__(self, self.basedir, self.problem_id)

            # Set domain
            self.domain = self._get_domain_from_par(self.par)
            Lbox = set(self.domain['Lx'])
            self.dx, self.dy, self.dz = self.domain['dx']
            self.dV = self.dx*self.dy*self.dz
            if len(Lbox) == 1:
                self.Lbox = Lbox.pop()
            else:
                raise ValueError("Box must be cubic")

            self.tcr = 0.5*self.Lbox/self.Mach
            self.sonic_length = tools.get_sonic(self.Mach, self.Lbox)

            # Find the collapse time and corresponding snapshot numbers
            self._find_tcoll()

            try:
                # Load grid-dendro nodes
                self._load_cores()
            except FileNotFoundError:
                pass

            try:
                # Load radial profiles
                self._load_radial_profiles()
            except FileNotFoundError:
                pass

        elif isinstance(basedir_or_Mach, (float, int)):
            self.Mach = basedir_or_Mach
            LognormalPDF.__init__(self, self.Mach)
        elif basedir_or_Mach is None:
            pass
        else:
            raise ValueError("Unknown parameter type for basedir_or_Mach")

    def load_leaves(self, num):
        fname = pathlib.Path(self.basedir, 'GRID',
                             'leaves.{:05d}.p'.format(num))
        with open(fname, 'rb') as handle:
            return pickle.load(handle)

    def load_dendrogram(self, num):
        fname = pathlib.Path(self.basedir, 'GRID',
                             'dendrogram.{:05d}.p'.format(num))
        with open(fname, 'rb') as handle:
            return pickle.load(handle)

    def get_tJeans(self, lmb, rho=None):
        """e-folding time of the fastest growing mode of the Jeans instability
        lmb = wavelength of the mode
        """
        if rho is None:
            rho = self.rho0
        tJeans = 1/np.sqrt(4*np.pi*self.G*rho)*lmb/np.sqrt(lmb**2 - 1)
        return tJeans

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
        tes = TESe(p, xi_s)
        rat, xi0, m = tes.get_crit()
        rhoc = rat*rhoe
        R = LJ_e*xi0
        M = MJ_e*m
        return rhoc, R, M

    def _find_tcoll(self):
        """Read .csv output and find their collapse time and snapshot number.

        Additionally store their mass, position, velocity at the time of
        collapse.
        """
        # find collapse time and the snapshot numbers at the time of collapse
        self.dt_output = {}
        for k, v in self.par.items():
            if k.startswith('output'):
                self.dt_output[v['file_type']] = v['dt']

        x1, x2, x3, v1, v2, v3 = {}, {}, {}, {}, {}, {}
        time, num = {}, {}
        for pid in self.pids:
            phst = self.load_parhst(pid)
            phst0 = phst.iloc[0]
            x1[pid] = phst0.x1
            x2[pid] = phst0.x2
            x3[pid] = phst0.x3
            v1[pid] = phst0.v1
            v2[pid] = phst0.v2
            v3[pid] = phst0.v3
            time[pid] = phst0.time
            num[pid] = np.floor(phst0.time / self.dt_output['hdf5']
                                ).astype('int')
        self.tcoll_cores = pd.DataFrame(dict(x1=x1, x2=x2, x3=x3,
                                             v1=v1, v2=v2, v3=v3,
                                             time=time, num=num),
                                        dtype=object)
        self.tcoll_cores.index.name = 'pid'

    def _load_cores(self, method='veldisp'):
        self.cores = {}
        for pid in self.pids:
            fname = pathlib.Path(self.basedir, 'cores',
                                 'cores.par{}.p'.format(pid))
            self.cores[pid] = pd.read_pickle(fname)
            try:
                fname = pathlib.Path(self.basedir, 'cores',
                                     f'critical_tes_{method}.par{pid}.p')
                tes_crit = pd.read_pickle(fname)
                self.cores[pid] = pd.concat([self.cores[pid], tes_crit], axis=1)
            except FileNotFoundError:
                pass

    def _load_radial_profiles(self):
        self.rprofs = {}
        for pid in self.pids:
            fname = pathlib.Path(self.basedir, 'cores',
                                 'radial_profile.par{}.nc'.format(pid))
            rprf = xr.open_dataset(fname)
            for axis in [1, 2, 3]:
                rprf[f'dvel{axis}_sq_mw'] = (rprf[f'vel{axis}_sq_mw']
                                             - rprf[f'vel{axis}_mw']**2)
                rprf[f'dvel{axis}_sq'] = (rprf[f'vel{axis}_sq']
                                          - rprf[f'vel{axis}']**2)
            rprf = rprf.set_xindex('num')
            self.rprofs[pid] = rprf


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
                msg = "[LoadSimCoreFormationAll]: "\
                      "Model {0:s} doesn\'t exist: {1:s}".format(mdl, basedir)
                print(msg)
            else:
                self.models.append(mdl)
                self.basedirs[mdl] = basedir

    def set_model(self, model, savdir=None, load_method='pyathena',
                  verbose=False):
        self.model = model
        try:
            self.sim = self.simdict[model]
        except KeyError:
            self.sim = LoadSimCoreFormation(self.basedirs[model],
                                            savdir=savdir,
                                            load_method=load_method,
                                            verbose=verbose)
            self.simdict[model] = self.sim

        return self.sim
