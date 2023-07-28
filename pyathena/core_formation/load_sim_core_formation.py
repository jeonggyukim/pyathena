import os.path as osp
import pandas as pd
import xarray as xr
import numpy as np
import pathlib
import pickle
import logging

from pyathena.load_sim import LoadSim
from pyathena.util.units import Units
from pyathena.io.timing_reader import TimingReader
from pyathena.core_formation.hst import Hst
from pyathena.core_formation.tools import LognormalPDF
from pyathena.core_formation import tools


class LoadSimCoreFormation(LoadSim, Hst, LognormalPDF, TimingReader):
    """LoadSim class for analyzing core collapse simulations.

    Attributes
    ----------
    rho0 : float
        Mean density of the cloud in the code unit.
    cs : float
        Sound speed in the code unit.
    gconst : float
        Gravitational constant in the code unit.
    tff : float
        Free fall time in the code unit.
    tcr : float
        Half-box flow crossing time in the code unit.
    Mach : float
        Mach number.
    sonic_length : float
        Sonic length in the code unit.
    basedir : str
        Base directory
    problem_id : str
        Prefix of the Athena++ problem
    dx : float
        Uniform cell spacing in x direction.
    dy : float
        Uniform cell spacing in y direction.
    dz : float
        Uniform cell spacing in z direction.
    tcoll_cores : pandas DataFrame
        t_coll core information container.
    cores : dict of pandas DataFrame
        All preimages of t_coll cores.
    """

    def __init__(self, basedir_or_Mach=None, savdir=None, use_phitot=True,
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
        use_phitot : bool
            Gas-only potential or total potential for core definition.
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
        self.gconst = np.pi
        self.tff0 = tools.tfreefall(self.rho0, self.gconst)

        # configurations
        self.use_phitot = use_phitot

        if isinstance(basedir_or_Mach, (pathlib.PosixPath, str)):
            basedir = basedir_or_Mach
            super().__init__(basedir, savdir=savdir, load_method=load_method,
                             units=Units('code'), verbose=verbose)
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
            self._load_tcoll_cores()

            try:
                # Load grid-dendro nodes
                self._load_cores()
            except FileNotFoundError:
                logging.warning("Failed to load core information")
                pass

            try:
                # Load radial profiles
                self._load_radial_profiles()
            except (FileNotFoundError, KeyError):
                logging.warning("Failed to load radial profiles")
                pass

        elif isinstance(basedir_or_Mach, (float, int)):
            self.Mach = basedir_or_Mach
            LognormalPDF.__init__(self, self.Mach)
        elif basedir_or_Mach is None:
            pass
        else:
            raise ValueError("Unknown parameter type for basedir_or_Mach")

    def load_dendrogram(self, num):
        """Load pickled dendrogram object

        Parameters
        ----------
        num : int
            Snapshot number.
        """
        which = 'phitot' if self.use_phitot else 'phigas'
        fname = pathlib.Path(self.basedir, 'GRID',
                             'dendrogram.{}.{:05d}.p'.format(which, num))
        with open(fname, 'rb') as handle:
            return pickle.load(handle)

    def find_good_cores(self, ncells_min=10, ftff=0.5):
        """Examine the isolatedness and resolvedness of cores

        This function will examine whether the cores are isolated or
        resolved and assign attributes to the `cores`.

        Parameters
        ----------
        ncells_min : int, optional
            Minimum number of cells to be considered "resolved".
        ftff : float, optional
            fractional free fall time before t_coll, at which the
            resolvedness is examined.
        """
        self.good_cores = []
        for pid in self.pids:
            if tools.test_isolated_core(self, pid):
                self.cores[pid].attrs['isolated'] = True
            else:
                self.cores[pid].attrs['isolated'] = False
            if tools.test_resolved_core(self, pid, ncells_min, f=ftff):
                self.cores[pid].attrs['resolved'] = True
            else:
                self.cores[pid].attrs['resolved'] = False
            if (self.cores[pid].attrs['isolated'] and self.cores[pid].attrs['resolved']):
                self.good_cores.append(pid)

    def _load_tcoll_cores(self):
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
        dirname_cores = 'cores_phitot' if self.use_phitot else 'cores'
        dirname_tes = 'critical_tes_phitot' if self.use_phitot else 'critical_tes'
        for pid in self.pids:
            fname = pathlib.Path(self.basedir, dirname_cores,
                                 'cores.par{}.p'.format(pid))
            core = pd.read_pickle(fname)

            # Read corrected tidal radius and mass
            fname = pathlib.Path(self.basedir, dirname_cores,
                                 'rtidal_correction.par{}.p'.format(pid))
            try:
                core = pd.concat([core, pd.read_pickle(fname)], axis=1, join='inner')
                # Calculate derived quantities
                core['mean_density'] = (core.envelop_tidal_mass
                                        / (4*np.pi*core.envelop_tidal_radius**3/3))
            except FileNotFoundError:
                logging.warning("Failed to read envelop tidal radius")
                pass

            # Filter out true preimages by applying distance criterion
            if 'envelop_tidal_radius' in core:
                core = tools.apply_preimage_correction(self, core)
            else:
                logging.warning("Cannot perform preimage correction before"
                                " finding envelop tidal radii")

            # Assign to attribute
            self.cores[pid] = core.sort_index()

            # Read critical TES info and concatenate to self.cores
            try:
                # Try reading critical TES pickles
                tes_crit = []
                for num in core.index:
                    fname = pathlib.Path(self.basedir, dirname_tes,
                                          'critical_tes_{}.par{}.{:05d}.p'
                                          .format(method, pid, num))
                    tes_crit.append(pd.read_pickle(fname))
                tes_crit = pd.DataFrame(tes_crit).set_index('num')
                tes_crit = tes_crit.sort_index()
                self.cores[pid] = pd.concat([self.cores[pid], tes_crit],
                                            axis=1, join='inner').sort_index()
            except FileNotFoundError:
                logging.warning("Cannot find critical TES information.")
                pass

    def _load_radial_profiles(self):
        """
        Raises
        ------
        FileNotFoundError
            If individual radial profiles are not found
        KeyError
            If `cores` has not been initialized (due to missing files, etc.)
        """
        self.rprofs = {}
        dirname = 'radial_profile_phitot' if self.use_phitot else 'radial_profile'
        for pid in self.pids:
            try:
                # Try reading joined radial profile
                fname = pathlib.Path(self.basedir, dirname,
                                     'radial_profile.par{}.nc'.format(pid))
                rprf = xr.open_dataset(fname)
            except FileNotFoundError:
                # Read individual radial profiles and write joined file.
                core = self.cores[pid]
                rprf = []
                for num in core.index:
                    fname2 = pathlib.Path(self.basedir, dirname,
                                          'radial_profile.par{}.{:05d}.nc'
                                          .format(pid, num))
                    rprf.append(xr.open_dataset(fname2))
                rprf = xr.concat(rprf, 't')
                rprf = rprf.assign_coords(dict(num=('t', core.index)))
                rprf.to_netcdf(fname)
            for axis in [1, 2, 3]:
                rprf[f'dvel{axis}_sq_mw'] = (rprf[f'vel{axis}_sq_mw']
                                             - rprf[f'vel{axis}_mw']**2)
                rprf[f'dvel{axis}_sq'] = (rprf[f'vel{axis}_sq']
                                          - rprf[f'vel{axis}']**2)
            rprf = rprf.merge(tools.get_accelerations(rprf))
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

    def set_model(self, model, savdir=None, use_phitot=True,
                  load_method='pyathena', verbose=False,
                  reset=False):
        self.model = model
        if reset:
            self.sim = LoadSimCoreFormation(self.basedirs[model],
                                            savdir=savdir,
                                            use_phitot=use_phitot,
                                            load_method=load_method,
                                            verbose=verbose)
            self.simdict[model] = self.sim
        else:
            try:
                self.sim = self.simdict[model]
            except KeyError:
                self.sim = LoadSimCoreFormation(self.basedirs[model],
                                                savdir=savdir,
                                                use_phitot=use_phitot,
                                                load_method=load_method,
                                                verbose=verbose)
                self.simdict[model] = self.sim

        return self.sim
