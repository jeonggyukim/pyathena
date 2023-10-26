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
from pyathena.core_formation.slc_prj import SliceProj
from pyathena.core_formation.tools import LognormalPDF
from pyathena.core_formation import tools


class LoadSimCoreFormation(LoadSim, Hst, SliceProj, LognormalPDF,
                           TimingReader):
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

    def __init__(self, basedir_or_Mach=None, savdir=None,
                 load_method='pyathena', verbose=False, force_override=False):
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
        self.gconst = np.pi
        self.tff0 = tools.tfreefall(self.rho0, self.gconst)

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
            self.tcoll_cores = self._load_tcoll_cores(force_override=force_override)

            try:
                # Load grid-dendro nodes
                self.cores = self._load_cores(force_override=force_override)
            except FileNotFoundError:
                logging.warning("Failed to load core information")
                pass

            try:
                # Load radial profiles
                self.rprofs = self._load_radial_profiles(force_override=force_override)
            except (AttributeError, FileNotFoundError, KeyError):
                logging.warning("Failed to load radial profiles")
                pass

            if hasattr(self, "cores") and hasattr(self, "rprofs"):
                self.cores = self.update_core_props(force_override=force_override)

        elif isinstance(basedir_or_Mach, (float, int)):
            self.Mach = basedir_or_Mach
            LognormalPDF.__init__(self, self.Mach)
        elif basedir_or_Mach is None:
            pass
        else:
            raise ValueError("Unknown parameter type for basedir_or_Mach")

    def load_dendro(self, num):
        """Load pickled dendrogram object

        Parameters
        ----------
        num : int
            Snapshot number.
        """
        fname = pathlib.Path(self.savdir, 'GRID',
                             'dendrogram.{:05d}.p'.format(num))
        with open(fname, 'rb') as handle:
            return pickle.load(handle)

    def good_cores(self):
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
        good_cores = []
        for pid in self.pids:
            if self.cores[pid].attrs['isolated'] and self.cores[pid].attrs['resolved']:
                good_cores.append(pid)
        return good_cores

    def resolved_cores(self):
        resolved_cores = []
        for pid in self.pids:
            if self.cores[pid].attrs['resolved']:
                resolved_cores.append(pid)
        return resolved_cores

    def isolated_cores(self):
        isolated_cores = []
        for pid in self.pids:
            if self.cores[pid].attrs['isolated']:
                isolated_cores.append(pid)
        return isolated_cores

    @LoadSim.Decorators.check_pickle
    def update_core_props(self, ncells_min=8, prefix='cores',
                          savdir=None, force_override=False):
        """Update core properties

        Calculate lagrangian core properties using the radial profiles
        Set isolated and resolved flags
        Add normalized times

        Parameters
        ----------
        ncells_min : int, optional
            Minimum number of cells to be considered 'resolved'

        Returns
        -------
        pandas.DataFrame
            Updated core dataframe.
        """
        for pid in self.pids:
            cores = self.cores[pid]  # shallow copy
            if 'critical_radius' not in cores:
                continue
            rprofs = self.rprofs[pid]
            attrs = cores.attrs.copy()
            lprops = tools.calculate_lagrangian_props(self, cores, rprofs)
            attrs.update(lprops.attrs)
            cores = cores.join(lprops)
            cores.attrs = attrs

            # Workaround to use pid as an argument in the function calls below
            self.cores[pid] = cores

            # Test resolvedness and isolatedness
            if tools.test_isolated_core(self, pid):
                cores.attrs['isolated'] = True
            else:
                cores.attrs['isolated'] = False
            if tools.test_resolved_core(self, pid, ncells_min):
                cores.attrs['resolved'] = True
            else:
                cores.attrs['resolved'] = False

            cores.insert(1, 'tnorm1',
                         (cores.time - cores.attrs['tcoll'])
                          / cores.attrs['tff_crit'])
            cores.insert(2, 'tnorm2',
                         (cores.time - cores.attrs['tcrit'])
                          / (cores.attrs['tcoll'] - cores.attrs['tcrit']))
            cores.attrs['dt_build'] = cores.attrs['tcrit'] - cores.iloc[0].time
            cores.attrs['dt_coll'] = cores.attrs['tcoll'] - cores.attrs['tcrit']
            mcore = cores.attrs['mcore_crit']
            phst = self.load_parhst(pid)
            idx = phst.mass.sub(mcore).abs().argmin()
            if np.isnan(mcore) or idx == phst.index[-1]:
                tf = np.nan
            else:
                tf = phst.loc[idx].time
            cores.attrs['tinfall_end'] = tf
            cores.attrs['dt_infall'] = tf - cores.attrs['tcoll']

        return self.cores

    @LoadSim.Decorators.check_pickle
    def _load_tcoll_cores(self, prefix='tcoll_cores', savdir=None, force_override=False):
        """Read .csv output and find their collapse time and snapshot number.

        Additionally store their mass, position, velocity at the time of
        collapse.
        """
        # find collapse time and the snapshot numbers at the time of collapse
        dt_output = {}
        for k, v in self.par.items():
            if k.startswith('output'):
                dt_output[v['file_type']] = v['dt']

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
            num[pid] = np.floor(phst0.time / dt_output['hdf5']
                                ).astype('int')
        tcoll_cores = pd.DataFrame(dict(x1=x1, x2=x2, x3=x3,
                                             v1=v1, v2=v2, v3=v3,
                                             time=time, num=num),
                                        dtype=object)
        tcoll_cores.index.name = 'pid'
        return tcoll_cores

    @LoadSim.Decorators.check_pickle
    def _load_cores(self, prefix='cores', savdir=None, force_override=False):
        cores = {}
        pids_tes_not_found = []
        for pid in self.pids:
            fname = pathlib.Path(self.savdir, 'cores', 'cores.par{}.p'.format(pid))
            core = pd.read_pickle(fname).sort_index()

            # Remove duplicated nums that formed sink earlier.
            for pid_prev in range(1, pid):
                cc = cores[pid_prev].iloc[-1]
                num = cc.name
                lid = cc.leaf_id
                if num in core.index[:-1] and core.loc[num].leaf_id == lid:
                    core = core.loc[num+1:]

            # Read critical TES info and concatenate to self.cores
            try:
                # Try reading critical TES pickles
                tes_crit = []
                for num in core.index:
                    fname = pathlib.Path(self.savdir, 'critical_tes',
                                         'critical_tes.par{}.{:05d}.p'
                                         .format(pid, num))
                    tes_crit.append(pd.read_pickle(fname))
                tes_crit = pd.DataFrame(tes_crit).set_index('num').sort_index()
                core = core.join(tes_crit)

            except FileNotFoundError:
                pids_tes_not_found.append(pid)
                pass

            # Set attributes
            core.attrs['pid'] = pid

            # Assign to attribute
            cores[pid] = core

        if len(pids_tes_not_found) > 0:
            logging.warning("Cannot find critical TES information for pid: {}.".format(pids_tes_not_found))
        return cores

    @LoadSim.Decorators.check_pickle
    def _load_radial_profiles(self, prefix='radial_profile', savdir=None, force_override=False):
        """
        Raises
        ------
        FileNotFoundError
            If individual radial profiles are not found
        KeyError
            If `cores` has not been initialized (due to missing files, etc.)
        """
        rprofs = {}
        for pid in self.pids:
            core = self.cores[pid]
            if len(core) == 0:
                rprf = None
            else:
                rprf = []
                for num in core.index:
                    findv = pathlib.Path(self.savdir, 'radial_profile',
                                          'radial_profile.par{}.{:05d}.nc'
                                          .format(pid, num))
                    rprf.append(xr.open_dataset(findv))
                rprf = xr.concat(rprf, 't')
                rprf = rprf.assign_coords(dict(num=('t', core.index)))
                for axis in [1, 2, 3]:
                    rprf[f'dvel{axis}_sq_mw'] = (rprf[f'vel{axis}_sq_mw']
                                                 - rprf[f'vel{axis}_mw']**2)
                rprf['menc'] = (4*np.pi*rprf.r**2*rprf.rho).cumulative_integrate('r')
                rprf = rprf.merge(tools.calculate_accelerations(rprf))
                rprf = rprf.set_xindex('num')
            rprofs[pid] = rprf
        return rprofs


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

    def set_model(self, model, savdir=None,
                  load_method='pyathena', verbose=False,
                  reset=False, force_override=False):
        self.model = model
        if reset or force_override:
            self.sim = LoadSimCoreFormation(self.basedirs[model],
                                            savdir=savdir,
                                            load_method=load_method,
                                            verbose=verbose,
                                            force_override=force_override)
            self.simdict[model] = self.sim
        else:
            try:
                self.sim = self.simdict[model]
            except KeyError:
                self.sim = LoadSimCoreFormation(self.basedirs[model],
                                                savdir=savdir,
                                                load_method=load_method,
                                                verbose=verbose,
                                                force_override=force_override)
                self.simdict[model] = self.sim

        return self.sim
