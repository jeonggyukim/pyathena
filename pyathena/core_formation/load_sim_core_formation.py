import os.path as osp
import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path
import pickle
import logging
from scipy.interpolate import interp1d

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

        if isinstance(basedir_or_Mach, (Path, str)):
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

            try:
                self.cores = self.update_core_props(force_override=force_override)
            except (AttributeError, KeyError):
                logging.warning("Failed to update core properties")

        elif isinstance(basedir_or_Mach, (float, int)):
            self.Mach = basedir_or_Mach
            LognormalPDF.__init__(self, self.Mach)
        elif basedir_or_Mach is None:
            pass
        else:
            raise ValueError("Unknown parameter type for basedir_or_Mach")

    def load_dendro(self, num, pruned=True):
        """Load pickled dendrogram object

        Parameters
        ----------
        num : int
            Snapshot number.
        pruned : bool
            If true, load the pruned dendrogram
        """
        if pruned:
            fname = Path(self.savdir, 'GRID',
                         'dendrogram.pruned.{:05d}.p'.format(num))
        else:
            fname = Path(self.savdir, 'GRID',
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
            cores = self.cores[pid]
            if not cores.attrs['tcoll_resolved']:
                continue
            if cores.attrs['isolated'] and cores.attrs['resolved']:
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

            # Find critical time
            ncrit = tools.critical_time(self, pid)
            cores.attrs['numcrit'] = ncrit

            # Test resolvedness and isolatedness
            if tools.test_isolated_core(self, cores):
                cores.attrs['isolated'] = True
            else:
                cores.attrs['isolated'] = False
            if tools.test_resolved_core(self, cores, ncells_min):
                cores.attrs['resolved'] = True
            else:
                cores.attrs['resolved'] = False

            if not (cores.attrs['resolved'] and cores.attrs['isolated']):
                continue
            # Lines below are executed only for resolved and isolated cores.

            rprofs = self.rprofs[pid]
            lprops = tools.calculate_lagrangian_props(self, cores, rprofs)
            if set(lprops.columns).issubset(cores.columns):
                msg = ("Lagrangian core properties are already included in "
                       "cores attributes, even before computing them. "
                       "The pickle might be currupted.")
                raise ValueError(msg)

            # Save attributes before performing join, which will drop them.
            attrs = cores.attrs.copy()
            attrs.update(lprops.attrs)
            cores = cores.join(lprops)
            # Reattach attributes
            cores.attrs = attrs

            # Calculate normalized times
            cores.insert(1, 'tnorm1',
                         (cores.time - cores.attrs['tcoll'])
                          / cores.attrs['tff_crit'])
            cores.insert(2, 'tnorm2',
                         (cores.time - cores.attrs['tcrit'])
                          / (cores.attrs['tcoll'] - cores.attrs['tcrit']))

            mcore = cores.attrs['mcore']
            rcore = cores.attrs['rcore']
            ncoll = cores.attrs['numcoll']

            # Building time
            if np.isnan(ncrit):
                cores.attrs['dt_build'] = np.nan
            else:
                rprf = rprofs.sel(num=ncrit)
                mdot = (-4*np.pi*rcore**2*rprf.rho*rprf.vel1_mw).interp(r=rcore).data[()]
                cores.attrs['dt_build'] = mcore / mdot

            # Collapse time
            cores.attrs['dt_coll'] = cores.attrs['tcoll'] - cores.attrs['tcrit']

            # Infall time
            phst = self.load_parhst(pid)
            idx = phst.mass.sub(mcore).abs().argmin()
            if np.isnan(mcore) or idx == phst.index[-1]:
                tf = np.nan
            else:
                tf = phst.loc[idx].time
            cores.attrs['tinfall_end'] = tf
            cores.attrs['dt_infall'] = tf - cores.attrs['tcoll']

            # Velocity dispersion at t_crit
            if np.isnan(ncrit):
                sigma_r = np.nan
            else:
                sigma_r = cores.loc[ncrit].sigma_mw
            cores.attrs['sigma_r'] = sigma_r

            # Free-fall time at t_coll
            cores.attrs['tff_coll'] = tools.tfreefall(cores.loc[ncoll].mean_density, self.gconst)

            # Calculate some observable properties
            r_obs, rhoavg_obs, sigma_obs = [], [], []
            prestellar_cores = cores.loc[:cores.attrs['numcoll']]
            for num, core in prestellar_cores.iterrows():
                rprf = rprofs.sel(num=num)
                res = tools.calculate_observables(core, rprf, rprf.r.max())
                r_obs.append(res['r_obs'])
                rhoavg_obs.append(res['rhoavg_obs'])
                sigma_obs.append(res['sigma_obs'])
            obsprops = pd.DataFrame(dict(r_obs=r_obs,
                                         rhoavg_obs=rhoavg_obs,
                                         sigma_obs=sigma_obs),
                                    index=prestellar_cores.index)
            # Save attributes before performing join, which will drop them.
            attrs = cores.attrs.copy()
            attrs.update(obsprops.attrs)
            cores = cores.join(obsprops)
            # Reattach attributes
            cores.attrs = attrs

            # Sort attributes
            cores.attrs = {k: cores.attrs[k] for k in sorted(cores.attrs)}

            self.cores[pid] = cores

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
        cores_dict = {}
        pids_not_found = []

        # Try reading the go15 mass
        try:
            fname = Path(self.savdir) / 'mcore_go15.p'
            with open(fname, 'rb') as f:
                mcore_go15 = pickle.load(f)
            mcore_go15_found = True
        except FileNotFoundError:
            mcore_go15_found = False


        for pid in self.pids:
            fname = Path(self.savdir, 'cores', 'cores.par{}.p'.format(pid))
            cores = pd.read_pickle(fname).sort_index()

            if cores.attrs['tcoll_resolved']:
                # Read critical TES info and concatenate to self.cores

                # Try reading critical TES pickles
                tes_crit = []
                for num in cores.index:
                    try:
                        fname = Path(self.savdir, 'critical_tes',
                                     'critical_tes.par{}.{:05d}.p'
                                     .format(pid, num))
                        tes_crit.append(pd.read_pickle(fname))
                    except FileNotFoundError:
                        pids_not_found.append(pid)
                        break
                if len(tes_crit) > 0:
                    tes_crit = pd.DataFrame(tes_crit).set_index('num').sort_index()

                    # Save attributes before performing join, which will drop them.
                    attrs = cores.attrs.copy()
                    attrs.update(tes_crit.attrs)
                    cores = cores.join(tes_crit)

                    # Reattach attributes
                    cores.attrs = attrs
                if mcore_go15_found:
                    cores.attrs['mcore_go15'] = mcore_go15[pid]

            # Sort attributes
            cores.attrs = {k: cores.attrs[k] for k in sorted(cores.attrs)}

            cores_dict[pid] = cores

        if len(pids_not_found) > 0:
            msg = f"Some critical TES files are missing for pid {pids_not_found}"
            logging.warning(msg)
        return cores_dict

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
        rprofs_dict = {}
        pids_not_found = []
        for pid in self.pids:
            cores = self.cores[pid]
            if not cores.attrs['tcoll_resolved']:
                rprofs = None
            else:
                rprofs, nums = [], []
                min_nr = None
                for num in cores.index:
                    try:
                        fname = Path(self.savdir, 'radial_profile',
                                     'radial_profile.par{}.{:05d}.nc'
                                     .format(pid, num))
                        rprf = xr.open_dataset(fname)
                        if min_nr is None:
                            min_nr = rprf.sizes['r']
                        else:
                            min_nr = min(min_nr, rprf.sizes['r'])
                        rprofs.append(rprf)
                        nums.append(num)
                    except FileNotFoundError:
                        pids_not_found.append(pid)
                        break
                if len(rprofs) > 0:
                    rprofs = xr.concat(rprofs, 't')
                    rprofs = rprofs.assign_coords(dict(num=('t', nums)))
                    # Slice data to common range in r.
                    rprofs = rprofs.isel(r=slice(0, min_nr))
                    for axis in [1, 2, 3]:
                        rprofs[f'dvel{axis}_sq_mw'] = (rprofs[f'vel{axis}_sq_mw']
                                                     - rprofs[f'vel{axis}_mw']**2)
                    rprofs['menc'] = (4*np.pi*rprofs.r**2*rprofs.rho
                                      ).cumulative_integrate('r')
                    rprofs = rprofs.merge(tools.calculate_accelerations(rprofs))
                    rprofs = rprofs.set_xindex('num')
            rprofs_dict[pid] = rprofs
        if len(pids_not_found) > 0:
            msg = f"Some radial profiles are missing for pid {pids_not_found}."
            logging.warning(msg)
        return rprofs_dict


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
