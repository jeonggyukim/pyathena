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

from .rad_and_pionized import RadiationAndPartiallyIonized

from .phase_set import Phase, PhaseSet
from .phase_set import create_phase_set_with_LyC_mask
from .phase_set import create_phase_set_with_density_bins
from .zprof_from_vtk import ZprofFromVTK

from .rad_hst import RadiationHst
from .rad_source import RadiationSource
from .rad_slice import RadiationSlice

class LoadSimTIGRESSNCR(LoadSim, Hst, Zprof, ZprofFromVTK, SliceProj, StarPar, PDF,
                        Hist2d, H2, Profile1D, Snapshot_HIH2EM,
                        RadiationAndPartiallyIonized, RadiationHst, RadiationSource,
                        RadiationSlice):
    """LoadSim class for analyzing TIGRESS-NCR simulations.
    """
    def __init__(self, basedir, savdir=None, load_method='pyathena',
                 muH=1.4271, verbose=False):
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
        self.phase_set = self.get_phase_sets()
        if self.test_newcool():
            self.test_newcool_params()

    def test_newcool(self):
        try:
            if self.par["configure"]["new_cooling"] == "ON":
                newcool = True
            else:
                newcool = False
        except KeyError:
            newcool = False
        return newcool

    def test_spiralarm(self):
        try:
            if self.par["configure"]["SpiralArm"] == "yes":
                arm = True
            else:
                arm = False
        except KeyError:
            arm = False
        return arm

    def test_newcool_params(self):
        s = self
        try:
            s.iCoolH2colldiss = s.par["cooling"]["iCoolH2colldiss"]
        except KeyError:
            s.iCoolH2colldiss = 0

        try:
            s.iCoolH2rovib = s.par["cooling"]["iCoolH2rovib"]
        except KeyError:
            s.iCoolH2rovib = 0

        try:
            s.ikgr_H2 = s.par["cooling"]["ikgr_H2"]
        except KeyError:
            s.ikgr_H2 = 0

        # s.config_time = pd.to_datetime(s.par["configure"]["config_date"])
        # if "PDT" in s.par["configure"]["config_date"]:
        #     s.config_time = s.config_time.tz_localize("US/Pacific")
        if s.config_time < pd.to_datetime("2021-06-30 20:29:36 -04:00"):
            s.iCoolHIcollion = 0
        else:
            s.iCoolHIcollion = 1

        # check this is run with corrected CR heating
        # 85a7857bb7c797686a4e9630cba71f326e1097cd
        if s.config_time < pd.to_datetime("2022-05-23 22:23:43 -04:00"):
            s.oldCRheating = 1
        else:
            s.oldCRheating = 0

        try:
            s.iH2heating = s.par["cooling"]["iH2heating"]
        except KeyError:
            s.iH2heating = -1

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

    def get_output_nums(self, tMyr_range=[250,450], out_fmt='vtk', verbose=False):
        """
        Returns an output number array corresponding to time [range] in Myr.
        No guarantee that the snapshots in the requested time range exist.

        Parameters
        ----------
        tMyr_range : float or two floats
            time range in Myr
        out_fmt : str
            Output format (vtk, starpar_vtk, zprof, rst)
        verbose : bool
            Print summary of results. Default is False.

        Returns
        -------
        nums : list of int
            Snapshot numbers
        """
        u = self.u
        par = self.par
        for k in [k for k in par.keys() if 'output' in k]:
            if par[k]['out_fmt'] == out_fmt:
                dt_out = par[k]['dt']

        num_range_real = np.atleast_1d(tMyr_range)/u.Myr/dt_out
        num_range = [int(np.round(num)) for num in num_range_real]
        nums = [num for num in range(num_range[0], num_range[-1]+1)]
        if verbose:
            print('Model: ', self.basename)
            print('Output format: ', out_fmt)
            print('tMyr range: ', tMyr_range)
            print('num range: ', num_range)
            for num in sorted(set((nums[0], nums[-1]))):
                if out_fmt == 'vtk':
                    try:
                        ds = self.load_vtk(num)
                        print('num time time_Myr',
                              num, ds.domain['time'], ds.domain['time']*u.Myr)
                    except Exception as e:
                        print(e)
                        print('Error..some snapshots may not exist.')
                        continue
                else:
                    # Implementation of starpar_vtk, zprof..
                    continue

        if len(nums) == 1:
            return nums[0]
        else:
            return nums

    def get_phase_sets(self):
        f1 = lambda dd, v, op, c: op(dd[v], c)
        f2 = lambda dd, v, op, c, v2: op(dd[v], c*dd[v2])
        f3 = lambda dd, v, op, c, v2: op(dd[v], c+dd[v2])

        # Conditions used to select a phase in each PhaseSet need to be mutually exclusive
        # PhaseSet need not be all inclusive (e.g., can select warm gas only and subdivide)

        # Default used in NCR radiation paper
        phs0 = PhaseSet('default_rad',
                        [Phase('CpU', 1, [[f1, 'T', np.less, 6e3]]),
                         Phase('WIM', 2, [[f1, 'xHII', np.greater_equal, 0.5],
                                          [f1, 'T', np.greater_equal, 6e3],
                                          [f1, 'T', np.less, 3.5e4]]),
                         Phase('WNM', 3, [[f1, 'xHII', np.less, 0.5],
                                          [f1, 'T', np.greater_equal, 6e3],
                                          [f1, 'T', np.less, 3.5e4]]),
                         Phase('hot', 4, [[f1, 'T', np.greater_equal, 3.5e4]])]
                        )

        # Used to characterize warm gas in terms of xHII, xHII_eq
        # Note that temperature boundary for warm is lower than that used by default
        phs1_ = PhaseSet('warm_eq',
                         [Phase('w1_eq', 1, [[f1, 'T', np.greater_equal, 3e3],
                                             [f1, 'T', np.less, 1.5e4],
                                             [f2, 'xHII', np.less, 2.0, 'xHII_eq']]),
                          Phase('w2_eq', 2, [[f1, 'T', np.greater_equal, 1.5e4],
                                             [f1, 'T', np.less, 3.5e4],
                                             [f2, 'xHII', np.less, 2.0, 'xHII_eq']]),
                          Phase('w1_geq', 3, [[f1, 'T', np.greater_equal, 3e3],
                                              [f1, 'T', np.less, 1.5e4],
                                              [f2, 'xHII', np.greater_equal, 2.0, 'xHII_eq']]),
                          Phase('w2_geq', 4, [[f1, 'T', np.greater_equal, 1.5e4],
                                              [f1, 'T', np.less, 3.5e4],
                                              [f2, 'xHII', np.greater_equal, 2.0, 'xHII_eq']]),
                          Phase('CpU', 5, [[f1, 'T', np.less, 3e3]]),
                          Phase('hot', 6, [[f1, 'T', np.greater_equal, 3.5e4]])]
                         )

        # Used to characterize partially ionized gas & gas exposed to LyC radiation
        phs1 = create_phase_set_with_LyC_mask(phs1_, flag_phot=[1, 0, 1, 0, 0, 0])

        return {phs.name: phs for phs in (phs0, phs1)}

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
