import os.path as osp
import numpy as np
import pandas as pd

from .hst import Hst

# from .zprof import Zprof
from .slc_prj import SliceProj
from .fields import Fields
from .timing import Timing

# from .starpar import StarPar
from ..load_sim import LoadSim
from ..fields.fields import DerivedFields


class LoadSimTIGRESSPP(LoadSim, Hst, SliceProj, Fields, Timing):
    """LoadSim class for analyzing TIGRESS++ simulations running on Athena++"""

    def __init__(self, basedir, savdir=None, load_method="pyathena", verbose=False):
        """The constructor for LoadSimTIGRESSPP class

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

        super(LoadSimTIGRESSPP, self).__init__(
            basedir, savdir=savdir, load_method=load_method, verbose=verbose
        )
        self.domain = self._get_domain_from_par(self.par)
        self.dfi = DerivedFields(self.par).dfi

        if self.par["cooling"]["coolftn"] == "tigress":
            # cooltbl_file1 = osp.join(basedir,self.par["cooling"]["coolftn_file"])
            cooltbl_file2 = osp.join(basedir, "cool_ftn.runtime.csv")
            cooltbl = pd.read_csv(cooltbl_file2)
            from scipy.interpolate import interp1d

            logLam = interp1d(
                cooltbl["logT1"], np.log10(cooltbl["Lambda"]), fill_value="extrapolate"
            )
            logGam = interp1d(
                cooltbl["logT1"], np.log10(cooltbl["Gamma"]), fill_value="extrapolate"
            )
            mu = interp1d(cooltbl["logT1"], cooltbl["mu"], fill_value="extrapolate")
            self.coolftn = dict(logLambda=logLam, logGamma=logGam, mu=mu)
            self.dfi["T"] = self.temperature_dfi()

        if self.par["feedback"]["pop_synth"] == "KO17":
            # pop_synth_file1 = osp.join(basedir,self.par["feedback"]["pop_synth_file"])
            pop_synth_file2 = osp.join(basedir, "pop_synth.runtime.csv")
            self.pop_synth = pd.read_csv(pop_synth_file2)

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
        qshear = par["orbital_advection"]["qshear"]
        Omega0 = par["orbital_advection"]["Omega0"] * (u.kms * u.pc)
        deltay = np.fmod(qshear * Omega0 * domain["Lx"][0] * time, domain["Lx"][1])

        return deltay


class LoadSimTIGRESSPPAll(object):
    """Class to load multiple simulations"""

    def __init__(self, models=None, muH=None):
        # Default models
        if models is None:
            models = dict()
        self.models = []
        self.basedirs = dict()
        self.simdict = dict()

        for mdl, basedir in models.items():
            if not osp.exists(basedir):
                print(
                    "[LoadSimTIGRESSPPAll]: Model {0:s} doesn't exist: {1:s}".format(
                        mdl, basedir
                    )
                )
            else:
                self.models.append(mdl)
                self.basedirs[mdl] = basedir

    def set_model(self, model, savdir=None, load_method="pyathena", verbose=False):
        self.model = model
        try:
            self.sim = self.simdict[model]
        except KeyError:
            self.sim = LoadSimTIGRESSPP(
                self.basedirs[model],
                savdir=savdir,
                load_method=load_method,
                verbose=verbose,
            )
            self.simdict[model] = self.sim

        return self.sim
