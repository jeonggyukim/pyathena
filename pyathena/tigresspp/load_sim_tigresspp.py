import os.path as osp
import glob
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

base_path = osp.dirname(__file__)

cpp_to_cc = {
    "rho": "density",
    "press": "pressure",
    "vel1": "velocity1",
    "vel2": "velocity2",
    "vel3": "velocity3",
    "Bcc1": "cell_centered_B1",
    "Bcc2": "cell_centered_B2",
    "Bcc3": "cell_centered_B3",
}

cc_to_cpp = {
    "specific_scalar[0]": "rmetal",
    "specific_scalar[1]": "rSN",
}


class LoadSimTIGRESSClassic(LoadSim, Hst, SliceProj, Fields):
    """LoadSim class for analyzing TIGRESS++ simulations running on Athena-Cversion"""

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

        super(LoadSimTIGRESSClassic, self).__init__(
            basedir, savdir=savdir, load_method=load_method, verbose=verbose
        )
        self.domain = self._get_domain_from_par(self.par)
        self.dfi = DerivedFields(self.par).dfi

        try:
            cooltbl_file = osp.join(base_path, "./cool_ftn.runtime.csv")
            cooltbl = pd.read_csv(cooltbl_file)

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

            pop_synth_file = osp.join(base_path, "./pop_synth.runtime.csv")

            self.pop_synth = pd.read_csv(pop_synth_file)
        except KeyError:
            pass

    def test_newcool(self):
        """Test if NCR is used"""
        try:
            if self.par["configure"]["new_cooling"] == "ON":
                newcool = True
            else:
                newcool = False
        except KeyError:
            newcool = False
        return newcool

    def get_data(self, num, fields=None):
        """
        a warpper function to make data reading easier
        """

        ds = self.load_vtk(num=num)
        if fields is None:
            flist = ds.field_list
        else:
            flist = np.atleast_1d(fields)
        data = ds.get_field(flist)
        rename_dict = {k: v for k, v in cc_to_cpp.items() if k in data}
        data = data.rename(rename_dict)
        data.attrs["time"] = ds.domain["time"]
        return data

    def get_slice(self, num, ax, pos, fields=None):
        """
        a warpper function to make data reading easier
        """

        ds = self.load_vtk(num=num)
        if fields is None:
            flist = ds.field_list
        else:
            flist = np.atleast_1d(fields)
        data = ds.get_slice(ax, flist, pos=pos, method="nearest")
        rename_dict = {k: v for k, v in cc_to_cpp.items() if k in data}
        data = data.rename(rename_dict)
        data.attrs["time"] = ds.domain["time"]
        return data

    @staticmethod
    def get_phase_Tlist(kind="ncr"):
        if kind == "ncr":
            return [500, 6000, 15000, 35000, 5.0e5]
        elif kind == "classic":
            return [200, 5000, 15000, 20000, 5.0e5]

    @staticmethod
    def get_phase_T1list():
        return [500, 6000, 13000, 24000, 1.0e6]


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

        try:
            if self.par["cooling"]["coolftn"] == "tigress":
                # cooltbl_file1 = osp.join(basedir,self.par["cooling"]["coolftn_file"])
                cooltbl_file2 = osp.join(basedir, "cool_ftn.runtime.csv")
                try:
                    cooltbl = pd.read_csv(cooltbl_file2)
                except FileNotFoundError:
                    # cooltbl = pd.read_csv(cooltbl_file1)
                    return
                from scipy.interpolate import interp1d

                logLam = interp1d(
                    cooltbl["logT1"],
                    np.log10(cooltbl["Lambda"]),
                    fill_value="extrapolate",
                )
                logGam = interp1d(
                    cooltbl["logT1"],
                    np.log10(cooltbl["Gamma"]),
                    fill_value="extrapolate",
                )
                mu = interp1d(cooltbl["logT1"], cooltbl["mu"], fill_value="extrapolate")
                self.coolftn = dict(logLambda=logLam, logGamma=logGam, mu=mu)
                self.dfi["T"] = self.temperature_dfi()
                self.dfi["pokCRsinj"] = self.pokCRscalar_inj_dfi()
                self.dfi["pokCRs"] = self.pokCRscalar_dfi()
                self.dfi["pokCR"] = self.pokCR_dfi()


            if self.par["feedback"]["pop_synth"] == "SB99":
                # pop_synth_file1 = osp.join(basedir,self.par["feedback"]["pop_synth_file"])
                pop_synth_file2 = osp.join(basedir, "pop_synth.runtime.csv")
                if osp.isfile(pop_synth_file2):
                    self.pop_synth = pd.read_csv(pop_synth_file2)

            if "configure" in self.par:
                self.nghost = self.par["configure"]["Number_of_ghost_cells"]
        except KeyError:
            pass

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

    def test_newcool(self):
        """Test if NCR is used"""
        try:
            if self.par["configure"]["new_cooling"] == "ON":
                newcool = True
            else:
                newcool = False
        except KeyError:
            newcool = False
        return newcool

    def get_data(self, num, fields=None):
        """
        a warpper function to make data reading easier
        """
        if fields is not None:
            flist = self.get_fields(fields)
        else:
            flist = None
        ds = self.load_hdf5(num=num, quantities=flist)
        rename_dict = {k: v for k, v in cpp_to_cc.items() if k in ds}
        ds = ds.rename(rename_dict)
        ds.attrs["time"] = ds.attrs["Time"]
        self.domain["time"] = ds.attrs["Time"]
        if fields is not None:
            ds = self.update_dfi(ds, fields)
        return ds

    def get_slice(self, num, ax, pos, fields=None):
        """
        a warpper function to make data reading easier
        """
        if fields is not None:
            flist = self.get_fields(fields)
        else:
            flist = None
        ds = self.load_hdf5(num=num, quantities=flist)
        rename_dict = {k: v for k, v in cpp_to_cc.items() if k in ds}
        ds = ds.rename(rename_dict)
        data = ds.sel({ax: pos}, method="nearest")
        data.attrs["time"] = ds.attrs["Time"]
        if fields is not None:
            data = self.update_dfi(data, fields)
        return data

    def get_fields(self, flist):
        dflist = set()
        for f in np.atleast_1d(flist):
            if f in self.dfi:
                dflist |= set(self.dfi[f]["field_dep"])
            else:
                if f in cpp_to_cc:
                    dflist |= set(cpp_to_cc[f])
                else:
                    dflist |= set([f])

        if "velocity" in dflist:
            dflist = set(dflist) - {"velocity"}
            dflist |= {"velocity1", "velocity2", "velocity3"}
        if "cell_centered_B" in dflist:
            dflist = set(dflist) - {"cell_centered_B"}
            dflist |= {"cell_centered_B1", "cell_centered_B2", "cell_centered_B3"}
        flist = [k for k, v in cpp_to_cc.items() if v in list(dflist)]
        return flist

    def update_dfi(self, data, fields):
        loaded_flist = list(data.keys())
        dfilist = set(fields) - set(loaded_flist)
        for f in dfilist:
            if f in self.dfi:
                data[f] = self.dfi[f]["func"](data, self.u)
            else:
                print("{} is not available".format(f))
        return data[fields]

    def load_parcsv(self):
        par_pattern = osp.join(self.basedir, f"{self.problem_id}.par*.csv")
        self.files["parcsv"] = glob.glob(par_pattern)
        self.nums_parcsv = sorted(
            [int(f[f.rfind(".par") + 4 : -4]) for f in self.files["parcsv"]]
        )
        parlist = []
        for i in self.nums_parcsv:
            parname = osp.join(self.basedir, f"TIGRESS.par{i}.csv")
            par = pd.read_csv(parname)
            parlist.append(par)
        return parlist

    @staticmethod
    def get_phase_Tlist(kind="ncr"):
        if kind == "ncr":
            return [500, 6000, 15000, 35000, 5.0e5]
        elif kind == "classic":
            return [200, 5000, 15000, 20000, 5.0e5]

    @staticmethod
    def get_phase_T1list():
        return [500, 6000, 13000, 24000, 1.0e6]


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
