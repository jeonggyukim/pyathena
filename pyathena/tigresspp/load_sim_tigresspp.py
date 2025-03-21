import os.path as osp
import glob
import numpy as np
import pandas as pd
import xarray as xr
import warnings

from ..load_sim import LoadSim

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


class LoadSimTIGRESSPP(LoadSim):
    """LoadSim class for analyzing TIGRESS++ simulations running on Athena++"""

    def __init__(self, basedir, savdir=None, load_method="xarray", verbose=False):
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
        # self.domain = self._get_domain_from_par(self.par)
        # self.dfi = DerivedFields(self.par).dfi

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

                # Suppress divide-by-zero warning
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
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
                # self.dfi["T"] = self.temperature_dfi()
                # self.dfi["pokCRsinj"] = self.pokCRscalar_inj_dfi()
                # self.dfi["pokCRs"] = self.pokCRscalar_dfi()
                # self.dfi["pokCR"] = self.pokCR_dfi()

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

    def add_temperature(self, ds):
        T1 = ds["press"] / ds["rho"] * (self.u.temperature_mu).value
        if hasattr(self, "coolftn"):
            logT1 = np.log10(T1)
            mu = self.coolftn["mu"](logT1)
        else:
            mu = 1.27  # default for neutral 1.4/1.1
        ds["temperature"] = mu * T1

    @LoadSim.Decorators.check_netcdf
    def get_slice(
        self,
        num,
        prefix,
        slc_kwargs=dict(z=0, method="nearest"),
        savdir=None,
        force_override=False,
    ):
        """
        a warpper function to make data reading easier
        """
        mb = self.par["meshblock"]
        if self.par[f"output{self.hdf5_outid[0]}"]["ghost_zones"] == "true":
            nghost = self.nghost
        else:
            nghost = 0
        ds = self.load_hdf5(
            num=num,
            num_ghost=nghost,
            chunks=dict(x=mb["nx1"], y=mb["nx2"], z=mb["nx3"]),
        )
        if "press" in ds:
            self.add_temperature(ds)
        # rename the variables to match athena convention so that we can use
        # the same derived fields as in athena
        rename_dict = {k: v for k, v in cpp_to_cc.items() if k in ds}
        ds = ds.rename(rename_dict)
        slc = ds.sel(**slc_kwargs)
        slc.attrs = dict(time=ds.attrs["Time"])
        return slc

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

    @LoadSim.Decorators.check_netcdf
    def load_zprof(self, prefix="merged_zprof", savdir=None, force_override=False):
        dlist = dict()
        for fname in self.files["zprof"]:
            with open(fname, "r") as f:
                header = f.readline()
            data = pd.read_csv(fname, skiprows=1)
            data.index = data.x3v
            time = eval(
                header[header.find("time") : header.find("cycle")]
                .split("=")[-1]
                .strip()
            )
            phase = (
                header[header.find("phase") : header.find("variable")]
                .split("=")[-1]
                .strip()
            )
            for ph in self.phase:
                if ph in fname:
                    if phase not in dlist:
                        dlist[phase] = []
                    dlist[phase].append(
                        data.to_xarray().assign_coords(time=time).rename(x3v="z")
                    )

        dset = []
        for phase in dlist:
            dset.append(xr.concat(dlist[phase], dim="time").assign_coords(phase=phase))
        dset = xr.concat(dset, dim="phase")

        return dset

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

    def set_model(self, model, savdir=None, load_method="xarray", verbose=False):
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
