import os.path as osp
import glob
import numpy as np
import pandas as pd
import xarray as xr
import warnings
from matplotlib.colors import Normalize, LogNorm, SymLogNorm
from matplotlib import cm
import cmasher as cmr

import astropy.units as au

from .hst import Hst
from .timing import Timing
from .zprof import Zprof
from .slc_prj import SliceProj
from .pdf import PDF
from .prostproc_zprof import PostProcessingZprof
from ..load_sim import LoadSim
from pyathena.fields.fields import DerivedFields
import pyathena as pa

base_path = osp.dirname(__file__)

class LoadSimTIGRESSPP(LoadSim,Hst,Timing,Zprof,SliceProj,PDF,PostProcessingZprof):
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

        # set configure options boolean
        self.check_configure_options()

        # set cooling function
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
                    logL=np.log10(cooltbl["Lambda"])
                    logG=np.log10(cooltbl["Gamma"])
                    logLam = interp1d(
                        cooltbl["logT1"],
                        logL,
                        fill_value=(logL[0],"extrapolate"),
                    )
                    logGam = interp1d(
                        cooltbl["logT1"],
                        logG,
                        fill_value=("extrapolate",logG[0]),
                    )
                mu = interp1d(cooltbl["logT1"], cooltbl["mu"], fill_value="extrapolate")
                self.coolftn = dict(logLambda=logLam, logGamma=logGam, mu=mu)

            if self.par["feedback"]["pop_synth"] == "SB99":
                # pop_synth_file1 = osp.join(basedir,self.par["feedback"]["pop_synth_file"])
                pop_synth_file2 = osp.join(basedir, "pop_synth.runtime.csv")
                if osp.isfile(pop_synth_file2):
                    self.pop_synth = pd.read_csv(pop_synth_file2)

            if "configure" in self.par:
                self.nghost = self.par["configure"]["Number_of_ghost_cells"]

            self.shear = self.par["mesh"]["ix1_bc"] == "shear_periodic"
            if self.shear:
                self.qshear = self.par["orbital_advection"]["qshear"]
                self.Omega0 = self.par["orbital_advection"]["Omega0"]
        except KeyError:
            pass

        # set external gravity field
        try:
            if self.par["problem"]["ext_grav"] in ["force", "potential"]:
                extgrav_file = osp.join(basedir, "extgrav.runtime.csv")
                if osp.isfile(extgrav_file):
                    self.extgrav = pd.read_csv(extgrav_file)
        except KeyError:
            pass

        # update dfi
        self.update_derived_fields()

        # set variable name conversion
        self.cpp_to_cc = {
            "rho": "density",
            "press": "pressure",
            "vel1": "velocity1",
            "vel2": "velocity2",
            "vel3": "velocity3",
            "Bcc1": "cell_centered_B1",
            "Bcc2": "cell_centered_B2",
            "Bcc3": "cell_centered_B3",
            "rHI": "xHI",
            "rH2": "xH2",
            "rEL": "xe",
        }

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
        if self.options["newcool"]:
            mu = 1.4/(1.1 + ds['rEL'] - ds['rH2'])
        else:
            if hasattr(self, "coolftn"):
                logT1 = np.log10(T1)
                mu = self.coolftn["mu"](logT1)
            else:
                mu = 1.27  # default for neutral 1.4/1.1
        ds["temperature"] = mu * T1

    def add_coolheat(self, ds):
        T1 = ds["press"] / ds["rho"] * (self.u.temperature_mu).value
        density_to_nH = (self.u.density.cgs/(self.u.mH*self.u.muH/au.cm**3))
        nH = ds["rho"] * density_to_nH
        if hasattr(self, "coolftn"):
            logT1 = np.log10(T1)
            cool = 10**self.coolftn["logLambda"](logT1)
            heat = 10**self.coolftn["logGamma"](logT1)
        ds["cool_rate"] = cool*nH**2
        ds["heat_rate"] = heat*nH

    def get_data(self, num, outid=None, load_derived=False):
        """a wrapper function to load data with automatically assigned
        num_ghost and chunks"""

        mb = self.par["meshblock"]
        if self.par[f"output{self.hdf5_outid[0]}"]["ghost_zones"] == "true":
            nghost = self.nghost
        else:
            nghost = 0
        ds = self.load_hdf5(
            num=num,
            num_ghost=nghost,
            outid=outid,
            chunks=dict(x=mb["nx1"], y=mb["nx2"], z=mb["nx3"]),
        )

        if "press" in ds:
            self.add_temperature(ds)

        if load_derived:
            rename_dict = {k: v for k, v in self.cpp_to_cc.items() if k in ds}
            ds = ds.rename(rename_dict)
            for f in self.dfi:
                try:
                    ds[f] = self.dfi[f]["func"](ds, self.u)
                except KeyError:
                    continue
        return ds

    def load_parcsv(self):
        par_pattern = osp.join(self.basedir, f"{self.problem_id}.par*.csv")
        self.files["parcsv"] = glob.glob(par_pattern)
        self.nums_parcsv = sorted(
            [int(f[f.rfind(".par") + 4 : -4]) for f in self.files["parcsv"]]
        )
        parlist = []
        for i in self.nums_parcsv:
            parname = osp.join(self.basedir, f"{self.problem_id}.par{i}.csv")
            par = pd.read_csv(parname)
            parlist.append(par)
        return parlist

    def update_derived_fields(self):
        dfi = DerivedFields(self.par)
        # dfi.dfi["T"]["imshow_args"]["cmap"] = "Spectral_r"
        # dfi.dfi["T"]["imshow_args"]["norm"] = LogNorm(vmin=1e2, vmax=1e8)
        dfi.dfi["nH"]["imshow_args"]["cmap"] = cmr.rainforest
        dfi.dfi["nH"]["imshow_args"]["norm"] = LogNorm(vmin=1e-5, vmax=1e2)
        dfi.dfi["pok"]["imshow_args"]["norm"] = LogNorm(vmin=1e1, vmax=1e7)
        dfi.dfi["vz"]["imshow_args"]["norm"] = Normalize(vmin=-200, vmax=200)
        if self.options["newcool"]:
            dfi.dfi["nHI"]["imshow_args"]["cmap"] = cmr.rainforest
            dfi.dfi["nHI"]["imshow_args"]["norm"] = LogNorm(vmin=1e-4, vmax=1e2)
            dfi.dfi["nHII"]["imshow_args"]["cmap"] = cmr.rainforest
            dfi.dfi["nHII"]["imshow_args"]["norm"] = LogNorm(vmin=1e-4, vmax=1e2)
            dfi.dfi["ne"]["imshow_args"]["cmap"] = cmr.rainforest
            dfi.dfi["ne"]["imshow_args"]["norm"] = LogNorm(vmin=1e-4, vmax=1e2)
            dfi.dfi["xe"]["imshow_args"]["norm"] = Normalize(0, 1.2)
            dfi.dfi["xe"]["imshow_args"]["cmap"] = cm.ocean_r
            dfi.dfi["cool_rate_cgs"]["imshow_args"]["cmap"] = cmr.get_sub_cmap(cmr.freeze_r, 0.0, 0.7)
            dfi.dfi["cool_rate_cgs"]["imshow_args"]["norm"] = LogNorm(vmin=1e-30, vmax=1e-20)
            dfi.dfi["heat_rate_cgs"]["imshow_args"]["cmap"] = cmr.get_sub_cmap(cmr.flamingo_r, 0.0, 0.7)
            dfi.dfi["heat_rate_cgs"]["imshow_args"]["norm"] = LogNorm(vmin=1e-30, vmax=1e-20)

        if self.options["cosmic_ray"]:
            dfi.dfi["vmag"]["imshow_args"]["cmap"] = dfi.dfi["Vcr_mag"]["imshow_args"][
                "cmap"
            ]
            dfi.dfi["vmag"]["imshow_args"]["norm"] = dfi.dfi["Vcr_mag"]["imshow_args"][
                "norm"
            ]
            dfi.dfi["pok_cr"]["imshow_args"]["norm"] = LogNorm(1.e2,5.e4)
            for pok_field in ["pok","pok_mag","pok_trbz","pok_cr"]:
                # cmap = dfi.dfi["pok_cr"]["imshow_args"]["cmap"]
                cmap = cm.plasma
                dfi.dfi[pok_field]["imshow_args"]["cmap"] = cmap
                norm = dfi.dfi["pok_cr"]["imshow_args"]["norm"]
                dfi.dfi[pok_field]["imshow_args"]["norm"] = norm
        for k in dfi.dfi:
            sp = dfi.dfi[k]["label"].split(r"\;")
            if len(sp) == 2:
                name = sp[0]
                unit = sp[1]
                dfi.dfi[k]["label_name"]= name+"$"
                dfi.dfi[k]["label_unit"]= "$" + unit
            else:
                dfi.dfi[k]["label_name"]= dfi.dfi[k]["label"]
                dfi.dfi[k]["label_unit"]= ""
        self.dfi = dfi.dfi

    def get_pdf(
        self,
        dchunk,
        xf,
        yf,
        wf,
        xlim,
        ylim,
        Nx=128,
        Ny=128,
        logx=False,
        logy=False,
        phase=None,
    ):
        try:
            xdata = dchunk[xf]
        except KeyError:
            xdata = self.dfi(dchunk, self.u)
        try:
            ydata = dchunk[yf]
        except KeyError:
            ydata = self.dfi(dchunk, self.u)
        if wf is not None:
            try:
                wdata = dchunk[wf]
            except KeyError:
                wdata = self.dfi(dchunk, self.u)
            if phase is not None:
                wdata = wdata * phase.data.flatten()
            name = f"{wf}-pdf"
        else:
            name = "vol-pdf"

        if logx:
            xdata = np.log10(np.abs(xdata))
            xf = f"log_{xf}"
        if logy:
            ydata = np.log10(np.abs(ydata))
            yf = f"log_{yf}"

        b1 = np.linspace(xlim[0], xlim[1], Nx)
        b2 = np.linspace(ylim[0], ylim[1], Ny)
        h, b1, b2 = np.histogram2d(
            xdata.data.flatten(),
            ydata.data.flatten(),
            weights=wdata.data.flatten() if wf is not None else None,
            bins=[b1, b2],
        )
        dx = b1[1] - b1[0]
        dy = b2[1] - b2[0]
        pdf = h.T / dx / dy
        da = xr.DataArray(
            pdf,
            coords=[0.5 * (b2[1:] + b2[:-1]), 0.5 * (b1[1:] + b1[:-1])],
            dims=[yf, xf],
            name=name,
        )
        return da

    def check_configure_options(self):
        par = self.par
        athenapp = "mesh" in par

        # default configuration
        cooling = False
        newcool = False
        mhd = False
        radps = False
        sixray = False
        wind = False
        xray = False
        cosmic_ray = False
        feedback_scalars = False

        if athenapp:
            # Athena++ configuration
            try:
                mhd = par["configure"]["Magnetic_fields"] == "ON"
            except KeyError:
                pass

            try:
                newcool = par["photchem"]["mode"] == "ncr"
            except KeyError:
                pass

            try:
                cooling = par["cooling"]["cooling"] != "none"
            except KeyError:
                pass

            try:
                cosmic_ray = par["configure"]["Cosmic_Ray_Transport"] == "Multigroups"
            except KeyError:
                pass

            try:
                feedback_scalars = par["configure"]["Number_of_feedback_scalar"] > 0
            except KeyError:
                pass
        else:
            # Athena configuration
            try:
                newcool = par["configure"]["new_cooling"] == "ON"
            except KeyError:
                pass

            try:
                mhd = par["configure"]["gas"] == "mhd"
            except KeyError:
                pass

            try:
                cooling = par["configure"]["cooling"] == "ON"
            except KeyError:
                pass

            try:
                radps = par["configure"]["radps"] == "ON"
            except KeyError:
                pass

            try:
                sixray = par["configure"]["sixray"] == "ON"
            except KeyError:
                pass

            try:
                wind = par["feedback"]["iWind"] != 0
            except KeyError:
                pass

            try:
                xray = (
                    (par["feedback"]["iSN"] > 0)
                    or (par["feedback"]["iWind"] > 0)
                    or (par["feedback"]["iEarly"] > 0)
                )
            except KeyError:
                pass
        options = dict(
            athenapp=athenapp,
            cooling=cooling,
            newcool=newcool,
            mhd=mhd,
            radps=radps,
            sixray=sixray,
            wind=wind,
            xray=xray,
            cosmic_ray=cosmic_ray,
            feedback_scalars=feedback_scalars,
        )

        self.options = options

    @staticmethod
    def get_phase_Tlist(kind="ncr"):
        if kind == "ncr":
            return [500, 6000, 15000, 35000, 5.0e5]
        elif kind == "classic":
            return [184, 5050, 20000, 5.0e5]

    @staticmethod
    def get_phase_T1list():
        return [500, 6000, 13000, 24000, 1.0e6]

    def set_phase(self, data):
        temp_cuts = self.get_phase_Tlist("classic")
        phase = xr.zeros_like(data["temperature"],dtype="int") + len(temp_cuts)
        self.phlist = ["CNM","UNM","WNM","WHIM","HIM"]
        for i,tcut in enumerate(temp_cuts):
            phase = xr.where(data["temperature"] < tcut, phase-1, phase)
        return phase

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

    def __repr__(self):
        """Return a hierarchical string representation of __dict__."""
        def format_dict(d, indent=0):
            lines = []
            for k, v in d.items():
                pad = '  ' * indent
                if isinstance(v, dict):
                    lines.append(f"{pad}{k}:")
                    lines.extend(format_dict(v, indent + 1))
                else:
                    lines.append(f"{pad}{k}: {repr(v)}")
            return lines
        lines = format_dict(self.__dict__)
        return f"<{self.__class__.__name__}>\n" + "\n".join(lines)

    def __getattr__(self, key):
        """Return the simulation object from simdict for the given key."""
        if key in self.simdict:
            return self.simdict[key]
        raise AttributeError(f"{self.__class__.__name__} has no attribute '{key}'")

