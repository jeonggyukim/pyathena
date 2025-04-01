import os.path as osp
import glob
import numpy as np
import pandas as pd
import xarray as xr
import warnings
from matplotlib.colors import Normalize, LogNorm, SymLogNorm
from matplotlib import cm
import cmasher as cmr

from .hst import Hst
from .timing import Timing
from ..load_sim import LoadSim
from pyathena.fields.fields import DerivedFields
import pyathena as pa

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


class LoadSimTIGRESSPP(LoadSim,Hst,Timing):
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

            if self.par["feedback"]["pop_synth"] == "SB99":
                # pop_synth_file1 = osp.join(basedir,self.par["feedback"]["pop_synth_file"])
                pop_synth_file2 = osp.join(basedir, "pop_synth.runtime.csv")
                if osp.isfile(pop_synth_file2):
                    self.pop_synth = pd.read_csv(pop_synth_file2)

            if "configure" in self.par:
                self.nghost = self.par["configure"]["Number_of_ghost_cells"]
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

    def get_data(self, num):
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
            chunks=dict(x=mb["nx1"], y=mb["nx2"], z=mb["nx3"]),
        )

        if "press" in ds:
            self.add_temperature(ds)
        return ds

    @LoadSim.Decorators.check_netcdf
    def get_slice(
        self,
        num,
        prefix,
        savdir=None,
        force_override=False,
        filebase=None,
        slc_kwargs=dict(z=0, method="nearest"),
        dryrun=False
    ):
        """
        a warpper function to make data reading easier
        """
        ds = self.get_data(num)

        if dryrun:
            return max(osp.getmtime(self.fhdf5),osp.getmtime(__file__))

        # rename the variables to match athena convention so that we can use
        # the same derived fields as in athena
        rename_dict = {k: v for k, v in cpp_to_cc.items() if k in ds}
        ds = ds.rename(rename_dict)
        slc = ds.sel(**slc_kwargs)
        slc.attrs = dict(time=ds.attrs["Time"])
        return slc

    def set_prj_dfi(self):
        prjkwargs = dict()
        prjkwargs["Sigma"] = dict(norm=LogNorm(1.e-2,1.e2),cmap=cm.pink_r)
        prjkwargs["mflux"] = dict(norm=SymLogNorm(1.e-4,vmin=-1.e-1,vmax=1.e-1),cmap=cmr.fusion_r)
        prjkwargs["mZflux"] = prjkwargs["mflux"]
        prjkwargs["teflux"] = dict(norm=SymLogNorm(1.e40,vmin=-1.e46,vmax=1.e46),cmap=cmr.viola)
        prjkwargs["keflux"] = prjkwargs["teflux"]
        prjkwargs["creflux"] = prjkwargs["teflux"]
        prjkwargs["creflux_diff"] = prjkwargs["creflux"]
        prjkwargs["creflux_adv"] = prjkwargs["creflux"]
        prjkwargs["creflux_str"] = prjkwargs["creflux"]
        labels = dict()
        labels["Sigma"] = r"$\Sigma_{\rm gas}\,[{\rm M_\odot\,pc^{-2}}]$"
        labels["mflux"] = r"$\mathcal{F}_{\rho}\,[{\rm M_\odot\,kpc^{-2}\,yr^{-1}}]$"
        labels["mZflux"] = r"$\mathcal{F}_{\rho Z}\,[{\rm M_\odot\,kpc^{-2}\,yr^{-1}}]$"
        labels["teflux"] = r"$\mathcal{F}_{e_{\rm th}}\,[{\rm erg\,kpc^{-2}\,yr^{-1}}]$"
        labels["keflux"] = r"$\mathcal{F}_{e_{\rm kin}}\,[{\rm erg\,kpc^{-2}\,yr^{-1}}]$"
        labels["creflux"] = r"$\mathcal{F}_{e_{\rm cr}}\,[{\rm erg\,kpc^{-2}\,yr^{-1}}]$"
        labels["creflux_diff"] = r"$\mathcal{F}_{e_{\rm cr},{\rm diff}}\,[{\rm erg\,kpc^{-2}\,yr^{-1}}]$"
        labels["creflux_adv"] = r"$\mathcal{F}_{e_{\rm cr},{\rm adv}}\,[{\rm erg\,kpc^{-2}\,yr^{-1}}]$"
        labels["creflux_str"] = r"$\mathcal{F}_{e_{\rm cr},{\rm str}}\,[{\rm erg\,kpc^{-2}\,yr^{-1}}]$"
        return prjkwargs,labels

    @LoadSim.Decorators.check_netcdf
    def get_prj(self,num,ax,
            prefix,
            savdir=None,
            force_override=False,
            filebase=None,
            dryrun=False):
        data = self.get_data(num)

        if dryrun:
            return max(osp.getmtime(self.fhdf5),osp.getmtime(__file__))

        axtoi = dict(x=0, y=1, z=2)

        prjdata = xr.Dataset()
        gamma = self.par["hydro"]["gamma"]
        Lx = self.domain["Lx"]
        conv_surf = (self.u.length*self.u.density).to("Msun/pc**2").value
        conv_mflux = (self.u.density*self.u.velocity).to("Msun/(kpc2*yr)").value
        conv_eflux = (self.u.energy_density*self.u.velocity).to("erg/(kpc2*yr)").value
        prjdata["Sigma"] = data["rho"] * conv_surf
        prjdata["mflux"] = data["rho"]*data["vel3"] * conv_mflux
        prjdata["mZflux"] = data["rho"]*data["rmetal"]*data["vel3"] * conv_mflux
        prjdata["teflux"] = gamma/(gamma-1)*data["press"]*data["vel3"] * conv_eflux
        prjdata["keflux"] = 0.5*data["rho"]*data["vel3"]*(data["vel1"]**2+data["vel2"]**2+data["vel3"]**2) * conv_eflux
        if self.options["cosmic_ray"]:
            prjdata["creflux"] = data["0-Fc3"]*conv_eflux
            prjdata["creflux_diff"] = data["0-Ec"]*data["0-Vd3"]*conv_eflux*4/3.
            prjdata["creflux_adv"] = data["0-Ec"]*data["vel3"]*conv_eflux*4/3.
            prjdata["creflux_str"] = data["0-Ec"]*data["0-Vs3"]*conv_eflux*4/3.

        i = axtoi[ax]
        dx = self.domain["dx"][i]
        Lx = self.domain["Lx"][i]
        res_ax = []
        for phase in ["whole","hot","wc"]:
            if phase == "hot":
                cond = data["temperature"]>2.e4
            elif phase == "wc":
                cond = data["temperature"]<=2.e4
            else:
                cond = 1.0
            prj = (prjdata*cond).sum(dim=ax)*dx/Lx
            prj["Sigma"] *= Lx
            res_ax.append(prj.assign_coords(phase=phase))
        prj = xr.concat(res_ax,dim="phase")
        prj.attrs = dict(time=data.attrs["Time"])
        return prj

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
    def load_zprof(self, prefix="merged_zprof", filebase=None,
                   savdir=None, force_override=False, dryrun=False):
        if dryrun:
            mtime = -1
            for f in self.files["zprof"]:
                mtime = max(osp.getmtime(f),mtime)
            return max(mtime,osp.getmtime(__file__))

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

    def update_derived_fields(self):
        dfi = DerivedFields(self.par)
        dfi.dfi["T"]["imshow_args"]["cmap"] = "Spectral_r"
        dfi.dfi["T"]["imshow_args"]["norm"] = LogNorm(vmin=1e2, vmax=1e8)
        dfi.dfi["nH"]["imshow_args"]["cmap"] = cmr.rainforest
        dfi.dfi["nH"]["imshow_args"]["norm"] = LogNorm(vmin=1e-4, vmax=1e2)

        if self.options["cosmic_ray"]:
            dfi.dfi["vmag"]["imshow_args"]["cmap"] = dfi.dfi["Vcr_mag"]["imshow_args"][
                "cmap"
            ]
            dfi.dfi["vmag"]["imshow_args"]["norm"] = dfi.dfi["Vcr_mag"]["imshow_args"][
                "norm"
            ]
            dfi.dfi["pok"]["imshow_args"]["cmap"] = dfi.dfi["pok_cr"]["imshow_args"][
                "cmap"
            ]
            dfi.dfi["pok"]["imshow_args"]["norm"] = dfi.dfi["pok_cr"]["imshow_args"][
                "norm"
            ]
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
            xf = f"log{xf}"
        if logy:
            ydata = np.log10(np.abs(ydata))
            yf = f"log{yf}"

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
