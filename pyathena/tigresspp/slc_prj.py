# slc_prj.py

import os
import os.path as osp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cmasher as cmr
from matplotlib import cm

# import astropy.units as au
# import astropy.constants as ac
from matplotlib.colors import Normalize, LogNorm, SymLogNorm
import xarray as xr

from ..load_sim import LoadSim


cpp_to_cc = {
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

class SliceProj:
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
        ds = self.load_hdf5(num=num, file_only=True)

        if dryrun:
            return max(osp.getmtime(self.fhdf5),osp.getmtime(__file__))

        ds = self.get_data(num, load_derived=False)
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
        prjkwargs["Sigma_HI"] = prjkwargs["Sigma"]
        prjkwargs["Sigma_HII"] = prjkwargs["Sigma"]
        prjkwargs["Sigma_H2"] = prjkwargs["Sigma"]
        prjkwargs["Sigma_EL"] = prjkwargs["Sigma"]
        prjkwargs["EM"] = dict(norm=LogNorm(1.0e-2, 1.0e4), cmap=plt.cm.plasma)
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
        labels["Sigma_HI"] = r"$\Sigma_{\rm gas,H}\,[{\rm M_\odot\,pc^{-2}}]$"
        labels["Sigma_H2"] = r"$\Sigma_{\rm gas,H_2}\,[{\rm M_\odot\,pc^{-2}}]$"
        labels["Sigma_HII"] = r"$\Sigma_{\rm gas,H^+}\,[{\rm M_\odot\,pc^{-2}}]$"
        labels["Sigma_EL"] = r"$\Sigma_{\rm e}\,[{\rm M_\odot\,pc^{-2}}]$"
        labels["EM"] = r"${\rm EM}\,[{\rm cm^{-6}\,pc}]$"
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
        data = self.load_hdf5(num=num, file_only=True)

        if dryrun:
            return max(osp.getmtime(self.fhdf5),osp.getmtime(__file__))

        data = self.get_data(num, load_derived=False)

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
            if "0-Vd3" in data:
                prjdata["creflux_diff"] = data["0-Ec"]*data["0-Vd3"]*conv_eflux*4/3.
                prjdata["creflux_adv"] = data["0-Ec"]*data["vel3"]*conv_eflux*4/3.
                prjdata["creflux_str"] = data["0-Ec"]*data["0-Vs3"]*conv_eflux*4/3.
        if self.options["newcool"]:
            prjdata["Sigma_HI"] = data["rho"]*data["rHI"] * conv_surf
            prjdata["Sigma_H2"] = 2*data["rho"]*data["rH2"] * conv_surf
            prjdata["Sigma_HII"] = data["rho"]*(1-data["rHI"]-2*data["rH2"]) * conv_surf
            prjdata["Sigma_EL"] = data["rho"]*data["rEL"] * conv_surf
            prjdata["EM"] = (data["rho"]*data["rEL"])**2
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
            for f in prjdata:
                if f.startswith("Sigma") or f.startswith("EM"):
                    prj[f] *= Lx
            res_ax.append(prj.assign_coords(phase=phase))
        prj = xr.concat(res_ax,dim="phase")
        prj.attrs = dict(time=data.attrs["Time"])
        return prj

    @LoadSim.Decorators.check_netcdf
    def get_windpdf(self, num,
            prefix,
            savdir=None,
            force_override=False,
            filebase=None,
            dryrun=False,
            zlist = [500,1000,2000,3000], dz = 50):
        ds = self.load_hdf5(num=num, file_only=True)

        if dryrun:
            return max(osp.getmtime(self.fhdf5),osp.getmtime(__file__))

        ds = self.get_data(num, load_derived=False)

        pdf={"out":xr.Dataset(),"in":xr.Dataset()}

        bin = np.logspace(0,4,201)
        dbin = np.log10(bin[1]/bin[0])
        bcc = np.log10(bin)[:-1] + dbin
        munits = (self.u.density*self.u.velocity).to("Msun/(kpc**2*yr)").value
        eunits = (self.u.energy_density*self.u.velocity).to("erg/(kpc**2*yr)").value
        for z0 in zlist:
            ds_sel = ds.sel(z=slice(z0-dz,z0+dz)).stack(xyz=["x","y","z"])
            cs = np.sqrt(ds_sel["press"]/ds_sel["rho"])
            zsgn= ds_sel.z/np.abs(ds_sel.z)
            vout = ds_sel["vel3"]*zsgn
            rho = ds_sel["rho"]

            mflux_out = rho*vout
            vsq = (ds_sel["vel1"]**2 + ds_sel["vel2"]**2 + ds_sel["vel3"]**2)
            csq = self.par["hydro"]["gamma"]/(self.par["hydro"]["gamma"]-1)*cs**2
            vBsq = vsq + csq
            eflux_kin =0.5*mflux_out*vsq
            eflux_th =mflux_out*csq
            eflux = eflux_kin + eflux_th
            if self.options["mhd"]:
                Bout = ds_sel["Bcc3"]*zsgn
                vAsq = (ds_sel["Bcc1"]**2 + ds_sel["Bcc2"]**2 + ds_sel["Bcc3"]**2)/rho
                eflux_mag = mflux_out*vAsq
                eflux_magt = (ds_sel["Bcc1"]*ds_sel["vel1"] + \
                              ds_sel["Bcc2"]*ds_sel["vel2"] + \
                              ds_sel["Bcc3"]*ds_sel["vel3"])*Bout
                eflux += eflux_mag - eflux_magt


            mZflux_out = mflux_out*ds_sel["rmetal"]
            fluxlist = {
                "mflux": mflux_out*munits,
                "mZflux": mZflux_out*munits,
                "teflux": eflux_th*eunits,
                "keflux": eflux_kin*eunits,
                "eflux": eflux*eunits,
            }
            if self.options["mhd"]:
                fluxlist["meflux"] = (eflux_mag-eflux_magt)*eunits
                fluxlist["meflux_p"] = eflux_mag*eunits
                fluxlist["meflux_t"] = eflux_magt*eunits
            outpdflist = []
            inpdflist =[]
            for f in fluxlist:
                outpdf,_,_ = np.histogram2d(vout,cs,
                                            bins=[bin,bin],
                                            weights=fluxlist[f])
                inpdf,_,_ = np.histogram2d(-vout,cs,
                                            bins=[bin,bin],
                                            weights=-fluxlist[f])

                outpdf=xr.DataArray(outpdf.T,dims=["logcs","logvz"],coords=[bcc,bcc])
                inpdf=xr.DataArray(inpdf.T,dims=["logcs","logvz"],coords=[bcc,bcc])
                outpdflist.append(outpdf.assign_coords(flux=f))
                inpdflist.append(inpdf.assign_coords(flux=f))
            pdf["out"][z0]=xr.concat(outpdflist,dim="flux")
            pdf["in"][z0]=xr.concat(inpdflist,dim="flux")

        pdf["out"]=pdf["out"].to_array("z")
        pdf["in"]=pdf["in"].to_array("z")

        return xr.Dataset(pdf).assign_coords(time=ds.attrs["Time"])
