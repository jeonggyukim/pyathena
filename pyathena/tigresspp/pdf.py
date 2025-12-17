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

class PDF:
    @LoadSim.Decorators.check_netcdf
    def get_nPpdf(
        self,
        num,
        prefix,
        savdir=None,
        force_override=False,
        filebase=None,
        xf="nH",yf="pok",xlim=(-10,10),ylim=(-10,10),
        Nx=256,Ny=256,logx=True,logy=True,
        outid=None,
        dryrun=False
    ):
        """
        a warpper function to make data reading easier
        """
        ds = self.load_hdf5(num=num, outid=outid, file_only=True)

        if dryrun:
            return max(osp.getmtime(self.fhdf5),osp.getmtime(__file__))

        data = self.get_data(num,outid=outid,load_derived=True)

        dset = xr.Dataset()
        wlist = [None, "nH"]

        data = data[[xf,yf]+wlist[1:]].load()
        for wf in wlist:
            if wf is None:
                total = np.prod(self.domain["Nx"])
                coord_dict = {"vol": total}
            else:
                total = data[wf].sum().data
                coord_dict = {wf: total}
            da = self.get_pdf(data,xf,yf,wf,xlim,ylim,Nx=Nx,Ny=Ny,logx=logx,logy=logy)
            dset[da.name] = da
            dset = dset.assign_coords(coord_dict)
        return dset.assign_coords(time=data.attrs["Time"])

    @LoadSim.Decorators.check_netcdf
    def get_windpdf(self, num,
            prefix,
            savdir=None,
            force_override=False,
            filebase=None,
            dryrun=False,
            zlist = [-3000,-2000,-1000,-500,0,500,1000,2000,3000], dz = 50):
        ds = self.load_hdf5(num=num, file_only=True)

        if dryrun:
            return osp.getmtime(self.fhdf5)
            # return max(osp.getmtime(self.fhdf5),osp.getmtime(__file__))

        ds = self.get_data(num, load_derived=False)

        pdf={"out":xr.Dataset(),"in":xr.Dataset()}

        bin = np.logspace(0,4,201)
        dbin = np.log10(bin[1]/bin[0])
        bcc = np.log10(bin)[:-1] + dbin
        mfunits = (self.u.density*self.u.velocity).to("Msun/(kpc**2*yr)").value
        pfunits = (self.u.energy_density).to("(Msun*km)/(kpc**2*yr*s)").value
        efunits = (self.u.energy_density*self.u.velocity).to("erg/(kpc**2*yr)").value
        for z0 in zlist:
            ds_sel = ds.sel(z=slice(z0-dz,z0+dz)).stack(xyz=["x","y","z"])
            cs = np.sqrt(ds_sel["press"]/ds_sel["rho"])
            zsgn= ds_sel.z/np.abs(ds_sel.z)
            vout = ds_sel["vel3"]*zsgn
            vol = ds_sel["rho"]/ds_sel["rho"]
            rho = ds_sel["rho"]
            press = ds_sel["press"]

            # mass flux
            mflux_out = rho*vout
            vsq = (ds_sel["vel1"]**2 + ds_sel["vel2"]**2 + ds_sel["vel3"]**2)
            csq = self.par["hydro"]["gamma"]/(self.par["hydro"]["gamma"]-1)*cs**2
            vBsq = vsq + csq
            # momentum flux
            pflux_kin = mflux_out*vout
            pflux_th = press
            pflux = pflux_kin+pflux_th
            if self.options["mhd"]:
                pflux_mag = 0.5*(ds_sel["Bcc1"]**2+ds_sel["Bcc2"]**2-ds_sel["Bcc3"]**2)
                pflux += pflux_mag

            # energy flux
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

            # metal flux
            mZflux_out = mflux_out*ds_sel["rmetal"]
            fluxlist = {
                "vol": vol,
                "mass": rho,
                "mflux": mflux_out*mfunits,
                "mflux_Z": mZflux_out*mfunits,
                "pflux": pflux*pfunits,
                "eflux": eflux*efunits,
                "pflux_kin": pflux_kin*pfunits,
                "pflux_th": pflux_th*pfunits,
                "eflux_kin": eflux_kin*efunits,
                "eflux_th": eflux_th*efunits
            }
            if self.options["mhd"]:
                fluxlist["eflux_mag"] = (eflux_mag-eflux_magt)*efunits
                fluxlist["eflux_magp"] = eflux_mag*efunits
                fluxlist["eflux_magt"] = eflux_magt*efunits
                fluxlist["pflux_mag"] = pflux_mag*pfunits
            outpdflist = []
            inpdflist =[]
            for f in fluxlist:
                outpdf,_,_ = np.histogram2d(vout,cs,
                                            bins=[bin,bin],
                                            weights=fluxlist[f])
                vsgn = -1 if ("mflux" in f) or ("eflux" in f) else 1
                inpdf,_,_ = np.histogram2d(-vout,cs,
                                            bins=[bin,bin],
                                            weights=vsgn*fluxlist[f])

                outpdf=xr.DataArray(outpdf.T,dims=["logcs","logvz"],coords=[bcc,bcc])
                inpdf=xr.DataArray(inpdf.T,dims=["logcs","logvz"],coords=[bcc,bcc])
                outpdflist.append(outpdf.assign_coords(flux=f))
                inpdflist.append(inpdf.assign_coords(flux=f))
            pdf["out"][z0]=xr.concat(outpdflist,dim="flux")
            pdf["in"][z0]=xr.concat(inpdflist,dim="flux")

        pdf["out"]=pdf["out"].to_array("z")
        pdf["in"]=pdf["in"].to_array("z")

        return xr.Dataset(pdf).assign_coords(time=ds.attrs["Time"])
