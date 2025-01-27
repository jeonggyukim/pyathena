# read_hst.py

# import os
import os.path as osp

# import glob
# import xarray as xr
import numpy as np
import pandas as pd

# from scipy import integrate
import matplotlib.pyplot as plt
# import astropy.units as au
# import astropy.constants as ac

from ..io.read_hst import read_hst
from ..load_sim import LoadSim


class Hst:
    @LoadSim.Decorators.check_pickle_hst
    def read_hst(self, savdir=None, force_override=False):
        """Function to read hst and convert quantities to convenient units"""

        par = self.par
        u = self.u
        domain = self.domain

        # Area of domain (code unit)
        LxLy = domain["Lx"][0] * domain["Lx"][1]
        # Lz
        # Lz = domain['Lx'][2]

        # qshear = par["orbital_advection"]['qshear']
        Omega0 = par["orbital_advection"]["Omega0"] * (u.kms * u.pc)
        if Omega0 > 0:
            time_orb = 2 * np.pi / Omega0 * u.Myr  # Orbital time in Myr
        else:
            time_orb = 1.0

        hst = read_hst(self.files["hst"], force_override=force_override)

        h = pd.DataFrame()
        for k in hst:
            h[k] = hst[k]
        if "1ME" in hst:
            mhd = True
        else:
            mhd = False

        # Time in code unit
        h["time_code"] = hst["time"]
        # Time in Myr
        h["time"] = h["time_code"] * u.Myr
        h["time_orb"] = h["time"] / time_orb
        # Time step
        h["dt_code"] = hst["dt"]
        h["dt"] = hst["dt"] * u.Myr

        # Total gas mass in Msun
        h["mass"] = hst["mass"] * u.Msun
        if "p0m" in hst:
            h["mass_sp"] = hst["p0m"] * u.Msun
            h["Sigma_sp"] = h["mass_sp"] / (LxLy * u.pc**2)
            if "msp_removed" in hst:
                h["mass_sp_rm"] = hst["msp_removed"] * u.Msun
                h["Sigma_sp_rm"] = h["mass_sp_rm"] / (LxLy * u.pc**2)
        if "mass_loss_upper" in hst:
            h["mass_out"] = (hst["mass_loss_upper"]-hst["mass_loss_lower"]) * u.Msun
            h["Sigma_out"] = h["mass_out"] / (LxLy * u.pc**2)
        # Mass surface density in Msun/pc^2
        h["Sigma_gas"] = h["mass"] / (LxLy * u.pc**2)

        # Calculate (cumulative) SN ejecta mass
        # JKIM: only from clustered type II(?)
        # try:
        #     sn = read_hst(self.files['sn'], force_override=force_override)
        #     t_ = np.array(hst['time'])
        #     Nsn, snbin = np.histogram(sn.time, bins=np.concatenate(([t_[0]], t_)))
        #     h['mass_snej'] = Nsn.cumsum()*self.par['feedback']['MejII'] # Mass of SN ejecta [Msun]
        #     h['Sigma_snej'] = h['mass_snej']/(LxLy*u.pc**2)
        # except KeyError:
        #     pass

        # Kinetic, thermal and magnetic energy
        h["KE"] = hst["1KE"] + hst["2KE"] + hst["3KE"]
        h["TE"] = hst["totE"] - h["KE"]
        if mhd:
            h["ME"] = hst["1ME"] + hst["2ME"] + hst["3ME"]
            h["TE"] -= h["ME"]
        h["pok"] = h["TE"] / 1.5 * u.pok  # assuming gamma=5/3

        for ax in ("1", "2", "3"):
            Ekf = "{}KE".format(ax)
            # if ax == '2':
            # Ekf = 'x2dke'
            # Mass weighted velocity dispersions
            h["v{}".format(ax)] = np.sqrt(2 * hst[Ekf] / hst["mass"]) * u.kms
            if mhd:
                h["vA{}".format(ax)] = (
                    np.sqrt(2 * hst["{}ME".format(ax)] / hst["mass"]) * u.kms
                )

        h["cs"] = np.sqrt(h["TE"] / hst["mass"]) * u.kms

        # Star formation rate per area [Msun/kpc^2/yr]
        h["sfr10"] = hst["sfr10"]
        h["sfr40"] = hst["sfr40"]
        if "SFUV" in h:
            h["SFUV"] = hst["SFUV"]  # L_sun/pc^2
        if "heat_ratio" in h:
            h["heat_ratio"] = hst["heat_ratio"]  # L_sun/pc^2

        h.index = h["time_code"]

        self.hst = h

        # SN data
        # if "sn" in self.files:
        #     if osp.exists(self.files["sn"]):
        #         sn = pd.read_csv(self.files["sn"])
        #         snr = get_snr(sn["time"] * self.u.Myr, hst["time"] * self.u.Myr)

        #         self.sn = sn
        #         self.snr = snr / LxLy

        return h

    def plt_hst(self):
        h = self.read_hst()
        if "mass_loss_upper" in h:
            fig, axes = plt.subplots(2, 3, figsize=(8, 6), num=0)
        else:
            fig, axes = plt.subplots(1, 3, figsize=(8, 3), num=0)
        axes = axes.flatten()
        plt.sca(axes[0])
        plt.plot(h["time"], h["Sigma_gas"], label="gas")
        if "Sigma_sp" in h:
            plt.plot(h["time"], h["Sigma_sp"], label="sink")
        if "Sigma_sp" in h:
            plt.plot(h["time"], h["Sigma_sp"]+h["Sigma_sp_rm"], label="sink,total")
        if "Sigma_out" in h:
            plt.plot(h["time"], h["Sigma_out"], label="out")
        plt.ylabel(r"$\Sigma$")
        plt.xlabel("time")
        plt.legend(fontsize="xx-small",frameon=False)

        plt.sca(axes[1])
        plt.plot(h["time"], h["sfr10"])
        plt.plot(h["time"], h["sfr40"])
        plt.ylabel(r"$\Sigma_{\rm SFR}$")
        plt.xlabel("time")

        plt.yscale("log")

        if "SFUV" in h:
            plt.sca(axes[2])
            plt.plot(h["time"], h["SFUV"])
            plt.plot(h["time"], h["heat_ratio"])
            plt.ylabel(r"$\Sigma_{\rm FUV}, \Sigma_{\rm FUV}/\Sigma_{\rm FUV,0}$")
            plt.xlabel("time")
            plt.yscale("log")

        if "mass_loss_upper" in h:
            dm = h["mass"]-h["mass"][0]
            mloss = h["mass_out"]
            dmsp = h["mass_sp"] - h["mass_sp"][0]
            dmsp_removed = h["mass_sp_rm"]
            mtot = h["mass"][0]

            plt.sca(axes[3])
            plt.plot(h["time"],dm,label=r"$\Delta m_{\rm gas}$")
            plt.plot(h["time"],-dmsp-dmsp_removed-mloss, label=r"$-\Delta m_{\rm sink}-\Delta m_{\rm sink,rm}-m_{\rm out}$",ls=":")
            plt.plot(h["time"],dmsp, label=r"$\Delta m_{\rm sink}$")
            plt.plot(h["time"],dmsp_removed, label=r"$\Delta m_{\rm sink,rm}$")
            plt.plot(h["time"],mloss, label=r"$\Delta m_{\rm out}$")
            plt.axhline(0,ls="--",color="k")
            plt.legend(fontsize="xx-small",frameon=False)
            plt.xlabel("time")
            plt.ylabel(r"mass [$M_\odot$]")

            plt.sca(axes[4])
            plt.plot(h["time"],((dm)+dmsp+dmsp_removed+mloss)/mtot,ls=":")
            plt.xlabel("time")
            plt.ylabel(r"$(\Delta m_{\rm gas}+\Delta m_{\rm sink}+\Delta m_{\rm sink,rm}$"
                    r"$+\Delta m_{\rm out})/m_{\rm gas}(0)$")
        plt.tight_layout()

        return fig


def get_snr(sntime, time, tbin="auto", snth=100.0):
    import xarray as xr

    snt = sntime.to_numpy()
    t = time.to_numpy()
    if tbin == "auto":
        tbin = 0.0
        dtbin = 0.1
        snrmean = 0.0
        while (tbin < 40) & (snrmean < snth):
            tbin += dtbin
            idx = np.less(snt[np.newaxis, :], t[:, np.newaxis]) & np.greater(
                snt[np.newaxis, :], (t[:, np.newaxis] - tbin)
            )
            snr = idx.sum(axis=1)
            snrmean = snr.mean()

        snr = snr / tbin
    else:
        idx = np.less(snt[np.newaxis, :], t[:, np.newaxis]) & np.greater(
            snt[np.newaxis, :], (t[:, np.newaxis] - tbin)
        )
        snr = idx.sum(axis=1) / tbin
    snr = xr.DataArray(snr, coords=[time], dims=["time"])
    return snr
