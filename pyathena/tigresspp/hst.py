# read_hst.py

# import os
import os.path as osp

import glob
import xarray as xr
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

        h = dict()
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
        # sink mass history list
        for head in ["msink", "mstar"]:
            for tail in ["", "_old", "10", "40", "100"]:
                mf = f"{head}{tail}"
                if mf in hst:
                    h[mf] = hst[mf] * u.Msun
                    h[f"Sigma_{mf}"] = h[mf] / (LxLy * u.pc**2)
        if "p0m" in hst:
            h["mass_sp"] = hst["p0m"] * u.Msun
            h["Sigma_sp"] = h["mass_sp"] / (LxLy * u.pc**2)
            if "msp_removed" in hst:
                h["mass_sp_rm"] = hst["msp_removed"] * u.Msun
                h["Sigma_sp_rm"] = h["mass_sp_rm"] / (LxLy * u.pc**2)
            if "msp_star_removed" in hst:
                h["mass_sp_rm"] = hst["msp_removed"] * u.Msun
                h["Sigma_sp_rm"] = h["mass_sp_rm"] / (LxLy * u.pc**2)
                h["mass_star_rm"] = hst["msp_star_removed"] * u.Msun
                h["Sigma_star_rm"] = h["mass_star_rm"] / (LxLy * u.pc**2)
                h["mass_sink_rm"] = hst["msp_removed"] * u.Msun
                h["Sigma_sink_rm"] = h["mass_sink_rm"] / (LxLy * u.pc**2)
        if "scalar_indices" in par:
            if "IRT" in par["scalar_indices"]:
                IRT = par["scalar_indices"]["IRT"]
                # Mass return
                h["mass_return"] = hst[f"{IRT}scalar"] * u.Msun
                h["Sigma_return"] = h["mass_return"] / (LxLy * u.pc**2)
            if "ISN" in par["scalar_indices"]:
                ISN = par["scalar_indices"]["ISN"]
                # Mass return
                h["mass_sn"] = hst[f"{ISN}scalar"] * u.Msun
                h["Sigma_sn"] = h["mass_sn"] / (LxLy * u.pc**2)
            if "IDZ" in par["scalar_indices"]:
                IDZ = par["scalar_indices"]["IDZ"]
                # Mass return
                h["mass_metal"] = hst[f"{IDZ}scalar"] * u.Msun
                h["Sigma_metal"] = h["mass_metal"] / (LxLy * u.pc**2)
        if "mass_loss_upper" in hst:
            h["mass_out"] = (hst["mass_loss_upper"] - hst["mass_loss_lower"]) * u.Msun
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
        if "totE" in h:
            h["TE"] = hst["totE"] - h["KE"]
        if mhd:
            h["ME"] = hst["1ME"] + hst["2ME"] + hst["3ME"]
            if "TE" in h:
                h["TE"] -= h["ME"]
        if "TE" in h:
            h["pok"] = h["TE"] / 1.5 * u.pok  # assuming gamma=5/3
            h["cs"] = np.sqrt(h["TE"] / hst["mass"]) * u.kms

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

        # Star formation rate per area [Msun/kpc^2/yr]
        if "sfr10" in hst:
            h["sfr10"] = hst["sfr10"]
            h["sfr40"] = hst["sfr40"]
        elif "mstar10" in hst:
            h["sfr10"] = h["mstar10"] / (LxLy * u.pc**2) / 10
            h["sfr40"] = h["mstar40"].copy() / (LxLy * u.pc**2) / 40

        if "SFUV" in h:
            h["SFUV"] = hst["SFUV"]  # L_sun/pc^2
        elif "LFUV" in h:
            h["SFUV"] = h["LFUV"].copy() / (LxLy * u.pc**2)  # L_sun/pc^2
        if "heat_ratio" in h:
            h["heat_ratio"] = hst["heat_ratio"]  # L_sun/pc^2

        h = pd.DataFrame(h)
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

    # def read_hst_phase(self):
    #     hstfile_base=f"{self.files['hst']}.phase"

    #     phasenames = ['cold','unstable','warm','warmhot','hot']
    #     colors = ['blue','cyan','green','orange','red']
    #     cdict = dict()

    #     hst = dict()
    #     for iph,(name,color) in enumerate(zip(phasenames,colors)):
    #         hst[name] = read_hst(f"{hstfile_base}{iph}")
    #         cdict[name] = color

    #     # sum all phases
    #     hst['all'] = hst['cold'].copy()
    #     for name in phasenames[1:]:
    #         hst['all'] += hst[name]
    #     hst['all']['time'] = hst['cold']['time']
    #     hst['all']['dt'] = hst['cold']['dt']
    #     phasenames += ["all"]
    #     cdict["all"] = "black"

    #     for j,var in enumerate(["vol","mass"]):
    #         for iph,name in enumerate(phasenames):
    #             hst[name][f"{var}_frac"] = hst[name][var]/hst["all"][var]

    #     for j,var in enumerate(["1KE","2KE","3KE"]):
    #         for iph,name in enumerate(phasenames):
    #             hst[name][f"v{var[0]}"] = np.sqrt(2.0*hst[name][var]/hst[name]["mass"])

    #     self.hst_phase = hst
    #     self.cdict_phase = cdict

    def set_phase_history(self):
        if "phase_hst" not in self.files:
            self.files["phase_hst"] = sorted(glob.glob(self.files["hst"] + ".phase?"))
        phlist = []
        for fname in self.files["phase_hst"]:
            with open(fname, "r") as fp:
                line = fp.readline()
                phname = line[line.find("phase") :].strip().split("=")[-1]
            phlist.append(phname)
        self.phlist = phlist

    @LoadSim.Decorators.check_netcdf
    def load_phase_hst(
        self, prefix="merged_phase_hst", savdir=None, force_override=False, dryrun=False, filebase=None,
    ):
        if "phase_hst" not in self.files:
            self.set_phase_history()

        if dryrun:
            mtime = -1
            for f in self.files["phase_hst"]:
                mtime = max(osp.getmtime(f), mtime)
            return max(mtime, osp.getmtime(__file__))

        hst = dict()
        for fname, ph in zip(self.files["phase_hst"], self.phlist):
            h = read_hst(fname)
            h.index = h.pop("time")
            hst[ph] = h

        return xr.Dataset(hst).to_array("phase").to_dataset(dim="dim_1")

    def plt_hst(self):
        h = self.read_hst()
        if "mass_loss_upper" in h:
            fig, axes = plt.subplots(2, 3, figsize=(8, 6), num=0)
        else:
            fig, axes = plt.subplots(1, 3, figsize=(8, 3), num=0)
        axes = axes.flatten()
        plt.sca(axes[0])
        plt.plot(h["time"], h["Sigma_gas"], label="gas")
        if "Sigma_msink" in h:
            plt.plot(h["time"], h["Sigma_msink"], label="sink")
        else:
            if "Sigma_sp" in h:
                plt.plot(h["time"], h["Sigma_sp"], label="sink")
            if "Sigma_sp_rm" in h:
                plt.plot(
                    h["time"], h["Sigma_sp"] + h["Sigma_sp_rm"], label="sink,total"
                )
        if "Sigma_mstar" in h:
            plt.plot(h["time"], h["Sigma_mstar"], label="star")
        if "Sigma_out" in h:
            plt.plot(h["time"], h["Sigma_out"], label="out")
        plt.ylabel(r"$\Sigma$")
        plt.xlabel("time")
        plt.legend(fontsize="xx-small", frameon=False)

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
            dm = h["mass"] - h["mass"].iloc[0]
            mloss = h["mass_out"] - h["mass_out"].iloc[0]
            if "msink" in h:
                dmsp = h["msink"] - h["msink"].iloc[0]
            else:
                dmsp = h["mass_sp"] - h["mass_sp"].iloc[0]
            if "mass_sp_rm" in h:
                dmsp_removed = h["mass_sp_rm"] - h["mass_sp_rm"].iloc[0]
            else:
                dmsp_removed = np.zeros_like(h["time"])
            mtot = h["mass"].iloc[0]

            plt.sca(axes[3])
            plt.plot(h["time"], dm, label=r"$\Delta m_{\rm gas}$")
            plt.plot(
                h["time"],
                -dmsp - dmsp_removed - mloss,
                label=r"$-\Delta m_{\rm sink}-\Delta m_{\rm sink,rm}-m_{\rm out}$",
                ls=":",
            )
            plt.plot(h["time"], dmsp, label=r"$\Delta m_{\rm sink}$")
            plt.plot(h["time"], dmsp_removed, label=r"$\Delta m_{\rm sink,rm}$")
            plt.plot(h["time"], mloss, label=r"$\Delta m_{\rm out}$")
            plt.axhline(0, ls="--", color="k")
            plt.legend(fontsize="xx-small", frameon=False)
            plt.xlabel("time")
            plt.ylabel(r"mass [$M_\odot$]")

            plt.sca(axes[4])
            plt.plot(h["time"], ((dm) + dmsp + dmsp_removed + mloss) / mtot, ls=":")
            plt.xlabel("time")
            plt.ylabel(
                r"$(\Delta m_{\rm gas}+\Delta m_{\rm sink}+\Delta m_{\rm sink,rm}$"
                r"$+\Delta m_{\rm out})/m_{\rm gas}(0)$"
            )
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
