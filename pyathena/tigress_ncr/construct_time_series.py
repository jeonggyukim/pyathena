import os
import glob
import xarray as xr
import numpy as np
from pyathena.tigress_ncr.phase import assign_phase
from pyathena.tigress_ncr.get_cooling import (
    get_heating,
    get_other_cooling,
    get_hydrogen_cooling,
)
from pyathena.microphysics.cool import heatCR
import astropy.constants as ac

# import astropy.units as au
from pyathena.tigress_ncr.ncr_paper_lowz import LowZData
from mpi4py import MPI


def f1(T, T0=2e4, T1=3.5e4):
    """transition function"""
    return np.where(
        T > T1,
        1.0,
        np.where(
            T <= T0,
            0.0,
            1.0 / (1.0 + np.exp(-10.0 * (T - 0.5 * (T0 + T1)) / (T1 - T0))),
        ),
    )


class athena_data(object):
    def __init__(self, s, data):
        self.sim = s
        self.u = s.u
        self.data = data

    def __repr__(self):
        return self.data.__repr__()

    def keys(self):
        return list(self.data.keys())

    def derived_keys(self):
        return list(self.sim.dfi.keys())

    def assign_phase(self):
        self.phase = assign_phase(self.sim, self, kind="six")

    def update_cooling_heating(self):
        w2 = f1(
            self["T"],
            T0=self.sim.par["cooling"]["Thot0"],
            T1=self.sim.par["cooling"]["Thot1"],
        )
        w1 = 1 - w2
        self.data_heat = get_heating(self.sim, self) * self["nH"] * w1
        self.data_cool = get_other_cooling(self.sim, self) * self["nH"] * w1
        self.data_cool.update(get_hydrogen_cooling(self.sim, self) * self["nH"])
        for var in self.data_cool:
            if var.startswith("Lambda") or var.startswith("Gamma"):
                continue
            self.data_cool[f"Lambda_{var}"] = self.data_cool[var] / self["nH"] ** 2
            self.data_cool[f"Lambda_nH_{var}"] = self.data_cool[var] / self["nH"]
        for var in self.data_heat:
            if var.startswith("Lambda") or var.startswith("Gamma"):
                continue
            self.data_heat[f"Gamma_{var}"] = self.data_heat[var] / self["nH"]

    def __getitem__(self, field):
        if field in self.data:
            return self.data[field]
        elif field == "T":
            self.data["T"] = self.data["temperature"]
        elif field in self.sim.dfi:
            self.data[field] = self.sim.dfi[field]["func"](self.data, self.sim.u)
        elif field == "charging":
            self.data[field] = (
                1.7 * self["chi_FUV"] * np.sqrt(self["T"]) / (self["ne"]) + 50.0
            )
        elif field == "eps_PE":
            CPE_ = np.array([5.22, 2.25, 0.04996, 0.00430, 0.147, 0.431, 0.692])
            T = self["T"]
            x = self["charging"]
            eps = (CPE_[0] + CPE_[1] * np.power(T, CPE_[4])) / (
                1.0
                + CPE_[2]
                * np.power(x, CPE_[5])
                * (1.0 + CPE_[3] * np.power(x, CPE_[6]))
            )
            self.data[field] = eps
        elif field == "heat_PE":
            heat = 1.7e-26 * self["chi_FUV"] * self.sim.Zdust * self["eps_PE"]
            self.data[field] = heat.where(
                self.data["T"] < self.sim.par["cooling"]["Thot1"]
            ).fillna(1.0e-35)
        elif field == "heat_CR":
            heat = heatCR(
                self["nH"], self["xe"], self["xHI"], self["xH2"], self["xi_CR"]
            )
            self.data[field] = heat.where(
                self.data["T"] < self.sim.par["cooling"]["Thot1"]
            ).fillna(1.0e-35)
        elif field == "ftau":
            self.data[field] = (
                self["rad_energy_density_PE"] / self["rad_energy_density_PE_unatt"]
            )
        elif field == "xHII":
            self.data["xHII"] = 1 - self["xHI"] - 2.0 * self["xH2"]
        elif field == "tcool":
            self.data["tcool"] = 1.5 * self["pressure"] / self["cool_rate"] * self.u.Myr
        elif field == "theat":
            self.data["theat"] = 1.5 * self["pressure"] / self["heat_rate"] * self.u.Myr
        elif field == "tnetcool":
            self.data["tnetcool"] = (
                1.5
                * self["pressure"]
                / np.abs(self["cool_rate"] - self["heat_rate"])
                * self.u.Myr
            )
        elif field == "vx2":
            self.data["vx2"] = self["vx"] ** 2
        elif field == "vy2":
            self.data["vy2"] = self["vy"] ** 2
        elif field == "vz2":
            self.data["vz2"] = self["vz"] ** 2
        elif field == "qcr":
            self.data["qcr"] = self.data_heat["Gamma_CR"] / self["xi_CR"]
        else:
            raise KeyError("{} is not available".format(field))
        return self.data[field]

    def __setitem__(self, field, value):
        self.data[field] = value

    def __contains__(self, attribute_name):
        return attribute_name in self.data

    def get_means(self, ph):
        flist = [
            "chi_FUV",
            "cool_rate",
            "heat_rate",
            "tcool",
            "theat",
            "tnetcool",
            "eps_PE",
            "charging",
            "pok",
            "nH",
            "T",
            "xe",
            "xHI",
            "xHII",
            "ne",
            "nHI",
            "nHII",
            "vx",
            "vx2",
            "vy",
            "vy2",
            "vz",
            "vz2",
            "xi_CR",
            "qcr",
        ]
        for f in flist:
            self[f]
        # density sum
        nsum = (self["nH"] * ph).sum(dim=["x", "y"])
        vsum = ph.sum(dim=["x", "y"])
        # volume weighted means
        v_means = (self.data[flist] * ph).sum(dim=["x", "y"]) / vsum
        v_means_heat = (self.data_heat * ph).sum(dim=["x", "y"]) / vsum
        v_means_cool = (self.data_cool * ph).sum(dim=["x", "y"]) / vsum
        v_means.update(v_means_heat)
        v_means.update(v_means_cool)
        # density weigthed means
        n_means = (self["nH"] * self.data[flist] * ph).sum(dim=["x", "y"]) / nsum
        n_means_heat = (self["nH"] * self.data_heat * ph).sum(dim=["x", "y"]) / nsum
        n_means_cool = (self["nH"] * self.data_cool * ph).sum(dim=["x", "y"]) / nsum
        n_means.update(n_means_heat)
        n_means.update(n_means_cool)
        return v_means, n_means


def construct_timeseries(s, m, force_override=False):
    outdir = os.path.join(s.savdir, "hst2")
    outfile = os.path.join(outdir, "PEheating.nc")
    outfile2 = os.path.join(outdir, "phase_vmeans.nc")
    outfile3 = os.path.join(outdir, "phase_nmeans.nc")
    # if os.path.isfile(outfile2) and (not force_override):
    #     with xr.open_dataarray(outfile) as da:
    #         da.load()
    #     with xr.open_dataset(outfile2) as ds1:
    #         ds1.load()
    #     with xr.open_dataset(outfile3) as ds2:
    #         ds2.load()

    #     return da, ds1, ds2

    if not hasattr(s, "slc") or force_override:
        allslc_files = sorted(
            glob.glob(os.path.join(s.basedir, "allslc", "allslc_*.p"))
        )
        slcnums = [int(os.path.basename(f)[-6:-2]) for f in allslc_files]
        fields = [
            "density",
            "pressure",
            "temperature",
            "rad_energy_density_PH",
            "rad_energy_density_LW",
            "rad_energy_density_PE",
            "rad_energy_density_LW_diss",
            "CR_ionization_rate",
            "cool_rate",
            "heat_rate",
            "xHI",
            "xH2",
            "xHII",
            "xe",
            "velocity1",
            "velocity2",
            "velocity3",
        ]
        s.slc = s.read_slc_time_series(
            nums=slcnums, fields=fields, sfr=True, radiation=True
        )
    s.data = athena_data(s, s.slc.sel(z=0))
    s.data.update_cooling_heating()
    s.data.assign_phase()
    dx = (s.slc.x[1] - s.slc.x[0]).data
    dy = (s.slc.y[1] - s.slc.y[0]).data
    (Nx,) = s.slc.x.shape
    (Ny,) = s.slc.y.shape
    Lx = Nx * dx
    Ly = Ny * dy
    area = Lx * Ly

    SFUV = s.slc["Ltot_FUV"] / area
    twop = s.data.phase < 3
    cnm = s.data.phase == 0
    cold = s.data.phase < 2
    wnm = s.data.phase == 2
    wim = s.data.phase == 3

    # JFUV, ePE, charging, xe, T
    # get weighted mean
    JFUV_2p = (s.data["chi_FUV"] * twop).sum(dim=["x", "y"]) / (twop).sum(
        dim=["x", "y"]
    )
    data_list = [
        SFUV.assign_coords(variable="SFUV"),
        JFUV_2p.assign_coords(variable="JFUV"),
    ]
    wdata = s.data["chi_FUV"] * twop
    wsum = wdata.sum(dim=["x", "y"])
    for fi in ["eps_PE", "charging", "xe", "T", "nH", "pok"]:
        data = (s.data[fi] * wdata).sum(dim=["x", "y"]) / wsum
        data_list.append(data.assign_coords(variable=fi))

    # get tau and ftau
    sigma_PE = s.par["opacity"]["sigma_dust_PE0"]
    sigma_LW = s.par["opacity"]["sigma_dust_LW0"]
    sigma_FUV = (sigma_PE * s.slc["Ltot_PE"] + sigma_LW * s.slc["Ltot_LW"]) / s.slc[
        "Ltot_FUV"
    ]
    NH = s.slc["Sigma_gas"] * ((ac.M_sun / ac.pc**2) / (s.u.muH * ac.m_p)).cgs.value
    H = s.slc["H_2p"]
    nmid = s.slc["nmid"]
    tau = s.Zdust * sigma_FUV * NH
    ftau_2p = 4 * np.pi * JFUV_2p / SFUV
    data_list += [
        NH.assign_coords(variable="NH"),
        nmid.assign_coords(variable="nmid"),
        H.assign_coords(variable="H"),
        tau.assign_coords(variable="tau"),
        ftau_2p.assign_coords(variable="ftau"),
    ]

    os.makedirs(outdir, exist_ok=True)
    da = xr.concat(data_list, "variable")
    da.name = m
    da.to_netcdf(outfile)
    da.close()

    # phase-separated weighted means
    vmeans = []
    nmeans = []
    for ph, phname in zip(
        [cnm, cold, wnm, twop, wim], ["CNM", "Cold", "WNM", "2p", "WIM"]
    ):
        vavg, navg = s.data.get_means(ph)
        eps = (s.data["nH"] * s.data["chi_FUV"] * s.data["eps_PE"] * ph).sum(
            dim=["x", "y"]
        ) / (s.data["nH"] * s.data["chi_FUV"] * ph).sum(dim=["x", "y"])
        navg["eps_PE_nchi"] = eps
        vmeans.append(vavg.assign_coords(phase=phname))
        nmeans.append(navg.assign_coords(phase=phname))

    ds1 = xr.concat(vmeans, "phase")
    ds1.to_netcdf(outfile2)
    ds1.close()

    ds2 = xr.concat(nmeans, "phase")
    ds2.to_netcdf(outfile3)
    ds2.close()

    return da, ds1, ds2


if __name__ == "__main__":
    pdata = LowZData()
    nmodels = len(pdata.mlist)
    COMM = MPI.COMM_WORLD
    mylist = [pdata.mlist[i] for i in range(nmodels) if i % COMM.size == COMM.rank]
    print(COMM.rank, mylist)
    for m in mylist:
        print(m)
        s = pdata.sa.set_model(m)
        da, vavg, navg = construct_timeseries(s, m, force_override=True)
