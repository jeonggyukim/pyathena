import os
import os.path as osp
import glob
import sys

filepath = os.path.dirname(__file__)
sys.path.insert(0, osp.join(filepath, "../"))

import xarray as xr
import astropy.constants as ac

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize, LogNorm, SymLogNorm
import cmasher as cmr

# import pyathena as pa
from .load_sim_tigresspp import LoadSimTIGRESSPPAll

plt.style.use(osp.join(filepath, "paper.mplstyle"))
model_name = {
    "crmhd-16pc-b1-diode-lngrad_out": "crmhd_v1",
    "crmhd_v2-16pc-b1-diode-lngrad_out": "crmhd_v2",
    "crmhd_v2-8pc-b1-diode-lngrad_out": "crmhd",
    "mhd-16pc-b1-diode": "mhd_v1",
    "mhd-16pc-b1-lngrad_out": "mhd_lngrad",
    "mhd_v2-16pc-b1-diode": "mhd_v2",
    "mhd_v2-8pc-b1-diode": "mhd",
    "crmhd-16pc-b0.1-diode-lngrad_out": "b0.1",
    "crmhd-16pc-b1-diode-lngrad_out-Vmax10": "Vmax10",
    "crmhd-16pc-b1-lngrad_out-lngrad_out-sigma27": "σ27",
    "crmhd-16pc-b1-lngrad_out-lngrad_out-sigma29": "σ29",
    "crmhd-16pc-b1-diode-lngrad_out-sigma28": "σ28",
    "crmhd-16pc-b1-diode-lngrad_out-sigma27_va0": "σ27_vA",
    "crmhd-16pc-b1-diode-lngrad_out-sigma28_va0": "σ28_vA",
    "crmhd-16pc-b1-diode-lngrad_out-sigma29_va0": "σ29_vA",
    "crmhd-16pc-b1-diode-lngrad_out-sigma27_va1": "σ27_vAi",
    "crmhd-16pc-b1-diode-lngrad_out-sigma28_va1": "σ28_vAi",
    "crmhd-16pc-b1-diode-lngrad_out-sigma29_va1": "σ29_vAi",
    "crmhd-16pc-b10-diode-lngrad_out": "b10",
    "crmhd-16pc-tallbox-b1-diode-lngrad_out": "tall",
    "crmhd-16pc-tallbox-b1-diode-diode": "tall-diode",
    "crmhd-16pc-fullgrav-b1-diode-lngrad_out": "fullgrav",
}

model_color = {
    "crmhd-16pc-b1-diode-lngrad_out": "royalblue",
    "crmhd_v2-16pc-b1-diode-lngrad_out": "cornflowerblue",
    "crmhd_v2-8pc-b1-diode-lngrad_out": "midnightblue",
    "crmhd-16pc-b1-diode-lngrad_out-sigma28": "gold",
    "mhd-16pc-b1-diode": "crimson",
    "mhd-16pc-b1-lngrad_out": "salmon",
    "mhd_v2-16pc-b1-diode": "salmon",
    "mhd_v2-8pc-b1-diode": "maroon",
    "crmhd-16pc-b0.1-diode-lngrad_out": "turquoise",
    "crmhd-16pc-b1-diode-lngrad_out-Vmax10": "orchid",
    "crmhd-16pc-b1-lngrad_out-lngrad_out-sigma27": "sienna",
    "crmhd-16pc-b1-lngrad_out-lngrad_out-sigma29": "darkorange",
    "crmhd-16pc-b1-diode-lngrad_out-sigma27_va0": "sienna",
    "crmhd-16pc-b1-diode-lngrad_out-sigma28_va0": "gold",
    "crmhd-16pc-b1-diode-lngrad_out-sigma29_va0": "darkorange",
    "crmhd-16pc-b1-diode-lngrad_out-sigma27_va1": "sienna",
    "crmhd-16pc-b1-diode-lngrad_out-sigma28_va1": "gold",
    "crmhd-16pc-b1-diode-lngrad_out-sigma29_va1": "darkorange",
    "crmhd-16pc-b10-diode-lngrad_out": "teal",
    "crmhd-16pc-tallbox-b1-diode-lngrad_out": "indigo",
    "crmhd-16pc-tallbox-b1-diode-diode": "violet",
    "crmhd-16pc-fullgrav-b1-diode-lngrad_out": "orange",
}

model_default = [
    "crmhd_v2-8pc-b1-diode-lngrad_out",
    "mhd_v2-8pc-b1-diode",
]

model_crmhd_compare = [
    "crmhd-16pc-b1-diode-lngrad_out",
    "crmhd_v2-16pc-b1-diode-lngrad_out",
    "crmhd_v2-8pc-b1-diode-lngrad_out",
]

model_mhd_compare = [
    "mhd-16pc-b1-diode",
    "mhd_v2-16pc-b1-diode",
    "mhd_v2-8pc-b1-diode",
]

model_beta = [
    "crmhd-16pc-b1-diode-lngrad_out",
    "crmhd-16pc-b0.1-diode-lngrad_out",
    "crmhd-16pc-b10-diode-lngrad_out",
    "crmhd-16pc-b1-diode-lngrad_out-Vmax10",
]
model_sigma = [
    "crmhd-16pc-b1-diode-lngrad_out",
    "crmhd-16pc-b1-diode-lngrad_out-sigma27_va1",
    "crmhd-16pc-b1-diode-lngrad_out-sigma28_va1",
    "crmhd-16pc-b1-diode-lngrad_out-sigma29_va1",
    "crmhd-16pc-b1-diode-lngrad_out-sigma27_va0",
    "crmhd-16pc-b1-diode-lngrad_out-sigma28_va0",
    "crmhd-16pc-b1-diode-lngrad_out-sigma29_va0",
    # "crmhd-16pc-b1-lngrad_out-lngrad_out-sigma27",
    # "crmhd-16pc-b1-lngrad_out-lngrad_out-sigma29",
    # "crmhd-16pc-b1-diode-lngrad_out-sigma28",
]
model_tall = [
    "crmhd-16pc-b1-diode-lngrad_out",
    "crmhd-16pc-tallbox-b1-diode-lngrad_out",
    "crmhd-16pc-tallbox-b1-diode-diode",
    "crmhd-16pc-fullgrav-b1-diode-lngrad_out",
]

outdir = "./figures"


def get_model_table_line(s):
    par = s.par
    prob = s.par["problem"]
    mesh = s.par["mesh"]

    # varied parameters
    beta = prob["beta0"]
    Lz = mesh["x3max"] - mesh["x3min"]
    dx = int(Lz) / int(mesh["nx3"])
    mhdbc = mesh["mhd_outflow_bc"]
    grav = "skipped" if par["gravity"]["solve_grav_hyperbolic_dt"] == "true" else "full"

    physics = (
        "crmhd" if par["configure"]["Cosmic_Ray_Transport"] == "Multigroups" else "mhd"
    )

    if physics == "crmhd":
        cr = s.par["cr"]
        crbc = mesh["cr_outflow_bc"]
        vmax = f"{cr['vmax'] / 1.0e9:<5.0f}"
        sigma = f"{cr['sigma']}" if cr["self_consistent_flag"] == 0 else "full"
    else:
        crbc = "\\nodata"
        vmax = "\\nodata"
        sigma = "\\nodata"
    name = s.basename.replace("mhdbc_", "").replace("crbc_", "").replace("-icpx", "")
    try:
        name = model_name[name]
    except KeyError:
        pass
    line = (
        f"{name:<20s} & {physics:<10s} & {beta:<5.1f} & {Lz:<10.0f} & {dx:<5.0f} & "
        f"{mhdbc:<10s} & {crbc:<10s} & {vmax:<10s} & {sigma:<10s} & {grav:<10s} \\\\"
    )

    if beta == 0.1:
        group = "b0.1"
    elif beta == 10:
        group = "b10"
    else:
        if physics == "mhd":
            group = "mhd"
        elif sigma != "full":
            group = "sigma"
        elif cr["vmax"] / 1.0e9 == 10:
            group = "vmax"
        elif Lz != 8192:
            group = "tall"
        # elif mhdbc == "diode":
        # group = "diode"
        # elif mhdbc == "lngrad_out":
        # group = "lngrad"
        else:
            group = "bcs"
    return name, line, group


def cr_data_load(
    basedir="/scratch/gpfs/EOST/changgoo/tigress_classic/", pattern="*mhd*"
):
    folders = sorted(glob.glob(osp.join(basedir, pattern)))
    icolor = 0
    model_dict = dict()
    for folder in folders:
        name = os.path.basename(folder)
        name = name.replace("mhdbc_", "").replace("crbc_", "").replace("-icpx", "")
        model_dict[name] = folder
        if name not in model_color:
            print(name)
            model_color[name] = f"C{icolor}"
            model_name[name] = name
            icolor += 1

    return model_dict


def cr_data_group(sims):
    sim_group = dict()
    sim_group["default"] = dict()
    for m in sims.models:
        s = sims.set_model(m)
        name, line, group = get_model_table_line(s)
        if group not in sim_group:
            sim_group[group] = dict()
        sim_group[group][name] = s
        if name in ["crmhd-b1-diode-lngrad_out"]:
            sim_group["default"][name] = s

    return sim_group


def load_group(sim_group, group="default"):
    sg = sim_group[group]

    for name, s in sg.items():
        print(f"loading {name}...")
        if not hasattr(s, "zprof"):
            s.zprof = s.load_zprof()
        s.zp_ph = xr.concat(
            [
                s.zprof.sel(phase=["CNM", "UNM", "WNM"])
                .sum(dim="phase")
                .assign_coords(phase="wc"),
                s.zprof.sel(phase=["WHIM", "HIM"])
                .sum(dim="phase")
                .assign_coords(phase="hot"),
            ],
            dim="phase",
        )
        if not hasattr(s, "hst"):
            s.hst = s.read_hst()


def create_windpdf(simgroup, gr):
    for m, s in simgroup[gr].items():
        outpdf = []
        inpdf = []
        for num in s.nums:
            pdf = s.get_windpdf(num, "windpdf")
            outpdf.append(pdf["out"])
            inpdf.append(pdf["in"])
        outdir = os.path.join(s.savdir, "windpdf")
        xr.concat(outpdf, dim="time").to_netcdf(os.path.join(outdir, "outpdf.nc"))
        xr.concat(inpdf, dim="time").to_netcdf(os.path.join(outdir, "inpdf.nc"))
        load_windpdf(s)


def load_windpdf(s):
    outdir = os.path.join(s.savdir, "windpdf")
    with xr.open_dataarray(os.path.join(outdir, "outpdf.nc")) as da:
        s.outpdf = (
            da.sel(flux=["mflux", "eflux", "mZflux"])
            .sel(time=slice(150, 500))
            .mean(dim="time")
        )
    with xr.open_dataarray(os.path.join(outdir, "inpdf.nc")) as da:
        s.inpdf = (
            da.sel(flux=["mflux", "eflux", "mZflux"])
            .sel(time=slice(150, 500))
            .mean(dim="time")
        )
    zfc = np.linspace(s.domain["le"][2], s.domain["re"][2], s.domain["Nx"][2] + 1)
    zcc = 0.5 * (zfc[1:] + zfc[:-1])
    dnz = len(zcc[(zcc > 950) & (zcc < 1050)])
    Zsn = s.par["feedback"]["Z_SN"]
    Mej = s.par["feedback"]["M_ej"]
    dt = 0.1
    mstar = 1 / np.sum(s.pop_synth["snrate"] * dt)
    field = "sfr40"
    h = s.read_hst()
    sfr_avg = h[field].loc[150:].mean()
    sfr_std = h[field].loc[150:].std()
    ref_flux = dict(
        mflux=sfr_avg / mstar * mstar,
        eflux=sfr_avg / mstar * 1.0e51,
        mZflux=sfr_avg / mstar * Mej * Zsn,
    )
    ref_flux = xr.Dataset(ref_flux).to_array("flux")
    s.outflux = s.outpdf / np.prod(s.domain["Nx"][:-1]) / dnz

    s.influx = s.inpdf / np.prod(s.domain["Nx"][:-1]) / dnz


def plot_zprof_mean_quantile(ydata, quantile=True, **kwargs):
    q = ydata.quantile([0.16, 0.5, 0.84], dim="time")
    qmean = ydata.mean(dim="time")

    plt.plot(q.z / 1.0e3, qmean, **kwargs)
    if quantile:
        plt.fill_between(
            q.z / 1.0e3,
            q.sel(quantile=0.16),
            q.sel(quantile=0.84),
            color=kwargs["color"],
            alpha=0.2,
            linewidth=0,
        )


def plot_zprof_quantile(ydata, quantile=True, **kwargs):
    q = ydata.quantile([0.16, 0.5, 0.84], dim="time")

    plt.plot(q.z / 1.0e3, q.sel(quantile=0.5), **kwargs)
    if quantile:
        plt.fill_between(
            q.z / 1.0e3,
            q.sel(quantile=0.16),
            q.sel(quantile=0.84),
            color=kwargs["color"],
            alpha=0.2,
            linewidth=0,
        )


def plot_zprof(zprof, field, ph, norm=1.0, line="median", quantile=True, **kwargs):
    ydata = zprof[field].sel(phase=ph) / norm
    if ydata.phase.size > 1:
        ydata = ydata.sum(dim=["phase"])
    if line == "median":
        plot_zprof_quantile(ydata, quantile=quantile, **kwargs)
    elif line == "mean":
        plot_zprof_mean_quantile(ydata, quantile=quantile, **kwargs)


def plot_zprof_field(zprof, field, ph, line="median", quantile=True, **kwargs):
    ydata = zprof[field].sel(phase=ph)
    area = zprof["area"].sel(phase=ph)
    if ydata.phase.size > 1:
        ydata = ydata.sum(dim=["phase"]) / area.sum(dim=["phase"])
    else:
        ydata = ydata / area
    if line == "median":
        plot_zprof_quantile(ydata, quantile=quantile, **kwargs)
    elif line == "mean":
        plot_zprof_mean_quantile(ydata, quantile=quantile, **kwargs)


def plot_zprof_frac(
    zprof, field, ph, denominator="area", line="median", quantile=True, **kwargs
):
    if "whole" in zprof.phase:
        area = zprof[denominator].sel(phase="whole")
    else:
        area = zprof[denominator].sum(dim="phase")
    ydata = zprof[field].sel(phase=ph)
    if ydata.phase.size > 1:
        ydata = ydata.sum(dim=["phase"])
    if line == "median":
        plot_zprof_quantile(ydata / area, quantile=quantile, **kwargs)
    elif line == "mean":
        plot_zprof_mean_quantile(ydata / area, quantile=quantile, **kwargs)


def print_sim_table(sims):
    line = (
        f"{'name':<40s} & {'physics':<10s} & {'beta':<5.0s} & {'Lz':<10.0s} & {'dx':<5.0s} & "
        f"{'mhdbc':<10s} & {'crbc':<10s} & {'vmax':<10s} & {'sigma':<10s} & {'grav':<10s} \\\\"
    )
    print(line)
    for m in sims.models:
        s = sims.set_model(m)
        name, line, group = get_model_table_line(s)
        print(line + f" -- {group}")


def plot_massflux_tz(simgroup, gr, ph="wc"):
    with plt.style.context({"axes.grid": False}):
        sims = simgroup[gr]
        models = list(sims.keys())
        nmodels = len(models)
        fig, axes = plt.subplots(
            2,
            nmodels,
            figsize=(3 * nmodels, 5),
            sharey=True,
            sharex=True,
            constrained_layout=True,
        )
        for m, axs in zip(models, axes.T):
            s = sims[m]

            area = np.prod(s.domain["Lx"][:-1])
            units = s.u.Msun / s.u.pc**2 / s.u.Myr
            mflux_out = s.zp_ph["mom3"].sel(vz_dir=1).sel(phase=ph).T * units / area
            mflux_in = -s.zp_ph["mom3"].sel(vz_dir=-1).sel(phase=ph).T * units / area
            plt.sca(axs[0])
            im_out = plt.pcolormesh(
                s.zp_ph.time * s.u.Myr,
                s.zp_ph.z.sel(z=slice(0, s.zp_ph.z.max())) / 1.0e3,
                mflux_out.sel(z=slice(0, s.zp_ph.z.max())),
                cmap=cmr.ember,
                norm=LogNorm(1.0e-5, 1.0e-1),
            )
            im_out = plt.pcolormesh(
                s.zp_ph.time * s.u.Myr,
                s.zp_ph.z.sel(z=slice(s.zp_ph.z.min(), 0)) / 1.0e3,
                mflux_in.sel(z=slice(s.zp_ph.z.min(), 0)),
                cmap=cmr.ember,
                norm=LogNorm(1.0e-5, 1.0e-1),
            )
            plt.title(model_name[m], color=model_color[m])
            plt.sca(axs[1])
            im_in = plt.pcolormesh(
                s.zp_ph.time * s.u.Myr,
                s.zp_ph.z.sel(z=slice(0, s.zp_ph.z.max())) / 1.0e3,
                mflux_in.sel(z=slice(0, s.zp_ph.z.max())),
                cmap=cmr.cosmic,
                norm=LogNorm(1.0e-5, 1.0e-1),
            )
            im_in = plt.pcolormesh(
                s.zp_ph.time * s.u.Myr,
                s.zp_ph.z.sel(z=slice(s.zp_ph.z.min(), 0)) / 1.0e3,
                mflux_out.sel(z=slice(s.zp_ph.z.min(), 0)),
                cmap=cmr.cosmic,
                norm=LogNorm(1.0e-5, 1.0e-1),
            )
        plt.setp(axes[:, 0], ylabel=r"$z\, [{\rm kpc}]$")
        plt.setp(axes[1, :], xlabel=r"$t\, [{\rm Myr}]$")
        # plt.ylim(bottom=0)
        cbar_out = plt.colorbar(
            im_out,
            shrink=0.8,
            ax=axes[0, :],
            pad=0.02,
            label=f"$\\mathcal{{F}}_{{M}}^{{\\rm {ph},out}}$"
            r"$\,[{\rm M_\odot\,kpc^{-2}\,yr}]$",
        )
        cbar_in = plt.colorbar(
            im_in,
            shrink=0.8,
            ax=axes[1, :],
            pad=0.02,
            label=f"$\\mathcal{{F}}_{{M}}^{{\\rm {ph},in}}$"
            r"$\,[{\rm M_\odot\,kpc^{-2}\,yr}]$",
        )
        plt.savefig(osp.join(outdir, f"{gr}_massflux_tz.png"))


def plot_flux_tz(simgroup, gr):
    with plt.style.context({"axes.grid": False}):
        sims = simgroup[gr]
        models = list(sims.keys())
        nmodels = len(models)
        fig, axes = plt.subplots(
            4,
            nmodels,
            figsize=(4 * nmodels, 8),
            sharey=True,
            sharex=True,
            constrained_layout=True,
        )
        norm = dict(
            mflux=LogNorm(1.0e-5, 1.0e-1),
            pflux_MHD=LogNorm(1.0e-3, 10),
            eflux_MHD=LogNorm(1.0e42, 1.0e47),
            pflux_CR=LogNorm(1.0e-3, 10),
            eflux_CR=LogNorm(1.0e42, 1.0e47),
        )
        label_unit = dict(
            mflux=r"$\,[{\rm M_\odot\,kpc^{-2}\,yr}]$",
            pflux_MHD=r"$\,[{\rm M_\odot\,km/s\,kpc^{-2}\,yr}]$",
            eflux_MHD=r"$\,[{\rm erg\,kpc^{-2}\,yr}]$",
        )
        cmap_outin = [cmr.ember, cmr.cosmic]
        for m, axs in zip(models, axes.T):
            s = sims[m]

            dset_outin = []
            for vz_dir in [1, -1]:
                dset_upper = (
                    s.zp_ph.sel(vz_dir=vz_dir).sel(z=slice(0, s.domain["re"][2]))
                    * vz_dir
                )
                dset_lower = s.zp_ph.sel(vz_dir=-vz_dir).sel(
                    z=slice(s.domain["le"][2], 0)
                ) * (-vz_dir)
                dset = xr.concat([dset_lower, dset_upper], dim="z")
                dset_outin.append(update_stress(s, dset))
            flux_field = "mflux"
            ph = "wc"
            im_outin = []
            for ax, flux, cmap in zip(axs, dset_outin, cmap_outin):
                plt.sca(ax)
                if flux_field not in flux:
                    continue
                im = plt.pcolormesh(
                    flux.time * s.u.Myr,
                    flux.z / 1.0e3,
                    flux[flux_field].sel(phase=ph).T,
                    cmap=cmap,
                    norm=norm[flux_field],
                )
                im_outin.append(im)
            flux_field = "eflux_MHD"
            flux = dset_outin[0]
            cmap = cmr.sunburst
            for ax, ph in zip(axs[2:], ["wc", "hot"]):
                plt.sca(ax)
                if flux_field not in flux:
                    continue
                im = plt.pcolormesh(
                    flux.time * s.u.Myr,
                    flux.z / 1.0e3,
                    flux[flux_field].sel(phase=ph).T,
                    cmap=cmap,
                    norm=norm[flux_field],
                )
                im_outin.append(im)
            plt.sca(axs[0])
            plt.title(model_name[m], color=model_color[m])
        plt.setp(axes[:, 0], ylabel=r"$z\, [{\rm kpc}]$")
        plt.setp(axes[-1, :], xlabel=r"$t\, [{\rm Myr}]$")
        for axs, im, lab, flux_field, ph in zip(
            axes,
            im_outin,
            ["out", "in", "out", "out"],
            ["mflux", "mflux", "eflux_MHD", "eflux_MHD"],
            ["wc", "wc", "wc", "hot"],
        ):
            flux_name = flux_field.replace("_", ",").replace("flux", "").upper()
            plt.colorbar(
                im,
                shrink=0.8,
                ax=axs,
                pad=0.02,
                label=f"$\\mathcal{{F}}_{{{flux_name[0]}{{\\rm {flux_name[1:]}}}}}^{{\\;\\rm {ph},{lab}}}$"
                + label_unit[flux_field],
            )
        plt.savefig(osp.join(outdir, f"{gr}_flux_tz.png"))


def plot_pressure_tz(simgroup, gr, ph="wc"):
    with plt.style.context({"axes.grid": False}):
        sims = simgroup[gr]
        models = list(sims.keys())
        nmodels = len(models)
        fig, axes = plt.subplots(
            4,
            nmodels,
            figsize=(3 * nmodels, 10),
            sharey=True,
            sharex=True,
            constrained_layout=True,
        )
        imlist = []
        for m, axs in zip(models, axes.T):
            s = sims[m]
            area = np.prod(s.domain["Lx"][:-1])
            units = s.u.pok
            dset = s.zp_ph.sum(dim="vz_dir").sel(phase=ph)
            area = dset["area"]
            if "Pi_B" not in dset:
                dset["Pi_B"] = (dset["Pmag1"] + dset["Pmag2"] - dset["Pmag3"]) * s.u.pok
            if "Pok_B" not in dset:
                dset["Pok_B"] = (
                    dset["Pmag1"] + dset["Pmag2"] + dset["Pmag3"]
                ) * s.u.pok
            if "Pok_cr" not in dset and s.options["cosmic_ray"]:
                dset["Pok_cr"] = dset["0-Ec"] / 3.0 * s.u.pok
            if "Pok_th" not in dset:
                dset["Pok_th"] = dset["press"] * s.u.pok
            if "Pok_kin" not in dset:
                dset["Pok_kin"] = dset["Pturbz"] * s.u.pok
            for ax, pfield in zip(axs, ["Pok_cr", "Pok_th", "Pok_kin", "Pok_B"]):
                plt.sca(ax)
                if pfield in dset:
                    im = plt.pcolormesh(
                        dset.time * s.u.Myr,
                        dset.z / 1.0e3,
                        dset[pfield].T / area,
                        cmap=plt.cm.plasma,
                        norm=LogNorm(1.0e1, 5.0e4),
                    )
                    imlist.append(im)
                    if pfield == "Pok_cr":
                        plt.title(model_name[m])
                else:
                    plt.axis("off")
                    imlist.append(None)

        plt.setp(axes[:, 0], ylabel=r"$z\, [{\rm kpc}]$")
        plt.setp(axes[-1, :], xlabel=r"$t\, [{\rm Myr}]$")
        for im, axs, pfield in zip(
            imlist, axes, ["Pok_cr", "Pok_th", "Pok_kin", "Pok_B"]
        ):
            if im is None:
                continue
            lab = pfield.split("_")[-1]
            cbar = plt.colorbar(
                im,
                shrink=0.8,
                ax=axs,
                pad=0.02,
                label=f"$P_{{\\rm {lab} }}/k_B$" + r"$\,[{\rm cm^{-3}\,K}]$",
            )
        plt.savefig(osp.join(outdir, f"{gr}_pressures_tz.png"))


def plot_pressure_z(simgroup, gr, ph="wc"):
    sims = simgroup[gr]
    models = list(sims.keys())
    fig, axes = plt.subplots(1, 4, figsize=(8, 3), sharey=True, constrained_layout=True)
    for i, m in enumerate(models):
        s = sims[m]
        c = model_color[m]
        dset = s.zp_ph.sel(time=slice(150, 500)).sum(dim="vz_dir")
        dset = update_stress(s, dset)
        for ax, pfield in zip(axes, ["Pok_cr", "Pok_th", "Pok_kin", "Pi_B"]):
            plt.sca(ax)
            if pfield in dset:
                plot_zprof_field(dset, pfield, ph, color=c, label=model_name[m])

            lab = pfield.split("_")[-1]
            if pfield.startswith("Pok"):
                plt.title(f"$P_{{\\rm {lab}}}$")
            else:
                plt.title(f"$\\Pi_{{\\rm {lab}}}$")
    plt.sca(axes[0])
    plt.ylabel(r"$\overline{P}^{\rm \; wc}(z)/k_B\,[{\rm cm^{-3}\,K}]$")
    plt.setp(axes, "yscale", "log")
    plt.setp(axes, "xlabel", r"$z\,[{\rm kpc}]$")
    plt.setp(axes, "ylim", (10, 1.0e5))
    plt.setp(axes, "xlim", (-4, 4))
    for ax in axes:
        plt.sca(ax)
        plt.axvline(1, ls="--", color="k", lw=1)
        plt.axvline(-1, ls="--", color="k", lw=1)
    plt.sca(axes[1])
    plt.legend(fontsize="small")
    plt.savefig(osp.join(outdir, f"{gr}_pressures_z.pdf"))


def plot_volume_fraction_z(simgroup, gr):
    sims = simgroup[gr]
    models = list(sims.keys())
    nmodels = len(models)
    fig, axes = plt.subplots(
        1,
        nmodels,
        figsize=(4 * nmodels, 3),
        sharey=True,
        sharex=True,
        constrained_layout=True,
    )
    for i, m in enumerate(models):
        s = sims[m]

        dset = s.zprof.sum(dim="vz_dir").sel(time=slice(150, 500))
        plt.sca(axes[i])
        for ph, color, label in zip(
            [["CNM", "UNM"], "WNM", "WHIM", "HIM"],
            ["C0", "limegreen", "gold", "C3"],
            ["CNM+UNM", "WNM", "WHIM", "HIM"],
        ):
            plot_zprof_frac(dset, "area", ph, color=color, label=label)
        plt.title(model_name[m])
    plt.sca(axes[0])
    plt.legend()
    plt.ylim(0, 1)
    plt.xlabel(r"$z\,[{\rm kpc}]$")
    plt.ylabel(r"Volume Fraction")
    plt.savefig(osp.join(outdir, f"{gr}_volume_fraction.pdf"))


def plot_profile_frac_z(
    simgroup, gr, vz_dir=None, field="rho", line="median", savefig=True
):
    sims = simgroup[gr]
    models = list(sims.keys())
    nmodels = len(models)
    fig, axes = plt.subplots(
        1,
        nmodels,
        figsize=(4 * nmodels, 3),
        sharey=True,
        sharex=True,
        constrained_layout=True,
    )
    for i, m in enumerate(models):
        s = sims[m]
        if vz_dir is None:
            dset = s.zprof.sum(dim="vz_dir").sel(time=slice(150, 500))
        else:
            dset_upper = (
                s.zprof.sel(vz_dir=vz_dir)
                .sel(time=slice(150, 500))
                .sel(z=slice(0, s.domain["re"][2]))
            )
            dset_lower = (
                s.zprof.sel(vz_dir=-vz_dir)
                .sel(time=slice(150, 500))
                .sel(z=slice(s.domain["le"][2], 0))
            )
            dset = xr.concat([dset_lower, dset_upper], dim="z")
        plt.sca(axes[i])
        for ph, color, label in zip(
            [["CNM", "UNM"], "WNM", "WHIM", "HIM"],
            ["C0", "limegreen", "gold", "C3"],
            ["CNM+UNM", "WNM", "WHIM", "HIM"],
        ):
            plot_zprof_frac(dset, field, ph, line=line, color=color, label=label)

        plt.title(model_name[m])
    plt.sca(axes[0])
    plt.legend()
    # plt.ylim(0, 1)
    plt.xlim(-4, 4)
    plt.yscale("log")
    plt.ylabel(r"$\langle q \rangle$")
    plt.setp(axes, "xlabel", r"$z\,[{\rm kpc}]$")
    if savefig:
        plt.savefig(osp.join(outdir, f"{gr}_{field}_profile_frac_z.pdf"))


def plot_profile_z(simgroup, gr, field="rho", savefig=True):
    sims = simgroup[gr]
    models = list(sims.keys())
    nmodels = len(models)
    fig, axes = plt.subplots(
        1,
        nmodels,
        figsize=(4 * nmodels, 3),
        sharey=True,
        sharex=True,
        constrained_layout=True,
    )
    for i, m in enumerate(models):
        s = sims[m]

        dset = s.zprof.sum(dim="vz_dir").sel(time=slice(150, 500))
        plt.sca(axes[i])
        for ph, color, label in zip(
            [["CNM", "UNM"], "WNM", "WHIM", "HIM"],
            ["C0", "limegreen", "gold", "C3"],
            ["CNM+UNM", "WNM", "WHIM", "HIM"],
        ):
            plot_zprof_field(dset, field, ph, color=color, label=label)

        plt.title(model_name[m])
    plt.sca(axes[0])
    plt.legend()
    # plt.ylim(0, 1)
    plt.xlim(-4, 4)
    plt.yscale("log")
    plt.ylabel(r"$\overline{q}$")
    plt.setp(axes, "xlabel", r"$z\,[{\rm kpc}]$")
    if savefig:
        plt.savefig(osp.join(outdir, f"{gr}_{field}_profile_z.pdf"))


def plot_fraction_z(simgroup, gr, field="rho"):
    sims = simgroup[gr]
    models = list(sims.keys())
    nmodels = len(models)
    fig, axes = plt.subplots(
        1,
        nmodels,
        figsize=(4 * nmodels, 3),
        sharey=True,
        sharex=True,
        constrained_layout=True,
    )
    for i, m in enumerate(models):
        s = sims[m]

        dset = s.zprof.sum(dim="vz_dir").sel(time=slice(150, 500))
        plt.sca(axes[i])
        for ph, color, label in zip(
            [["CNM", "UNM"], "WNM", "WHIM", "HIM"],
            ["C0", "limegreen", "gold", "C3"],
            ["CNM+UNM", "WNM", "WHIM", "HIM"],
        ):
            plot_zprof_frac(
                dset, field, ph, denominator=field, color=color, label=label
            )

        plt.title(model_name[m])
    plt.sca(axes[0])
    plt.legend()
    plt.ylim(0, 1)
    plt.xlabel(r"$z\,[{\rm kpc}]$")
    plt.ylabel("Fraction")
    plt.savefig(osp.join(outdir, f"{gr}_{field}_fraction.pdf"))


def plot_fraction_ph_z(simgroup, gr, field="rho"):
    sims = simgroup[gr]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8, 3),
        sharey=True,
        sharex=True,
        constrained_layout=True,
    )
    for m, s in sims.items():
        s = sims[m]
        name = model_name[m]
        color = model_color[m]
        dset = s.zp_ph.sum(dim="vz_dir").sel(time=slice(150, 500))
        for ax, ph in zip(axes, ["wc", "hot"]):
            plt.sca(ax)
            plot_zprof_frac(dset, field, ph, denominator=field, color=color, label=name)

            plt.title(ph)
            plt.xlabel(r"$z\,[{\rm kpc}]$")
            plt.xlim(-4, 4)
    plt.sca(axes[0])
    plt.legend()
    plt.ylim(0, 1)
    plt.ylabel("Fraction")
    plt.savefig(osp.join(outdir, f"{gr}_{field}_fraction.pdf"))


def plot_mass_fraction_t(simgroup, gr):
    sims = simgroup[gr]
    models = list(sims.keys())
    nmodels = len(models)
    fig, axes = plt.subplots(
        1,
        nmodels,
        figsize=(3 * nmodels, 4),
        sharey=True,
        sharex=True,
        constrained_layout=True,
    )
    for i, m in enumerate(models):
        s = sims[m]

        dset = s.zprof.sum(dim="vz_dir").sel(z=slice(-50, 50))
        plt.sca(axes[i])
        total_mass = dset["rho"].sel(phase="whole").sum(dim="z")
        for ph, color, label in zip(
            [["CNM", "UNM"], "WNM", "WHIM", "HIM"],
            ["C0", "C1", "C3"],
            ["Cold", "Warm", "Hot"],
        ):
            frac = dset["rho"].sel(phase=ph).sum(dim="phase").sum(dim="z") / total_mass
            plt.plot(dset.time * s.u.Myr, frac, color=color, label=label)
        plt.title(model_name[m])
    plt.sca(axes[0])
    plt.legend()
    plt.ylim(0, 1)
    plt.xlabel(r"$t\,[{\rm Myr}]$")
    plt.ylabel(r"Mass Fraction")
    plt.savefig(osp.join(outdir, f"{gr}_mass_fraction_t.pdf"))


def plot_rho_z(simgroup, gr):
    sims = simgroup[gr]
    models = list(sims.keys())
    nmodels = len(models)

    fig, axes = plt.subplots(
        4, 2, figsize=(8, 10), sharex=True, sharey="row", constrained_layout=True
    )
    for i, m in enumerate(models):
        s = sims[m]
        color = model_color[m]
        dset = s.zp_ph.sel(time=slice(150, 500)).sum(dim="vz_dir")
        dset = update_stress(s, dset)
        # dset["Etot"] *= (s.u.energy_density/ac.k_B).cgs.value
        for axs, ph in zip(axes.T, ["wc", "hot"]):
            for ax, field in zip(axs, ["rho", "Pok_kin", "Pok_th", "Pok_B"]):
                plt.sca(ax)
                plot_zprof_frac(dset, field, ph, color=color, label=model_name[m])
                plot_zprof_field(dset, field, ph, color=color, lw=1)
                plt.yscale("log")
            # plt.ylim(1.0e-5, 10)
            # plt.title(f"phase={ph}")
    plt.sca(axes[0, 0])
    plt.ylabel(r"$\langle n_H \rangle\,[{\rm cm^{-3}}]$")
    plt.xlim(-4, 4)
    plt.ylim(bottom=1.0e-5)
    plt.sca(axes[1, 0])
    plt.ylabel(r"$\langle \mathcal{E}_{\rm tot} \rangle\,[{\rm erg\,cm^{-3}}]$")
    # plt.ylim(bottom=1.e-2)
    plt.setp(axes[1, :], "xlabel", r"$z\,[{\rm kpc}]$")
    plt.savefig(osp.join(outdir, f"{gr}_rho_z.pdf"))


def plot_rhov_z(simgroup, gr):
    sims = simgroup[gr]
    models = list(sims.keys())
    nmodels = len(models)

    fig, axes = plt.subplots(4, 1, figsize=(5, 8), sharex=True, constrained_layout=True)
    for i, m in enumerate(models):
        s = sims[m]
        c = model_color[m]
        dset = s.zp_ph.sel(time=slice(150, 500)).sum(dim="vz_dir")

        plt.sca(axes[0])
        ph = "wc"
        plot_zprof_field(dset, "rho", ["wc", "hot"], color=c, label=model_name[m])
        plt.yscale("log")
        plt.ylim(1.0e-3, 10)
        plt.ylabel(r"$\langle \rho \rangle$")

        plt.legend()

        plt.sca(axes[1])
        plot_zprof_field(dset, "vel3", ph, color=c)
        plt.ylabel(r"$\langle v_z \rangle_{\rm wc}$")

        plt.sca(axes[2])
        if s.options["cosmic_ray"]:
            plot_zprof_field(dset, "0-Vs3", ph, color=c)
        plt.ylabel(r"$\langle v_{A,i} \rangle_{\rm wc}$")

        plt.sca(axes[3])
        if s.options["cosmic_ray"]:
            plot_zprof_field(dset, "0-Vd3", ph, color=c)
        plt.xlabel("z")
        plt.ylabel(r"$\langle v_d \rangle_{\rm wc}$")

    plt.setp(axes[1:], "ylim", (-50, 50))
    plt.savefig(osp.join(outdir, f"{gr}_rho_vz.pdf"))


def update_stress(s, dset):
    # stresses
    dset["Pok_th"] = dset["press"] * s.u.pok
    dset["Pok_kin"] = dset["Pturbz"] * s.u.pok
    dset["Pok_tot"] = dset["Pok_th"] + dset["Pok_kin"]
    if s.options["mhd"]:
        dset["Pi_B"] = (dset["Pmag1"] + dset["Pmag2"] - dset["Pmag3"]) * s.u.pok
        dset["Pok_B"] = (dset["Pmag1"] + dset["Pmag2"] + dset["Pmag3"]) * s.u.pok
        dset["Pok_tot"] += dset["Pi_B"]

    # fluxes
    # total area
    area = np.prod(s.domain["Lx"][:-1])
    mflux_units = (s.u.density * s.u.velocity).to("Msun/(kpc**2*yr)").value
    pflux_units = (s.u.pressure).to("(Msun*km)/(kpc**2*yr*s)").value
    eflux_units = (s.u.energy_density * s.u.velocity).to("(erg)/(kpc**2*yr)").value
    dset["mflux"] = dset["mom3"] / area * mflux_units
    dset["pflux_MHD"] = (dset["Pturbz"] + dset["press"]) / area * pflux_units
    dset["eflux_MHD"] = (
        (
            dset["Ekin_flux1"]
            + dset["Ekin_flux2"]
            + dset["Ekin_flux3"]
            + dset["Eth_flux"]
        )
        / area
        * eflux_units
    )
    if s.options["mhd"]:
        dset["pflux_MHD"] += (
            (dset["Pmag1"] + dset["Pmag2"] - dset["Pmag3"]) / area * pflux_units
        )
        dset["eflux_MHD"] += (dset["Sz_Bpress"] + dset["Sz_Btens"]) / area * eflux_units
    if s.options["cosmic_ray"]:
        vmax_kms = s.par["cr"]["vmax"] / 1.0e5
        dset["Pok_cr"] = dset["0-Ec"] / 3.0 * s.u.pok
        dset["pflux_CR"] = dset["0-Ec"] / 3.0 / area * pflux_units
        dset["eflux_CR"] = dset["0-Fc3"] * vmax_kms / area * eflux_units

    # weights
    gzext = np.interp(dset.z, s.extgrav["z"], s.extgrav["gz"])
    dz = s.domain["dx"][2]
    Wsg_from_bot = (dset["rhogz"] * dz).cumsum(dim="z") * s.u.pok
    Wsg_from_top = (-dset["rhogz"] * dz).sel(z=slice(None, None, -1)).cumsum(
        dim="z"
    ).sel(z=slice(None, None, -1)) * s.u.pok
    # Wsg_mean = 0.5 * (Wsg_from_bot + Wsg_from_top)
    dset["Wsg"] = xr.concat(
        (
            Wsg_from_bot.sel(z=slice(s.domain["le"][2], 0)),
            Wsg_from_top.sel(z=slice(0, s.domain["re"][2])),
        ),
        dim="z",
    )
    Wext_from_bot = (dset["rho"] * gzext * dz).cumsum(dim="z") * s.u.pok
    Wext_from_top = (-dset["rho"] * gzext * dz).sel(z=slice(None, None, -1)).cumsum(
        dim="z"
    ).sel(z=slice(None, None, -1)) * s.u.pok
    # Wext_mean = 0.5 * (Wext_from_bot + Wext_from_top)
    dset["Wext"] = xr.concat(
        (
            Wext_from_bot.sel(z=slice(s.domain["le"][2], 0)),
            Wext_from_top.sel(z=slice(0, s.domain["re"][2])),
        ),
        dim="z",
    )
    dset["Wtot"] = dset["Wsg"] + dset["Wext"]

    return dset


def plot_cr_velocity_sigma(simgroup, gr):
    sims = simgroup[gr]
    fig, axes = plt.subplots(5, 2, figsize=(6, 10), sharex=True, constrained_layout=True)
    print(len(axes.T))
    for m, s in sims.items():
        color = model_color[m]
        if s.options["cosmic_ray"]:
            dset = s.zp_ph.sel(time=slice(150, 500)).sum(dim="vz_dir")
            dset["cs"] = np.sqrt(dset["press"] / dset["dens"])
            dset["Cc"] = np.sqrt(dset["0-rhoCcr2"] / dset["dens"])
            dset["Ceff"] = np.sqrt((dset["0-rhoCcr2"] + dset["press"]) / dset["dens"])
            dset["vz"] = np.sqrt(dset["Pturbz"] / dset["dens"])
            vmax_kms = s.par["cr"]["vmax"] / 1.0e5
            dset["sigma"] = dset["0-Sigma_diff1"] / vmax_kms / s.u.cm**2 * s.u.s
            dset["kappa"] = dset["0-kappac"] * (s.u.cm**2 / s.u.s)
            dset["cr_heating"] = (
                -dset["0-heating_cr"] * (s.u.energy_density / s.u.time).cgs.value
            )
            if "0-work_cr" in dset:
                dset["cr_work"] = (
                    dset["0-work_cr"] * (s.u.energy_density / s.u.time).cgs.value
                )
            if m.endswith("va1"):
                ls_sigma = "-"
            else:
                ls_sigma = ":"
            for axs, ph in zip(axes.T, ["wc", "hot"]):
                len(axs)
                plt.sca(axs[0])
                for pfield, ls in zip(["Ceff", "vz"], ["-", ":"]):
                    plot_zprof(dset, pfield, ph, color=color, label=pfield, ls=ls)
                plt.ylim(bottom=0)
                plt.title(f"phase={ph}")
                plt.sca(axs[1])
                plot_zprof_field(dset, "sigma", ph, color=color, label=model_name[m])
                plt.sca(axs[2])
                plot_zprof_field(dset, "kappa", ph, color=color, label=model_name[m])
                plt.sca(axs[3])
                plot_zprof_field(dset, "cr_heating", ph, color=color, label=ph,ls=ls_sigma)
                plt.sca(axs[4])
                if "cr_work" in dset:
                    plot_zprof_field(dset, "cr_work", ph, color=color, label=ph)
    plt.sca(axes[0, 0])
    lines, labels = axes[0, 0].get_legend_handles_labels()
    custom_lines = [lines[0], lines[1]]
    plt.legend(
        custom_lines,
        [r"$\overline{C}_{\rm eff}$", r"$\overline{v}_z$"],
        fontsize="x-small",
    )
    plt.ylabel(r"velocity $[{\rm km/s}]$")
    plt.ylim(0, 100)
    plt.sca(axes[0, 1])
    plt.ylabel(r"velocity $[{\rm km/s}]$")
    plt.ylim(0, 350)

    plt.sca(axes[1, 0])
    # plt.legend(fontsize="x-small")
    plt.ylabel(r"$\tilde{\sigma}_{\parallel}[{\rm cm^{-2}\,s}]$")
    plt.ylim(0, 5.0e-28)
    plt.sca(axes[1, 1])
    plt.ylabel(r"$\tilde{\sigma}_{\parallel}[{\rm cm^{-2}\,s}]$")
    plt.ylim(0, 5.0e-28)

    plt.sca(axes[2, 0])
    plt.ylabel(r"$\tilde{\kappa}_{\parallel}[{\rm cm^{2}\,s^{-1}}]$")
    plt.yscale("log")
    plt.sca(axes[2, 1])
    plt.ylabel(r"$\tilde{\kappa}_{\parallel}[{\rm cm^{2}\,s^{-1}}]$")
    plt.yscale("log")
    plt.ylim(1.0e27, 1.0e29)

    plt.sca(axes[3, 0])
    plt.ylabel(r"$\tilde{\Gamma}_{\rm cr}[{\rm erg\,s^{-1}\,cm^{-3}}]$")
    plt.xlabel(r"$z\,[{\rm kpc}]$")
    plt.ylim(0, 7.0e-28)
    plt.xlim(-4, 4)
    plt.sca(axes[3, 1])
    plt.ylabel(r"$\tilde{\Gamma}_{\rm cr}[{\rm erg\,s^{-1}\,cm^{-3}}]$")
    plt.xlabel(r"$z\,[{\rm kpc}]$")
    plt.ylim(0, 7.0e-28)
    plt.xlim(-4, 4)
    plt.savefig(osp.join(outdir, f"{gr}_cr_velocity_sigma.pdf"))


# def plot_cr_field(s,field="sigma",ph="wc"):
#     area = np.prod(s.domain["Lx"][:-1])
#     if s.options["cosmic_ray"]:
#         dset = s.zp_ph.sel(time=slice(150,500)).sum(dim="vz_dir")
#         vmax_kms = s.par["cr"]["vmax"]/1.e5
#         dset["sigma"] = dset["0-Sigma_diff1"]/vmax_kms/s.u.cm**2*s.u.s
#         dset["cr_heating"] = -dset["0-heating_cr"]*(s.u.energy_density/s.u.time).cgs.value
#         # dset["kappa"] = 1/dset["0-kappac"]/(s.u.cm**2*s.u.s)
#         for j,ph in enumerate(["wc","hot"]):
#             plot_zprof_field(dset,field,ph,color=f"C{j}",label=ph)
#         plt.legend(fontsize="x-small")


def plot_flux_z(simgroup, gr, vz_dir=None):
    sims = simgroup[gr]
    fig, axes = plt.subplots(
        3, 2, figsize=(8, 7), sharey="row", sharex="col", constrained_layout=True
    )
    for m, s in sims.items():
        color = model_color[m]
        # dset = s.zp_ph.sum(dim="vz_dir").sel(time=slice(150,500))
        if vz_dir is None:
            dset = s.zp_ph.sum(dim="vz_dir").sel(time=slice(150, 500))
            dset = update_stress(s, dset)
        else:
            dset_outin = []
            for vz_dir_ in [1, -1]:
                dset_upper = (
                    s.zp_ph.sel(vz_dir=vz_dir_).sel(z=slice(0, s.domain["re"][2]))
                    * vz_dir_
                )
                dset_lower = s.zp_ph.sel(vz_dir=-vz_dir_).sel(
                    z=slice(s.domain["le"][2], 0)
                ) * (-vz_dir_)
                dset = xr.concat([dset_lower, dset_upper], dim="z")
                dset_outin.append(update_stress(s, dset))

        for axs, ph in zip(axes.T, ["wc", "hot"]):
            plt.sca(axs[0])
            plt.title(f"phase={ph}")
            if vz_dir == 1:
                dset = dset_outin[0]
            elif vz_dir == -1:
                dset = dset_outin[1]
            plot_zprof(dset, "mflux", ph, line="mean", color=color, label="out")
            if vz_dir == 1:
                mf_in = dset_outin[1]["mflux"].sel(phase=ph).mean(dim="time")
                plt.plot(mf_in.z / 1.0e3, mf_in, color=color, ls=":", label="in")
            elif vz_dir == -1:
                mf_in = dset_outin[0]["mflux"].sel(phase=ph).mean(dim="time")
                plt.plot(mf_in.z / 1.0e3, mf_in, color=color, ls=":", label="in")
            plt.ylim(1.0e-4, 1e-1)
            plt.yscale("log")

            plt.sca(axs[1])
            plot_zprof(
                dset, "pflux_MHD", ph, line="mean", color=color, label=model_name[m]
            )
            if s.options["cosmic_ray"]:
                plot_zprof(
                    dset,
                    "pflux_CR",
                    ph,
                    line="mean",
                    color=color,
                    lw=1,
                    ls="--",
                )
            plt.yscale("log")
            plt.ylim(1.0e-2, 10)

            plt.sca(axs[2])
            plot_zprof(dset, "eflux_MHD", ph, line="mean", color=color, label="MHD")
            if s.options["cosmic_ray"]:
                plot_zprof(
                    dset,
                    "eflux_CR",
                    ph,
                    line="mean",
                    color=color,
                    label="CR",
                    lw=1,
                    ls="--",
                )
            plt.yscale("log")
            plt.ylim(1.0e43, 1.0e47)
            plt.xlim(0, 4)
    axs = axes[:, 0]
    plt.sca(axs[0])
    lines, labels = axs[0].get_legend_handles_labels()
    custom_lines = [lines[0], lines[1]]
    plt.legend(custom_lines, ["outflow", "inflow"], fontsize="x-small")
    # plt.ylabel(r"$\langle n_H\rangle\,[{\rm cm^{-3}}]$")
    # plt.sca(axs[1])
    vout_label = "out" if vz_dir == 1 else "net"
    plt.ylabel(
        f"$\\mathcal{{F}}_M^{{\\rm {vout_label}}}$"
        r"$\,[M_\odot{\rm \,kpc^{-2}\,yr^{-1}}]$"
    )
    plt.sca(axs[1])
    plt.legend(fontsize="x-small")
    plt.ylabel(
        f"$\\mathcal{{F}}_p^{{\\rm {vout_label}}}$"
        r"$\,[M_\odot{\rm \,km/s\,kpc^{-2}\,yr^{-1}}]$"
    )
    plt.sca(axs[2])
    plt.ylabel(
        f"$\\mathcal{{F}}_E^{{\\rm {vout_label}}}$"
        r"$\,[{\rm erg\,kpc^{-2}\,yr^{-1}}]$"
    )
    lines, labels = axs[2].get_legend_handles_labels()
    custom_lines = [lines[0], lines[1]]
    plt.legend(custom_lines, ["MHD flux", "CR flux"], fontsize="x-small")
    plt.setp(axes[-1, :], "xlabel", r"$z\,[{\rm kpc}]$")

    plt.savefig(osp.join(outdir, f"{gr}_flux_{vout_label}_z.pdf"))


def plot_loading_z(simgroup, gr, vz_dir=None):
    sims = simgroup[gr]
    fig, axs = plt.subplots(
        3, 1, figsize=(3, 6), sharey="row", sharex="col", constrained_layout=True
    )
    for m, s in sims.items():
        Zsn = s.par["feedback"]["Z_SN"]
        Mej = s.par["feedback"]["M_ej"]
        dt = 0.1
        mstar = 1 / np.sum(s.pop_synth["snrate"] * dt)
        field = "sfr40"
        h = s.read_hst()
        sfr_avg = h[field].loc[150:].mean()
        sfr_std = h[field].loc[150:].std()
        ref_flux = dict(
            mflux=sfr_avg / mstar * mstar,
            pflux_MHD=sfr_avg / mstar * 1.25e5,
            pflux_CR=sfr_avg / mstar * 1.25e5,
            eflux_MHD=sfr_avg / mstar * 1.0e51,
            eflux_CR=sfr_avg / mstar * 1.0e51,
            mZflux=sfr_avg / mstar * Mej * Zsn,
        )

        color = model_color[m]
        # dset = s.zp_ph.sum(dim="vz_dir").sel(time=slice(150,500))
        if vz_dir is None:
            dset = s.zp_ph.sum(dim="vz_dir").sel(time=slice(150, 500))
            dset = update_stress(s, dset)
        else:
            dset_outin = []
            for vz_dir_ in [1, -1]:
                dset_upper = (
                    s.zp_ph.sel(vz_dir=vz_dir_).sel(z=slice(0, s.domain["re"][2]))
                    * vz_dir_
                )
                dset_lower = s.zp_ph.sel(vz_dir=-vz_dir_).sel(
                    z=slice(s.domain["le"][2], 0)
                ) * (-vz_dir_)
                dset = xr.concat([dset_lower, dset_upper], dim="z")
                dset_outin.append(update_stress(s, dset))

        # for axs, ph in zip(axes.T, [["wc", "hot"],"hot"]):
        ph = ["wc", "hot"]
        plt.sca(axs[0])
        # plt.title(f"phase={ph}")
        if vz_dir == 1:
            dset = dset_outin[0]
        elif vz_dir == -1:
            dset = dset_outin[1]
        plot_zprof(
            dset,
            "mflux",
            ph,
            norm=ref_flux["mflux"],
            line="mean",
            color=color,
            label=model_name[m],
        )
        plt.ylim(1.0e-2, 10)
        plt.yscale("log")

        plt.sca(axs[1])
        plot_zprof(
            dset,
            "pflux_MHD",
            ph,
            norm=ref_flux["pflux_MHD"],
            line="mean",
            color=color,
            label=model_name[m],
        )
        if s.options["cosmic_ray"]:
            plot_zprof(
                dset,
                "pflux_CR",
                ph,
                norm=ref_flux["pflux_CR"],
                line="mean",
                color=color,
                label="CR",
                lw=1,
                ls="--",
            )
        plt.yscale("log")
        plt.ylim(1.0e-2, 1)

        plt.sca(axs[2])
        plot_zprof(
            dset,
            "eflux_MHD",
            ph,
            norm=ref_flux["eflux_MHD"],
            line="mean",
            color=color,
            label="MHD",
        )
        if s.options["cosmic_ray"]:
            plot_zprof(
                dset,
                "eflux_CR",
                ph,
                norm=ref_flux["eflux_CR"],
                line="mean",
                color=color,
                label="CR",
                lw=1,
                ls="--",
            )
        plt.yscale("log")
        plt.ylim(1.0e-3, 5.0)
        plt.xlim(0, 4)
    # axs = axes[:, 0]
    plt.sca(axs[0])
    plt.legend(fontsize="x-small")

    # lines, labels = axs[0].get_legend_handles_labels()
    # custom_lines = [lines[0], lines[1]]
    # plt.legend(custom_lines, ["outflow", "inflow"], fontsize="x-small")
    # plt.ylabel(r"$\langle n_H\rangle\,[{\rm cm^{-3}}]$")
    # plt.sca(axs[1])
    vout_label = "out" if vz_dir == 1 else ""
    plt.ylabel(
        f"$\\eta_M^{{\\rm {vout_label}}}$"
        # r"$\,[M_\odot{\rm \,kpc^{-2}\,yr^{-1}}]$"
    )
    plt.sca(axs[1])
    plt.ylabel(
        f"$\\eta_p^{{\\rm {vout_label}}}$"
        # r"$\,[M_\odot{\rm \,km/s\,kpc^{-2}\,yr^{-1}}]$"
    )
    plt.sca(axs[2])
    plt.ylabel(
        f"$\\eta_E^{{\\rm {vout_label}}}$"
        # r"$\,[{\rm erg\,kpc^{-2}\,yr^{-1}}]$"
    )
    lines, labels = axs[2].get_legend_handles_labels()
    custom_lines = [lines[0], lines[1]]
    plt.legend(custom_lines, ["MHD flux", "CR flux"], fontsize="x-small")
    plt.setp(axs[-1], "xlabel", r"$z\,[{\rm kpc}]$")

    plt.savefig(osp.join(outdir, f"{gr}_loading_{vout_label}_z.pdf"))


def plot_area_mass_fraction_z(simgroup, gr):
    sims = simgroup[gr]
    models = list(sims.keys())
    nmodels = len(models)
    fig, axes = plt.subplots(
        2,
        nmodels,
        figsize=(4 * nmodels, 5),
        sharey="row",
        sharex=True,
        constrained_layout=True,
    )
    for i, m in enumerate(models):
        s = sims[m]

        dset = s.zprof.sum(dim="vz_dir").sel(time=slice(150, 500))
        dset_upper = (
            s.zprof.sel(vz_dir=1)
            .sel(time=slice(150, 500))
            .sel(z=slice(0, s.domain["re"][2]))
        )
        dset_lower = (
            s.zprof.sel(vz_dir=-1)
            .sel(time=slice(150, 500))
            .sel(z=slice(s.domain["le"][2], 0))
        )
        dset_out = xr.concat([dset_lower, dset_upper], dim="z")

        for axs, field in zip(axes, ["area", "rho"]):
            plt.sca(axs[i])
            for ph, color, label in zip(
                [["CNM", "UNM"], "WNM", ["WHIM", "HIM"]],
                ["C0", "limegreen", "C3"],
                ["CNM+UNM", "WNM", "WHIM+HIM"],
            ):
                plot_zprof_frac(dset, field, ph, line="mean", color=color, label=label)
                if field == "area":
                    fA_out = (
                        dset_out[field].sel(phase=np.atleast_1d(ph)).sum(dim="phase")
                        / dset_out["area"].sel(phase="whole")
                    ).mean(dim="time")
                    plt.plot(
                        fA_out.z / 1.0e3, fA_out, color=color, ls="--", label="outflow"
                    )

        plt.sca(axes[0, i])
        # plt.annotate(model_name[m],(0.05,0.95),xycoords="axes fraction",ha="left",va="top",color=model_color[m])
        plt.title(model_name[m], color=model_color[m])

    plt.sca(axes[0, 0])
    plt.ylim(0, 1)
    plt.xlim(-4, 4)
    plt.ylabel(r"$f_A, f_A^{\rm out}$")
    lines, labels = axes[0, 0].get_legend_handles_labels()
    custom_lines = [lines[4], lines[5]]
    plt.legend(
        custom_lines,
        [r"total", r"outflow"],
        fontsize="x-small",
    )
    plt.sca(axes[1, 0])
    plt.legend(fontsize="x-small")
    plt.yscale("log")
    plt.ylabel(r"$\langle{n}_{\rm H}\rangle\,[{\rm cm^{-3}}]$")
    plt.ylim(1.0e-5, 1)

    plt.setp(axes[-1, :], "xlabel", r"$z\,[{\rm kpc}]$")
    plt.savefig(osp.join(outdir, f"{gr}_area_nH_profile_frac_z.pdf"))


def plot_vertical_proflies_separate(simgroup, gr):
    field = "area"
    plot_profile_frac_z(simgroup, gr, field=field, line="mean", savefig=False)
    plt.ylabel(r"$f_A$")
    plt.legend().remove()
    plt.yscale("linear")
    plt.ylim(0, 1)
    plt.savefig(osp.join(outdir, f"{gr}_{field}_profile_frac_z.pdf"))

    field = "area"
    plot_profile_frac_z(simgroup, gr, field=field, vz_dir=1, line="mean")
    plt.ylabel(r"$f_A^{\rm out}$")
    plt.legend().remove()
    plt.yscale("linear")
    plt.ylim(0, 1)
    plt.savefig(osp.join(outdir, f"{gr}_{field}_profile_frac_out_z.pdf"))

    field = "rho"
    plot_profile_frac_z(simgroup, gr, field=field, line="mean")
    plt.ylabel(r"$\langle{n}_{\rm H}\rangle\,[{\rm cm^{-3}}]$")
    plt.legend(fontsize="xx-small")
    plt.ylim(1.0e-5, 5)
    plt.savefig(osp.join(outdir, f"{gr}_{field}_profile_frac_z.pdf"))

    field = "rho"
    plot_profile_frac_z(simgroup, gr, field=field, vz_dir=1, line="mean")
    plt.ylabel(r"$\langle{n}_{\rm H}\rangle^{\rm out}\,[{\rm cm^{-3}}]$")
    plt.legend(fontsize="xx-small")
    plt.ylim(1.0e-5, 5)
    plt.savefig(osp.join(outdir, f"{gr}_{field}_profile_frac_out_z.pdf"))

    field = "rho"
    plot_profile_z(simgroup, gr, field=field)
    plt.ylabel(r"$\overline{n}_{\rm H}\,[{\rm cm^{-3}}]$")
    plt.legend().remove()
    plt.ylim(1.0e-4, 10)
    plt.savefig(osp.join(outdir, f"{gr}_{field}_profile_z.pdf"))


def plot_velocity_z(simgroup, gr):
    sims = simgroup[gr]
    models = list(sims.keys())

    fig, axes = plt.subplots(
        3, 1, figsize=(3, 6), sharex=True, constrained_layout=True, squeeze=False
    )
    for i, m in enumerate(models):
        s = sims[m]
        c = model_color[m]
        dnet = s.zp_ph.sel(time=slice(150, 500), z=slice(0, s.zp_ph.z.max())).sum(
            dim="vz_dir"
        )
        dout = s.zp_ph.sel(time=slice(150, 500), z=slice(0, s.zp_ph.z.max()), vz_dir=1)
        for axs, ph in zip(axes.T, ["wc", "hot"]):
            plt.sca(axs[0])
            plot_zprof_field(
                dout, "vel3", ph, color=c, line="median", label=model_name[m]
            )
            plot_zprof_field(
                dnet, "vel3", ph, color=c, line="median", quantile=False, lw=1, ls=":"
            )
            # plt.ylim(-25, 75)
            plt.title(f"phase={ph}")

            plt.sca(axs[1])
            if s.options["cosmic_ray"]:
                plot_zprof_field(dout, "0-Vs3", ph, color=c, line="median", label="out")
                plot_zprof_field(
                    dnet,
                    "0-Vs3",
                    ph,
                    color=c,
                    line="median",
                    quantile=False,
                    lw=1,
                    ls=":",
                    label="net",
                )
            # plt.ylim(0, 50)

            plt.sca(axs[2])
            if s.options["cosmic_ray"]:
                plot_zprof_field(dout, "0-Vd3", ph, color=c, line="median")
                plot_zprof_field(
                    dnet,
                    "0-Vd3",
                    ph,
                    color=c,
                    line="median",
                    quantile=False,
                    lw=1,
                    ls=":",
                )
            plt.ylim(-50, 50)

    plt.sca(axes[0, 0])
    plt.ylabel(r"$\tilde{v}_z\,[{\rm km/s}]$")
    plt.legend(fontsize="x-small")
    plt.sca(axes[1, 0])
    plt.ylabel(r"$\tilde{v}_{s,z}\,[{\rm km/s}]$")
    # adding custom legend for two line styles
    lines, labels = axes[1, 0].get_legend_handles_labels()
    custom_lines = [lines[0], lines[1]]
    plt.legend(custom_lines, ["out", "net"], fontsize="x-small")

    plt.sca(axes[2, 0])
    plt.ylabel(r"$\tilde{v}_{d,z}\,[{\rm km/s}]$")
    plt.xlabel(r"$z\,[{\rm kpc}]$")
    plt.xlim(0, 4)
    plt.savefig(osp.join(outdir, f"{gr}_velocity_z.pdf"))


def plot_history(simgroup, gr):
    sims = simgroup[gr]
    fig, axes = plt.subplots(
        1, 2, figsize=(8, 2.5), sharex=True, constrained_layout=True
    )

    plt.sca(axes[0])
    field = "sfr40"
    for m, s in sims.items():
        color = model_color[m]
        name = model_name[m]
        h = s.hst
        plt.plot(h["time"], h[field], label=name, color=color)

        avg = h[field].loc[150:].mean()
        std = h[field].loc[150:].std()
        plt.axhline(avg, color=color, lw=1, ls="--")
        plt.axhspan(avg - std, avg + std, color=color, alpha=0.1, lw=0)

    plt.sca(axes[1])
    for field, label, ls in zip(
        ["Sigma_gas", "Sigma_out"],
        [r"$\Sigma_{\rm gas}$", r"$\Sigma_{\rm out}$"],
        ["-", "--"],
    ):
        i = 0
        for m, s in sims.items():
            color = model_color[m]
            name = model_name[m]
            h = s.hst
            plt.plot(
                h["time"], h[field], label=label if i == 0 else None, ls=ls, color=color
            )
            i += 1
    plt.sca(axes[0])
    plt.ylabel(r"$\Sigma_{\rm SFR}\,[M_\odot\,{\rm kpc^{-2}\,yr^{-1}}]$")
    plt.ylim(bottom=0)
    # plt.ylim(1.0e-3, 1.0e-2)
    # plt.yscale("log")
    plt.legend(fontsize="x-small")
    plt.xlabel(r"$t\,[{\rm Myr}]$")
    plt.annotate("(a)", xy=(0.05, 0.95), xycoords="axes fraction", ha="left", va="top")
    plt.sca(axes[1])
    plt.ylabel(r"$\Sigma\,[M_\odot\,{\rm pc^{-2}}]$")
    plt.ylim(0, 15)
    plt.legend(fontsize="x-small")
    plt.xlabel(r"$t\,[{\rm Myr}]$")
    plt.xlim(0, 500)
    plt.annotate("(b)", xy=(0.05, 0.95), xycoords="axes fraction", ha="left", va="top")
    plt.savefig(osp.join(outdir, f"{gr}_history.pdf"))


def plot_pressure_t(simgroup, gr, ph="wc"):
    sims = simgroup[gr]
    fig, axes = plt.subplots(
        1, 4, figsize=(8, 2), sharey=True, sharex=True, constrained_layout=True
    )
    for m, s in sims.items():
        color = model_color[m]
        name = model_name[m]
        dset = s.zp_ph.sel(z=slice(-50, 50)).sum(dim="vz_dir")
        dset = update_stress(s, dset)
        for ax, pfield in zip(axes, ["Pok_cr", "Pok_th", "Pok_kin", "Pi_B"]):
            plt.sca(ax)
            lab = pfield.split("_")[-1]
            if pfield not in dset:
                continue
            # if pfield == "Pok_cr":
            #     pok = (dset[pfield].sum(dim="phase")/dset["area"].sum(dim="phase")).mean(dim="z")
            #     plt.plot(pok.time * s.u.Myr, pok, color=color, lw=1)

            pok = (dset[pfield] / dset["area"]).sel(phase=ph).mean(dim="z")
            plt.plot(pok.time * s.u.Myr, pok, label=name, color=color)

            if pfield.startswith("Pok_"):
                label = f"$P_{{\\rm {lab}}}$"
            else:
                label = f"$\\Pi_{{\\rm {lab}}}$"
            # plt.annotate(label,(0.05,0.95),
            #              xycoords="axes fraction",ha="left",va="top")
            plt.title(label)
            plt.yscale("log")
            plt.axvline(150, color="k", ls=":")
            plt.ylim(5.0e2, 1.0e5)
            plt.xlabel(r"$t\,[{\rm Myr}]$")
            plt.xlim(0, 500)

    plt.sca(axes[1])
    plt.legend(fontsize="small")
    plt.sca(axes[0])
    plt.ylabel(r"$\overline{P}^{\rm \;wc}(z=0)/k_B\,[{\rm cm^{-3}\,K}]$")
    plt.savefig(osp.join(outdir, f"{gr}_pressure_t.pdf"))


def plot_vertical_equilibrium_t(simgroup, gr, ph="wc", exclude=[]):
    sims = simgroup[gr]
    for m in exclude:
        sims = dict(sims)
        sims.pop(m, None)
    nmodels = len(sims)
    fig, axes = plt.subplots(
        1,
        nmodels,
        figsize=(4 * nmodels, 2.5),
        sharey=True,
        sharex=True,
        constrained_layout=True,
    )
    for (m, s), ax in zip(sims.items(), axes):
        plt.sca(ax)
        color = model_color[m]
        name = model_name[m]
        # plt.annotate(name, xy=(0.05, 0.95), xycoords="axes fraction",
        #              ha="left", va="top", color=color)
        plt.title(name, color=color)
        dset = s.zp_ph.sum(dim="vz_dir").sel(phase=ph)
        # dset = s.zp_ph.sum(dim=["phase","vz_dir"])
        dset = update_stress(s, dset)

        # calculate delta between z=0 and z=1kpc
        z0 = 1000
        dz = s.domain["dx"][2]

        # total pressure
        Ptot = dset["Pok_tot"] / dset["area"]
        Ptot_mid = Ptot.sel(z=slice(-dz, dz)).mean(dim="z")
        Ptot_1kpc = 0.5 * (
            Ptot.sel(z=slice(z0 - dz, z0 + dz)).mean(dim="z")
            + Ptot.sel(z=slice(-z0 - dz, -z0 + dz)).mean(dim="z")
        )
        delta = Ptot_mid - Ptot_1kpc
        plt.plot(dset.time * s.u.Myr, delta * 1.0e-4, label=r"$P_{\rm tot}$")

        # total weight
        area_tot = s.domain["Lx"][0] * s.domain["Lx"][1]
        Wtot = dset["Wtot"] / area_tot
        Wtot_mid = Wtot.sel(z=slice(-dz, dz)).mean(dim="z")
        Wtot_1kpc = 0.5 * (
            Wtot.sel(z=slice(z0 - dz, z0 + dz)).mean(dim="z")
            + Wtot.sel(z=slice(-z0 - dz, -z0 + dz)).mean(dim="z")
        )
        delta = Wtot_mid - Wtot_1kpc
        plt.plot(dset.time * s.u.Myr, delta * 1.0e-4, label=r"$\mathcal{W}$")

        # CR
        if s.options["cosmic_ray"]:
            Pcr = dset["Pok_cr"] / dset["area"]
            Pcr_mid = Pcr.sel(z=slice(-dz, dz)).mean(dim="z")
            Pcr_1kpc = 0.5 * (
                Pcr.sel(z=slice(z0 - dz, z0 + dz)).mean(dim="z")
                + Pcr.sel(z=slice(-z0 - dz, -z0 + dz)).mean(dim="z")
            )
            delta = Pcr_mid - Pcr_1kpc
            plt.plot(dset.time * s.u.Myr, delta * 1.0e-4, label=r"$P_{\rm cr}$")
        # plt.yscale("log")
        # plt.ylim(5.0e2, 5.0e4)
        plt.xlabel(r"$t\,[{\rm Myr}]$")
    plt.xlim(0, 500)
    plt.sca(axes[0])
    plt.ylabel(  # r"$\langle P_{\rm tot} \rangle^{\rm wc}/f_A^{\rm wc}$"
        # r"$\,\langle \mathcal{W}_{\rm tot}\rangle^{\rm wc}$"
        r"$\Delta_{1{\rm kpc}} P\,[10^4 k_B{\rm cm^{-3}\,K}]$"
    )
    plt.legend(fontsize="small", loc=1)
    plt.savefig(osp.join(outdir, f"{gr}_vertical_equilibrium_t.pdf"))


def plot_jointpdf(simgroup, gr):
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(10, 4.8),
        sharey="row",
        sharex="col",
        constrained_layout=True,
        gridspec_kw={"hspace": 0.01, "wspace": 0.01},
    )
    for (m, s), axs in zip(simgroup[gr].items(), axes):
        outflux = s.outflux
        dbin = outflux.logvz[1] - outflux.logvz[0]
        dbinsq = dbin**2
        name = model_name[m]
        for z0, ax in zip(outflux.z, axs):
            plt.sca(ax)
            plt.pcolormesh(
                outflux.logvz,
                outflux.logcs,
                outflux.sel(z=z0, flux="mflux") / dbinsq,
                norm=LogNorm(1.0e-5, 1.0),
                cmap=cmr.fall_r,
            )
            if name == "crmhd":
                plt.title(f"$z={z0 / 1.0e3:3.1f} {{\\rm kpc}}$")
            if z0 == 500:
                plt.annotate(
                    name,
                    xy=(0.05, 0.95),
                    xycoords="axes fraction",
                    ha="left",
                    va="top",
                    color=model_color[m],
                )
            plt.xlim(0, 3.5)
            plt.ylim(0, 3.3)
            ax.set_aspect("equal")
    plt.setp(axes[:, 0], "ylabel", r"$\log c_s\,[{\rm km/s}]$")
    plt.setp(axes[1, :], "xlabel", r"$\log v_{\rm out}\,[{\rm km/s}]$")
    plt.colorbar(
        mappable=axes[0, 0].collections[0],
        ax=axes[:, -1],
        label=r"$d^2\mathcal{F}_M/d\log v_{\rm out}d\log c_s\,[M_\odot\,{\rm kpc^{-2}\,yr^{-1}\,dex^{-2}}]$",
    )
    plt.savefig(osp.join(outdir, f"{gr}_jointpdfs.png"))


def plot_voutpdf(simgroup, gr, savefig=True):
    fig, axes = plt.subplots(
        2, 2, figsize=(8, 6), sharex="col", sharey="row", constrained_layout=True
    )
    for (m, s), axs in zip(simgroup[gr].items(), axes.T):
        outflux = s.outflux
        dbin = outflux.logvz[1] - outflux.logvz[0]

        for z0 in outflux.z:
            plt.sca(axs[0])
            plt.plot(
                outflux.logvz,
                outflux.sel(logcs=slice(0, 1.2), z=z0, flux="mflux").sum(dim="logcs")
                / dbin,
                color="tab:blue",
                lw=z0 / 1.0e3,
                label=f"${z0 / 1.0e3:3.1f} {{\\rm kpc}}$",
            )
            plt.plot(
                outflux.logvz,
                outflux.sel(logcs=slice(1.2, 4), z=z0, flux="mflux").sum(dim="logcs")
                / dbin,
                color="tab:red",
                lw=z0 / 1.0e3,
            )
            plt.yscale("log")
            plt.ylim(1.0e-5, 1.0e-1)
            plt.sca(axs[1])
            plt.plot(
                outflux.logvz,
                outflux.sel(logcs=slice(0, 1.2), z=z0, flux="eflux").sum(dim="logcs")
                / dbin,
                color="tab:blue",
                lw=z0 / 1.0e3,
                label="wc" if z0 == 3000 else None,
            )
            plt.plot(
                outflux.logvz,
                outflux.sel(logcs=slice(1.2, 4), z=z0, flux="eflux").sum(dim="logcs")
                / dbin,
                color="tab:red",
                lw=z0 / 1.0e3,
                label="hot" if z0 == 3000 else None,
            )
            plt.yscale("log")
            plt.ylim(1.0e42, 5.0e46)
    plt.sca(axes[0, 0])
    plt.title("crmhd")
    plt.ylabel(
        r"$d\mathcal{F}_M/d\log v_{\rm out}$"
        + "\n"
        + r"$[M_\odot\,{\rm kpc^{-2}\,yr^{-1}\,dex^{-1}}]$"
    )
    plt.legend(fontsize="x-small")
    plt.xlim(0, 3.5)
    plt.sca(axes[0, 1])
    plt.title("mhd")
    plt.xlim(0, 3.5)
    plt.sca(axes[1, 0])
    plt.legend(fontsize="x-small")
    plt.xlabel(r"$\log v_{\rm out}\,[{\rm km/s}]$")
    plt.ylabel(
        r"$d\mathcal{F}_{E,{\rm MHD}}/d\log v_{\rm out}$"
        + "\n"
        + r"$[{\rm erg}\,{\rm kpc^{-2}\,yr^{-1}\,dex^{-1}}]$"
    )
    plt.sca(axes[1, 1])
    plt.xlabel(r"$\log v_{\rm out}\,[{\rm km/s}]$")
    if savefig:
        plt.savefig(osp.join(outdir, f"{gr}_voutpdf.pdf"))


if __name__ == "__main__":
    model_dict = cr_data_load()
    sims = LoadSimTIGRESSPPAll(model_dict)
    sim_group = cr_data_load(sims)
    print_sim_table(sims)

    crgroups = list(sim_group.keys())[1:-1]
    for gr in sim_group:
        load_group(sim_group, gr)
