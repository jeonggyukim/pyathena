import numpy as np
import xarray as xr
import os


def calc_pdf(
    dset_z,
    dbin=0.02,
    dVol=1.0,
    fields=["volume", "density", "massflux", "momflux", "energyflux"],
):
    pdfout = dict()
    pdfin = dict()
    bounds = dict(vin=[0, 3.5], vout=[0, 3.5], cs=[0, 3.5])
    bins = dict()
    for k in bounds:
        bins[k] = np.arange(bounds[k][0], bounds[k][1] + dbin, dbin)

    yf = "cs"
    normout = dict()
    normin = dict()
    for wf in fields:
        xf = "vout"
        x = dset_z[xf].load()
        y = dset_z[yf].load()
        key = "{}-{}-{}".format(xf, yf, wf)
        if wf == "volume":
            pdfout[key] = np.histogram2d(
                np.log10(x.data.flat), np.log10(y.data.flat), bins=[bins[xf], bins[yf]]
            )
            normout[wf] = pdfout[key][0].sum()
            normout["dVol"] = dVol
        else:
            w = dset_z[wf].load()
            pdfout[key] = np.histogram2d(
                np.log10(x.data.flat),
                np.log10(y.data.flat),
                bins=[bins[xf], bins[yf]],
                weights=w.data.flatten(),
            )
            normout[wf] = w.where(x > 0).sum().data
        xf = "vin"
        x = -dset_z["vout"].load()
        y = dset_z[yf].load()
        if wf == "volume":
            pdfin[key] = np.histogram2d(
                np.log10(x.data.flat), np.log10(y.data.flat), bins=[bins[xf], bins[yf]]
            )
            normin[wf] = pdfin[key][0].sum()
            normin["dVol"] = dVol
        else:
            if wf.endswith("flux"):
                w = -dset_z[wf].load()
            else:
                w = dset_z[wf].load()
            pdfin[key] = np.histogram2d(
                np.log10(x.data.flat),
                np.log10(y.data.flat),
                bins=[bins[xf], bins[yf]],
                weights=w.data.flatten(),
            )
            normin[wf] = w.where(x > 0).sum().data
    return pdfout, pdfin, normout, normin


def create_wind_pdf(dset, sim):
    savdir = "{}/vz_pdf".format(sim.basedir)
    if not os.path.isdir(savdir):
        os.mkdir(savdir)
    dVol = np.prod(sim.domain["dx"])
    Nx, Ny, Nz = sim.domain["Nx"]
    Nt = len(dset.time)
    dbin = 0.02
    default_fields = ["volume", "density", "massflux", "momflux", "energyflux"]
    extra_fields = [
        "momflux_kin",
        "momflux_th",
        "momflux_mag",
        "energyflux_kin_z",
        "energyflux_th",
        "energyflux_kin",
        "poyntingflux",
    ]
    fields = default_fields + extra_fields
    for z in [500, 1000]:
        print(z)
        dset_z = dset.sel(z=[z, -z])
        pdfout, pdfin, normout, normin = calc_pdf(
            dset_z, dbin=dbin, dVol=dVol, fields=fields
        )

        pdf_ds = xr.Dataset()
        for key in pdfout:
            xf, yf, wf = key.split("-")
            pdf_ds[wf] = xr.DataArray(
                pdfout[key][0].T / normout[wf] / dbin ** 2,
                coords=[pdfout[key][2][:-1], pdfout[key][1][:-1]],
                dims=["cs", "vout"],
            )
        pdf_ds.attrs = normout
        pdf_ds.attrs["dbin"] = dbin
        pdf_ds.attrs["sfr"] = dset.attrs["sfr"]
        pdf_ds.attrs["NxNyNt"] = Nx * Ny * Nt
        pdf_ds.to_netcdf(
            "{}/vz_pdf/{}.pdf-out.{:d}.nc".format(sim.basedir, sim.basename, z)
        )
        pdf_ds.close()

        pdf_ds = xr.Dataset()
        for key in pdfin:
            xf, yf, wf = key.split("-")
            pdf_ds[wf] = xr.DataArray(
                pdfin[key][0].T / normin[wf] / dbin ** 2,
                coords=[pdfin[key][2][:-1], pdfin[key][1][:-1]],
                dims=["cs", "vin"],
            )
        pdf_ds.attrs = normin
        pdf_ds.attrs["dbin"] = dbin
        pdf_ds.attrs["sfr"] = dset.attrs["sfr"]
        pdf_ds.attrs["NxNyNt"] = Nx * Ny * Nt
        pdf_ds.to_netcdf(
            "{}/vz_pdf/{}.pdf-in.{:d}.nc".format(sim.basedir, sim.basename, z)
        )
        pdf_ds.close()


def get_pdf_fname(sim, out="out", z0=500):
    return "{}/vz_pdf/{}.pdf-{}.{:d}.nc".format(sim.basedir, sim.basename, out, z0)


def load_wind_pdf(sim, z0=500):
    pdfname = get_pdf_fname(sim, z0=z0)
    pdf = xr.open_dataset(pdfname)
    units = dict()
    units["massflux"] = sim.u.mass_flux.to("Msun/(kpc^2*yr)").value
    units["metalflux"] = sim.u.mass_flux.to("Msun/(kpc^2*yr)").value
    units["energyflux"] = sim.u.energy_flux.to("erg/(kpc^2*yr)").value
    units["momflux"] = sim.u.momentum_flux.to("(Msun*km)/(kpc^2*yr*s)").value

    for f, u in units.items():
        pdf.attrs[f + "_unit"] = u
    vout = 10.0 ** pdf.vout
    cs = 10.0 ** pdf.cs
    pdf["vBz"] = np.sqrt(5 * (cs) ** 2 + (vout) ** 2)

    return pdf
