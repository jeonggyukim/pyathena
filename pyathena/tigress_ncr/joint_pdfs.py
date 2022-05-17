import numpy as np
import xarray as xr
import astropy.units as au
import astropy.constants as ac
import sys, os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import argparse

sys.path.insert(0, "../../")

import pyathena as pa
import gc

from pyathena.tigress_ncr.get_cooling import get_cooling_heating, get_pdfs


def calc_pdfs(
    s,
    ds,
    xf="T",
    yf="Lambda_cool",
    zmin=0,
    zmax=300,
    force_override=False,
    cooling=False,
    species=False,
    radiation=False,
):
    """calculate PDFs for given x and y with a variety of weight fields
    outputs are stored in individual folders for later uses

    Usage
    =====

    >>> sim = pa.LoadSim(path)
    >>> ds = sim.load_vtk(num)
    >>> calc_pdfs(ds,xf=xf,yf=yf)
    """

    pdf_z = dict()
    pdf_tot = dict()

    tot_volume = -1
    tot_mass = -1

    wfields = [None, "nH"]
    if cooling:
        wfields += ["cool_rate", "heat_rate", "net_cool_rate"]
        tot_cool_rate = -1
        if not s.test_newcool():
            data = s.get_classic_cooling_rate(ds)
            ds.arr["T"] = data["T"].data
            ds.arr["cool_rate"] = data["cool_rate"].data
            ds.arr["heat_rate"] = data["heat_rate"].data
            ds.arr["net_cool_rate"] = data["net_cool_rate"].data
            ds.arr["Lambda_cool"] = data["Lambda_cool"].data
    if species:
        wfields += ["nHI", "nHII", "nH2", "ne"]
    if radiation:
        wfields += [
            "xi_CR",
            "rad_energy_density_PE",
            "rad_energy_density_LW",
            "rad_energy_density_PH",
        ]
    for zmax, pdf_dict in zip([ds.domain["re"][2], zmax], [pdf_tot, pdf_z]):
        pdfdict = dict()
        for wf in wfields:
            pdf = s.load_one_jointpdf(
                xf, yf, wf, ds, zrange=(zmin, zmax), force_override=force_override
            )
            pdfdict["vol" if wf is None else wf] = pdf
        x = pdf.coords[pdf.dims[1]]
        y = pdf.coords[pdf.dims[0]]
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        if tot_mass < 0:
            tot_mass = pdfdict["nH"].sum().data * dx * dy
        if tot_volume < 0:
            tot_volume = np.prod(ds.domain["Nx"])
        pdf_dict["volume"] = pdfdict["vol"] / tot_volume
        pdf_dict["mass"] = pdfdict["nH"] / tot_mass

        if cooling and (tot_cool_rate < 0):
            tot_cool_rate = pdfdict["cool_rate"].sum().data * dx * dy
            pdf_dict["net_cooling"] = pdfdict["net_cool_rate"] / tot_cool_rate
            pdf_dict["cooling"] = pdfdict["cool_rate"] / tot_cool_rate
            pdf_dict["heating"] = pdfdict["heat_rate"] / tot_cool_rate

        if species:
            pdf_dict["xHI"] = pdfdict["nHI"] / pdfdict["nH"]
            pdf_dict["xHII"] = pdfdict["nHII"] / pdfdict["nH"]
            pdf_dict["xH2"] = pdfdict["nH2"] / pdfdict["nH"]
            pdf_dict["xe"] = pdfdict["ne"] / pdfdict["nH"]

        if radiation:
            pdf_dict["PE"] = pdfdict["rad_energy_density_PE"]
            pdf_dict["LW"] = pdfdict["rad_energy_density_LW"]
            pdf_dict["PH"] = pdfdict["rad_energy_density_PH"]

    return pdf_z, pdf_tot


if __name__ == "__main__":
    basedir_def = "/tigress/changgoo/TIGRESS-NCR/R8_8pc_NCR.full"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--basedir", type=str, default=basedir_def, help="Name of the basedir."
    )
    args = vars(parser.parse_args())
    locals().update(args)

    s = pa.LoadSimTIGRESSNCR(basedir, verbose=True)
    for num in range(200, 800):
        if not (num in s.nums):
            continue
        print(num)
        # for num in s.nums:
        ds = s.load_vtk(num)
        for zrange in [None, slice(-1000, 1000), slice(-300, 300)]:
            if zrange is None:
                zmin, zmax = 0, s.domain["re"][2]
            else:
                zmin, zmax = zrange.start, zrange.stop
                if zmin < 0:
                    zmin = 0
            if s.test_newcool():
                # do this only for new cooling
                savdir = s.get_savdir_pdf(zrange=zrange)
                if not os.path.isdir(savdir):
                    os.makedirs(savdir)
                coolfname = "{}.{:04d}.cool.pdf.nc".format(ds.problem_id, ds.num)
                heatfname = "{}.{:04d}.heat.pdf.nc".format(ds.problem_id, ds.num)
                # if not os.path.isfile(os.path.join(savdir,coolfname)):
                if not os.path.isfile(
                    os.path.join(savdir, coolfname.replace(".cool.", ".xHI.cool."))
                ):
                    data, coolrate, heatrate = get_cooling_heating(s, ds, zrange=zrange)

                    # get total cooling from vtk output for normalization
                    total_cooling = coolrate.attrs["total_cooling"]
                    # get total heating from vtk output for normalization
                    total_heating = heatrate.attrs["total_heating"]

                    pdf_cool = (
                        get_pdfs("nH", "T", data, coolrate).assign_coords(
                            time=ds.domain["time"]
                        )
                        / total_cooling
                    )
                    pdf_heat = (
                        get_pdfs("nH", "T", data, heatrate).assign_coords(
                            time=ds.domain["time"]
                        )
                        / total_heating
                    )

                    pdf_cool_xHI = (
                        get_pdfs("T", "xHI", data, coolrate).assign_coords(
                            time=ds.domain["time"]
                        )
                        / total_cooling
                    )
                    pdf_heat_xHI = (
                        get_pdfs("T", "xHI", data, heatrate).assign_coords(
                            time=ds.domain["time"]
                        )
                        / total_heating
                    )

                    pdf_cool.attrs = coolrate.attrs
                    pdf_heat.attrs = heatrate.attrs

                    pdf_cool_xHI.attrs = coolrate.attrs
                    pdf_heat_xHI.attrs = heatrate.attrs

                    pdf_cool.to_netcdf(os.path.join(savdir, coolfname))
                    pdf_heat.to_netcdf(os.path.join(savdir, heatfname))

                    pdf_cool_xHI.to_netcdf(
                        os.path.join(savdir, coolfname.replace(".cool.", ".xHI.cool."))
                    )
                    pdf_heat_xHI.to_netcdf(
                        os.path.join(savdir, heatfname.replace(".heat.", ".xHI.heat."))
                    )

                pdf_z, pdf_tot = calc_pdfs(
                    s,
                    ds,
                    "T",
                    "Lambda_cool",
                    zmin=0,
                    zmax=zmax,
                    force_override=False,
                    cooling=True,
                    species=True,
                    radiation=True,
                )
                pdf_z, pdf_tot = calc_pdfs(
                    s,
                    ds,
                    "nH",
                    "pok",
                    zmin=0,
                    zmax=zmax,
                    force_override=False,
                    cooling=True,
                    species=True,
                    radiation=True,
                )
                pdf_z, pdf_tot = calc_pdfs(
                    s,
                    ds,
                    "nH",
                    "T",
                    zmin=0,
                    zmax=zmax,
                    force_override=False,
                    cooling=True,
                    species=True,
                    radiation=True,
                )
            else:
                pdf_z, pdf_tot = calc_pdfs(
                    s,
                    ds,
                    "T",
                    "Lambda_cool",
                    zmin=0,
                    zmax=zmax,
                    force_override=False,
                    cooling=True,
                    species=False,
                    radiation=False,
                )
                pdf_z, pdf_tot = calc_pdfs(
                    s,
                    ds,
                    "nH",
                    "pok",
                    zmin=0,
                    zmax=zmax,
                    force_override=False,
                    cooling=True,
                    species=False,
                    radiation=False,
                )
                pdf_z, pdf_tot = calc_pdfs(
                    s,
                    ds,
                    "nH",
                    "T",
                    zmin=0,
                    zmax=zmax,
                    force_override=False,
                    cooling=True,
                    species=False,
                    radiation=False,
                )

        gc.collect()
