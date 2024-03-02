#!/usr/bin/env python

import os

# import os.path as osp
import gc
import time
from mpi4py import MPI
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pprint
import argparse
import sys

import astropy.constants as ac
import astropy.units as au
import cmasher as cmr

import pyathena as pa

# from pyathena.plt_tools.make_movie import make_movie
from pyathena.tigress_ncr.slc_prj import slc_to_xarray
from pyathena.tigress_ncr.do_tasks import scatter_nums

dirpath = os.path.dirname(__file__)

plt.style.use(f"{dirpath}/../../mplstyle/presentation-dark.mplstyle")
plt.rcParams["axes.grid"] = False
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["ytick.direction"] = "out"
plt.rcParams["ytick.minor.visible"] = False
plt.rcParams["xtick.direction"] = "out"
plt.rcParams["xtick.minor.visible"] = False

labels = dict()
labels["Sigma_gas"] = r"$\Sigma_{\rm gas}\,[M_\odot\,{\rm pc^{-2}}]$"
labels["Sigma_HI"] = r"$\Sigma_{\rm H}\,[M_\odot\,{\rm pc^{-2}}]$"
labels["Sigma_HII"] = r"$\Sigma_{\rm H^+}\,[M_\odot\,{\rm pc^{-2}}]$"
labels["Sigma_H2"] = r"$\Sigma_{\rm H_2}\,[M_\odot\,{\rm pc^{-2}}]$"
labels["Sigma_e"] = r"$\Sigma_{\rm e}\,[M_\odot\,{\rm pc^{-2}}]$"
labels["Sigma_scalar0"] = r"$\Sigma_{\rm Z}\,[M_\odot\,{\rm pc^{-2}}]$"
labels["Sigma_scalar1"] = r"$\Sigma_{\rm Z,SN}\,[M_\odot\,{\rm pc^{-2}}]$"
labels["ne2bar"] = r"$<n_e^2>/<n_e>\,[{\rm cm^{-3}}]$"
labels["nebar"] = r"$<n_e>\,[{\rm cm^{-3}}]$"
labels["EM"] = r"${\rm EM}\,[{\rm cm^{-6}\,pc}]$"


def do_surfmaps(s, num, outdir):
    # some constants
    dx = s.domain["dx"][0] * s.u.length
    conv_Sigma = (dx * s.u.muH * ac.u.cgs / au.cm**3).to("Msun/pc**2").value
    conv_EM = (dx * au.cm**-6).to("pc cm-6").value

    # update proejction files
    fields = [
        "density",
        "xHI",
        "xH2",
        "xHII",
        "xe",
        "nesq",
        "specific_scalar[0]",
        "specific_scalar[1]",
    ]
    prjdata = s.read_prj(num, fields=fields, force_override=False)

    if ("Sigma_HII" not in prjdata["x"]) or ("Sigma_gas" not in prjdata["x"]):
        prjdata = s.read_prj(num, fields=fields, force_override=True)

    prjdata = slc_to_xarray(prjdata)
    prjdata["nebar"] = (prjdata["Sigma_e"] / conv_Sigma) / s.domain["Nx"][2]
    prjdata["ne2bar"] = (prjdata["EM"] / conv_EM) / (prjdata["Sigma_e"] / conv_Sigma)

    # draw maps
    surf0 = s.par["problem"]["surf"]
    n0 = surf0 / 10
    Lz = s.domain["Lx"][2]
    EM0 = n0**2 * Lz
    for f in prjdata:
        d = prjdata[f]
        fig = plt.figure(num=0, figsize=(6, 4))
        if f == "EM":
            kwargs = dict(norm=LogNorm(EM0 * 1.0e-4, EM0 * 100), cmap=plt.cm.plasma)
        elif f.startswith("ne"):
            kwargs = dict(norm=LogNorm(n0 * 1.0e-4, n0 * 100), cmap=cmr.rainforest)
        else:
            kwargs = dict(norm=LogNorm(surf0 * 0.01, surf0 * 10), cmap=plt.cm.pink_r)
        d.plot(cbar_kwargs=dict(label=labels[f]), **kwargs)
        time = d["time"].data * s.u.Myr
        plt.title(f"$t={time:5.1f} {{\\rm Myr}}$")
        plt.xlabel(r"$x\, [{\rm pc}]$")
        plt.ylabel(r"$y\, [{\rm pc}]$")

        # save maps
        fig.savefig(
            os.path.join(outdir, f"{f}.{int(num):04d}.png"),
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)


if __name__ == "__main__":
    COMM = MPI.COMM_WORLD

    basedir = "/tigress/changgoo/TIGRESS-NCR/R8_4pc_NCR"

    savdir = None
    savdir_pkl = None

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--basedir", type=str, default=basedir, help="Name of the basedir."
    )
    args = vars(parser.parse_args())
    locals().update(args)

    s = pa.LoadSimTIGRESSNCR(basedir, verbose=False)

    # create folder
    outdir = os.path.join(s.basedir, "surfmaps")
    os.makedirs(outdir, exist_ok=True)

    # get my nums
    if s.nums is not None:
        mynums = scatter_nums(s, s.nums_starpar)
    else:
        mynums = []

    print("[rank, mynums]:", COMM.rank, mynums)

    time0 = time.time()
    for num in mynums:
        print(num, end=" ")
        do_surfmaps(s, num, outdir)

        n = gc.collect()
        print("Unreachable objects:", n, end=" ")
        print("Remaining Garbage:", end=" ")
        pprint.pprint(gc.garbage)
        sys.stdout.flush()

    # # Make movies
    # COMM.barrier()

    # if COMM.rank == 0:
    #     if not osp.isdir(osp.join(s.basedir, "movies")):
    #         os.mkdir(osp.join(s.basedir, "movies"))
    #     for field in labels:
    #         fin = osp.join(outdir, "{field}.*.png")
    #         fout = osp.join(s.basedir, f"movies/{s.basename}_{field}.mp4")
    #         make_movie(fin, fout, fps_in=15, fps_out=15)
    #         from shutil import copyfile

    #         copyfile(
    #             fout,
    #             osp.join(
    #                 "/tigress/changgoo/public_html/temporary_movies/TIGRESS-NCR",
    #                 osp.basename(fout),
    #             ),
    #         )
