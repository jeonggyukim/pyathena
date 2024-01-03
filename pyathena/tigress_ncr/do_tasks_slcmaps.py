#!/usr/bin/env python

import os
import gc
import time
from mpi4py import MPI
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
import pandas as pd
import pprint
import argparse
import sys

import cmasher as cmr

import pyathena as pa
from pyathena.util.split_container import split_container
from pyathena.tigress_ncr.ncr_paper_lowz import athena_data
from pyathena.tigress_ncr.phase import (
    assign_phase,
    draw_phase,
    recal_xT,
    add_phase_cuts,
)

dirpath = os.path.dirname(__file__)

plt.style.use(f"{dirpath}/../../mplstyle/presentation-dark.mplstyle")
plt.rcParams["axes.grid"] = False
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["ytick.direction"] = "out"
plt.rcParams["ytick.minor.visible"] = False
plt.rcParams["xtick.direction"] = "out"
plt.rcParams["xtick.minor.visible"] = False
plt.rcParams["figure.figsize"] = (4, 3)
plt.rcParams["font.size"] = 12


def set_my_dfi(dfi):
    def add_dfi(label, norm, cmap):
        tmp = dict(cbar_kwargs=dict())
        tmp["cbar_kwargs"]["label"] = label
        tmp["norm"] = norm
        tmp["cmap"] = cmap
        return tmp

    from pyathena.plt_tools import cmap

    jh_cmap = cmap.get_cmap_jh_colors()

    my_dfi = dict()
    for f in dfi:
        my_dfi[f] = dfi[f]["imshow_args"]
    my_dfi["density"] = my_dfi["nH"]
    my_dfi["density"]["norm"] = LogNorm(1.0e-4, 1.0e2)
    my_dfi["density"]["cmap"] = cmr.rainforest
    my_dfi["Sigma_gas"] = add_dfi(
        r"$\Sigma_{\rm gas}\,[M_\odot\,{\rm pc^{-2}}]$",
        LogNorm(1.0e-1, 1.0e2),
        plt.cm.pink_r,
    )
    my_dfi["xe"]["norm"] = Normalize(0, 1.2)
    my_dfi["xe"]["cmap"] = plt.cm.ocean_r
    my_dfi["xHI"]["norm"] = LogNorm(1.0e-4, 1.0)
    my_dfi["EM"] = add_dfi(
        r"${\rm EM}\,[{\rm cm^{-6}\,pc}]$", LogNorm(1.0e-2, 1.0e4), plt.cm.plasma
    )
    if "chi_FUV" in my_dfi:
        my_dfi["chi_FUV"]["cmap"] = cmr.amber
        my_dfi["chi_FUV"]["norm"] = LogNorm(1.0e-1, 1.0e3)
    if "Erad_LyC" in my_dfi:
        my_dfi["Erad_LyC"]["cmap"] = cmr.cosmic
        my_dfi["Erad_LyC"]["norm"] = LogNorm(1.0e-16, 1.0e-10)
    my_dfi["T"]["cmap"] = jh_cmap
    my_dfi["T"]["norm"] = LogNorm(1.0e1, 1.0e7)
    my_dfi["cool_rate"]["norm"] = LogNorm(1.0e-27, 1.0e-20)
    my_dfi["cool_rate"]["cmap"] = cmr.get_sub_cmap(cmr.freeze_r, 0.0, 0.7)
    my_dfi["heat_rate"]["norm"] = LogNorm(1.0e-27, 1.0e-20)
    my_dfi["heat_rate"]["cmap"] = cmr.get_sub_cmap(cmr.flamingo_r, 0.0, 0.7)
    my_dfi["net_cool_rate"] = add_dfi(
        r"$\mathcal{L-G}\;[{\rm erg}\,{\rm cm^{-3}}\,{\rm s}^{-1}]$",
        SymLogNorm(1.0e-27, vmin=-1.0e-20, vmax=1.0e-20),
        cmr.get_sub_cmap(cmr.fusion, 0.1, 0.9),
    )
    # my_dfi['Uion']=add_dfi(r'$U_{\rm ion}$',LogNorm(1.e-6,1.e-2),cmr.dusk)
    return my_dfi


def draw_xT(s, slcdata, pdfcmap, log=False):
    hist_bin = recal_xT(slcdata)
    if log:
        h, xbin, ybin = hist_bin[2]  # log
    else:
        h, xbin, ybin = hist_bin[0]  # linear
    dx = xbin[1] - xbin[0]
    dy = ybin[1] - ybin[0]
    h = h / h.sum() / dx / dy

    plt.pcolormesh(xbin, ybin, h.T, norm=LogNorm(1.0e-5, 50), cmap=pdfcmap)
    add_phase_cuts(
        Tlist=s.get_phase_Tlist(), lkwargs=dict(color="w", ls="--", lw=1), log=log
    )
    plt.xlim(xbin.min(), xbin.max())
    plt.ylim(ybin.min(), ybin.max())
    ylabel = r"$x_{H}$"
    plt.ylabel(ylabel)
    plt.xlabel(r"$\log\,T\,[{\rm K}]$")
    if log:
        plt.ylim(-3, 0)
    else:
        plt.ylim(-0.01, 1.01)

    plt.xlim(1, 7)
    plt.xlim(2, 6)
    cbar = plt.colorbar()
    cbar.set_label(
        r"$\frac{d^2 f_M}{dx_{H}\,d\log\,T}\,[{\rm dex^{-2}}]$",
        fontsize="large",
    )
    plt.legend(loc=4, fontsize="xx-small", frameon=False)


def do_slcmaps(s, num, outdir, axis="z"):
    dfi = set_my_dfi(s.dfi)
    pdfcmap = cmr.get_sub_cmap(cmr.seasons_s, 0.6, 1.0)

    slcds = s.read_slc_xarray(num, fields=None, axis=axis)
    for ax in ["x","y","z"]:
        if ax in slcds: slcds=slcds.assign({ax:slcds[ax]*1.e-3})
    zmax = s.domain['Lx'][2]*0.25*1.e-3
    xmax = s.domain['Lx'][1]*0.5*1.e-3
    slcdata = athena_data(s, slcds)

    if s.config_time < pd.to_datetime("2022-02-10 13:21:32 -0500"):
        cooling_rate_unit = 1.0
    else:
        cooling_rate_unit = (s.u.energy_density / s.u.time).cgs.value

    tMyr = slcdata["time"].data * s.u.Myr
    plt_kwargs = dict()
    for f in dfi:
        plt_kwargs[f] = dict(
            norm=dfi[f]["norm"],
            cmap=dfi[f]["cmap"],
            cbar_kwargs=dict(label=dfi[f]["cbar_kwargs"]["label"]),
        )

    flist = list(dfi.keys())
    flist = ["nH","ne","nHII","temperature","chi_FUV","Erad_LyC",
             "pok","xe","xi_CR","Bmag","pok_mag",
             "cool_rate","heat_rate","net_cool_reat"]
    # flist = ["nH"]
    for f in flist + ["phase", "xTpdf"]:
        plt.clf()
        if axis == "z" or f == "phase":
            plt.figure(num=0)
        else:
            plt.figure(num=0, figsize=(4,zmax/xmax*3))
        if f == "phase":
            ph = assign_phase(s, slcdata, kind="six")
            draw_phase(ph)
            plt.gca().set_aspect('equal')
            if axis != "z":
                plt.ylim(-zmax,zmax)
        elif f == "xTpdf":
            draw_xT(s, slcdata, pdfcmap, log=False)
        else:
            try:
                data = slcdata[f]
                if f in ["cool_rate", "heat_rate", "net_cool_rate"]:
                    data *= cooling_rate_unit
            except KeyError:
                continue
            data.plot(**plt_kwargs[f])
            if axis != "z":
                plt.ylim(-zmax,zmax)
            plt.gca().set_aspect('equal')
        plt.title(f"t = {tMyr:5.1f} Myr")
        plt.savefig(
            os.path.join(outdir, f"{f}.{num:04d}.{axis}.png"), bbox_inches="tight", dpi=200
        )


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
    outdir = os.path.join(s.basedir, "slcmaps")
    os.makedirs(outdir, exist_ok=True)

    # tar vtk files
    if s.nums_rawtar is not None:
        nums = s.nums_rawtar
        if COMM.rank == 0:
            print("basedir, nums", s.basedir, nums)
            nums = split_container(nums, COMM.size)
        else:
            nums = None

        mynums = COMM.scatter(nums, root=0)
        for num in mynums:
            s.create_tar(num=num, kind="vtk", remove_original=True, overwrite=True)
        COMM.barrier()

        # reading it again
        s = pa.LoadSimTIGRESSNCR(basedir, verbose=False)

    nums = s.nums_starpar

    if COMM.rank == 0:
        print("basedir, nums", s.basedir, nums)
        nums = split_container(nums, COMM.size)
    else:
        nums = None

    mynums = COMM.scatter(nums, root=0)
    print("[rank, mynums]:", COMM.rank, mynums)

    time0 = time.time()
    for num in mynums:
        print(num, end=" ")
        do_slcmaps(s, num, outdir, axis="z")
        do_slcmaps(s, num, outdir, axis="y")
        do_slcmaps(s, num, outdir, axis="x")

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
