#!/usr/bin/env python

import os

# import os.path as osp
import gc
import time
from mpi4py import MPI
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pprint
import argparse
import sys
from matplotlib.patheffects import withStroke

import astropy.constants as ac
import astropy.units as au
import cmasher as cmr
from astropy.visualization import make_lupton_rgb

import pyathena as pa
from pyathena.plt_tools.plt_starpar import scatter_sp

# from pyathena.plt_tools.make_movie import make_movie
from pyathena.tigress_ncr.slc_prj import slc_to_xarray
from pyathena.tigress_ncr.do_tasks import scatter_nums

dirpath = os.path.dirname(__file__)

plt.style.use(f"{dirpath}/../../mplstyle/presentation-dark.mplstyle")
plt.rcParams["axes.grid"] = False
# plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["ytick.direction"] = "out"
plt.rcParams["ytick.minor.visible"] = False
plt.rcParams["xtick.direction"] = "out"
plt.rcParams["xtick.minor.visible"] = False

plt.rcParams['image.interpolation']='quadric'
plt.rcParams["savefig.pad_inches"]=0
plt.rcParams["axes.facecolor"] = (1, 1, 1, 0)
plt.rcParams["savefig.facecolor"] = (1, 1, 1, 0)
plt.rcParams["savefig.transparent"] = True
plt.rcParams["savefig.dpi"] = 200

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
labels["Sigma_FUV"] = r"$\Sigma_{\rm FUV}\,[{\rm erg\, pc^{-2}}]$"
labels["Sigma_LyC"] = r"$\Sigma_{\rm LyC}\,[{\rm erg\, pc^{-2}}]$"

def prepare_dfi(s):
    from pyathena.plt_tools import cmap

    def add_dfi(label, norm, cmap):
        tmp = dict(cbar_kwargs=dict())
        tmp["label"] = label
        tmp["norm"] = norm
        tmp["cmap"] = cmap
        return tmp

    my_dfi = s.dfi

    jh_cmap = cmap.get_cmap_jh_colors()
    my_dfi["T"]["cmap"] = jh_cmap

    my_dfi['chi_FUV']['cmap'] = cmr.amber
    my_dfi['chi_FUV']['vminmax'] = (1.e-1,1.e3)
    my_dfi['Erad_LyC']['cmap'] = cmr.cosmic
    my_dfi['Erad_LyC']['vminmax'] = (1.e-16,1.e-10)
    my_dfi["density"] = my_dfi["nH"]
    my_dfi["density"]["vminmax"] = (1.0e-4, 1.0e2)
    my_dfi["density"]["cmap"] = cmr.rainforest
    my_dfi["Sigma_gas"] = add_dfi(
        r"$\Sigma_{\rm gas}\,[M_\odot\,{\rm pc^{-2}}]$",
        LogNorm(1.0e-1, 1.0e2),
        cmr.rainforest,
    )
    my_dfi["Sigma_gas"]["vminmax"] = (1.0e-1, 1.0e2)

    my_dfi["EM"] = add_dfi(
        r"${\rm EM}\,[{\rm cm^{-6}\,pc}]$", LogNorm(1.0e-2, 1.0e4), plt.cm.plasma
    )
    my_dfi["EM"]["vminmax"] = (1.e-2,1.e4)

    return my_dfi

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
        "Erad_FUV",
        "Erad_LyC"
    ]
    prjdata = s.read_prj(num, fields=fields, force_override=False)

    if(("Sigma_HII" not in prjdata["x"]) or
       ("Sigma_gas" not in prjdata["x"]) or
       ("Sigma_FUV" not in prjdata["x"])):
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
        elif f == "Sigma_FUV":
            kwargs = dict(norm=LogNorm(1.e5,1.e15), cmap=cmr.rainforest)
        elif f == "Sigma_LyC":
            kwargs = dict(norm=LogNorm(1.e5,1.e15), cmap=cmr.rainforest)
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

def add_bar(ax,cmap,barlength = 200, dx = 4):
    bx0 = 0.8
    bx1 = 0.8+0.1953125
    by0 = 0.93
    ax.plot([bx0,bx1],[by0,by0],color='w',transform=ax.transAxes)
    ax.annotate(f"{barlength} pc",[bx1/2+bx0/2,by0-0.01],
                xycoords="axes fraction",ha="center",va="top",
                fontsize="x-small",color="k",
                path_effects=[withStroke(foreground="w", linewidth=2)])

def add_inset_cbar(fig,label,vmin,vmax,cmap):
    inax = fig.add_axes([0.8,0.93,0.1953125,0.03])
    plt.colorbar(cax=inax,shrink=0.5,pad=0,fraction=0.1,orientation='horizontal')

    inax.axis('off')
    inax.annotate(label,(0.5,0.96),
                xycoords="axes fraction",ha="center",va="bottom",
                fontsize="x-small",color="k",
                path_effects=[withStroke(foreground="w", linewidth=2)])
    inax.annotate(f"{vmin}",(0,0.5),xycoords="axes fraction",ha="left",va="center",
                fontsize=7,color=cmap(255),
                path_effects=[withStroke(foreground="w", linewidth=1)])
    inax.annotate(f"{vmax}",(1,0.5),xycoords="axes fraction",ha="right",va="center",
                fontsize=7,color=cmap(0),
                path_effects=[withStroke(foreground="w", linewidth=1)])


def plot_vector(slcdata,ax=None,vec = "v",scalefunc = np.log10,units=1,color="grey"):
    if ax is None:
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_axes([0,0,1,1])
        ax.axis('off')
    mag = np.sqrt(slcdata[f"{vec}x"].data**2+slcdata[f"{vec}y"].data**2)*units
    st = ax.streamplot(slcdata["x"].data,slcdata["y"].data,
                slcdata[f"{vec}x"].data,slcdata[f"{vec}y"].data,
                minlength=0.1,
                linewidth=np.clip(scalefunc(mag),0.1,None),
                color=color,density=4)

    return st
def plot_image(data,label,ax=None,cbar=True,vec=None,vmin=0,vmax=100,cmap = plt.cm.viridis,extent=None):
    if ax is None:
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_axes([0,0,1,1])
        ax.axis('off')
    else:
        fig = ax.get_figure()
    plt.imshow(data,vmin=vmin,vmax=vmax,cmap=cmap,extent=extent)
    if cbar:
        Lx = extent[1]-extent[0]
        add_inset_cbar(fig,label,vmin,vmax,cmap)
        add_bar(ax,cmap,barlength=0.1953125*Lx)
    plt.sca(ax)
    return ax

def get_data(s, num, proj_ax="z"):
    prjdata = s.read_prj(num)
    slcdata = s.read_slc_from_allslc(num,fields=["density","T","chi_FUV","Erad_LyC","Bx","By","vx","vy"])
    slcdata = slc_to_xarray(slcdata,axis=proj_ax)
    sp = s.read_starpar(num)
    prjdata[proj_ax]["extent"]=np.array(prjdata["extent"][proj_ax])
    extent = prjdata[proj_ax]["extent"]
    prjdata[proj_ax]["extent"] = extent
    return prjdata[proj_ax], slcdata, sp

def get_field(prjdata, slcdata, slc_field, my_dfi, log=True):
    if slc_field in slcdata:
        data = slcdata[slc_field]
    else:
        data = prjdata[slc_field]
    vmin,vmax = my_dfi["vminmax"]
    label = my_dfi["label"]
    cmap = my_dfi["cmap"]
    if log:
        data = np.log10(data)
        vmin = np.log10(vmin)
        vmax = np.log10(vmax)
        label = "$\log\,$"+label
    extent = prjdata["extent"]

    return data, dict(label=label,vmin=vmin,vmax=vmax,cmap=cmap,extent=extent)

def plot_newimage(s, num, outdir, proj_ax="z"):
    my_dfi = prepare_dfi(s)

    # load data
    prjdata, slcdata, sp = get_data(s,num,proj_ax=proj_ax)

    # plot star particles
    extent = prjdata["extent"]

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_axes([0,0,1,1])
    ax.set_xlim(extent[:2])
    ax.set_ylim(extent[:2])
    # ax.set_ylim(extent[2:])
    ax.axis('off')
    scatter_sp(
        sp["sp"], ax, proj_ax, kpc=False,
        norm_factor=3, u=s.u, agemax=40, alpha=0.7
    )
    plt.savefig(os.path.join(outdir,f"{s.problem_id}.{num:04d}.starpar-{proj_ax}.png"))

    # Surface density in linear scale
    for f in ["Sigma_gas","Sigma_H2","Sigma_HI","Sigma_HII"]:
        if f not in prjdata: continue
        data, params = get_field(prjdata,slcdata,f,my_dfi["Sigma_gas"],log=False)
        params["cmap"]=cmr.combine_cmaps(cmr.get_sub_cmap('cmr.neutral_r', 0., 0.85),cmr.get_sub_cmap('cmr.ember', 0.15, 1), nodes=[0.2])
        params["vmin"]=0
        params["vmax"]=50
        ax = plot_image(data,**params)
        ax.set_xlim(extent[:2])
        ax.set_ylim(extent[:2])
        plt.savefig(os.path.join(outdir,f"{s.problem_id}.{num:04d}.{f}-{proj_ax}-linear.png"))
        scatter_sp(
                sp["sp"], ax, proj_ax, kpc=False,
                norm_factor=3, u=s.u, agemax=40, alpha=0.7
            )
        ax.set_xlim(extent[:2])
        ax.set_ylim(extent[:2])
        plt.savefig(os.path.join(outdir,f"{s.problem_id}.{num:04d}.{f}-{proj_ax}-linear-sp.png"))

    # Surface density in log scale
    for f in ["Sigma_gas","Sigma_H2","Sigma_HI","Sigma_HII"]:
        if f not in prjdata: continue
        data, params = get_field(prjdata,slcdata,f,my_dfi["Sigma_gas"],log=True)
        ax = plot_image(data,**params)
        ax.set_xlim(extent[:2])
        ax.set_ylim(extent[:2])
        plt.savefig(os.path.join(outdir,f"{s.problem_id}.{num:04d}.{f}-{proj_ax}-log.png"))
        scatter_sp(
                sp["sp"], ax, proj_ax, kpc=False,
                norm_factor=3, u=s.u, agemax=40, alpha=0.7
            )
        ax.set_xlim(extent[:2])
        ax.set_ylim(extent[:2])
        plt.savefig(os.path.join(outdir,f"{s.problem_id}.{num:04d}.{f}-{proj_ax}-log-sp.png"))

    # other maps
    for f in ["density","T","Sigma_gas","EM","chi_FUV","Erad_LyC"]:
        data, params = get_field(prjdata,slcdata,f,my_dfi[f],log=True)
        ax = plot_image(data,**params)
        ax.set_xlim(extent[:2])
        ax.set_ylim(extent[:2])
        plt.savefig(os.path.join(outdir,f"{s.problem_id}.{num:04d}.{f}-{proj_ax}.png"))
        scatter_sp(
                sp["sp"], ax, proj_ax, kpc=False,
                norm_factor=3, u=s.u, agemax=40, alpha=0.7
            )
        ax.set_xlim(extent[:2])
        ax.set_ylim(extent[:2])
        plt.savefig(os.path.join(outdir,f"{s.problem_id}.{num:04d}.{f}-{proj_ax}-sp.png"))

    if proj_ax == "z":
        # streamlines
        plot_vector(slcdata,color=(1.,0,0,0.7),units=0.3)
        plt.savefig(os.path.join(outdir,f"{s.problem_id}.{num:04d}.vstream.png"))
        plot_vector(slcdata,vec="B",units=s.u.muG,scalefunc=np.log,color=(0,0.7,1,0.7))
        plt.savefig(os.path.join(outdir,f"{s.problem_id}.{num:04d}.Bstream.png"))

        # combined plot
        f="density"
        data, params = get_field(prjdata,slcdata,f,my_dfi[f],log=False)
        params["cmap"]=cmr.combine_cmaps(cmr.get_sub_cmap('cmr.neutral_r', 0., 0.85),cmr.get_sub_cmap('cmr.ember', 0.15, 1), nodes=[0.05])
        params["vmin"]=0
        params["vmax"]=20
        ax = plot_image(data,**params)
        plt.savefig(os.path.join(outdir,f"{s.problem_id}.{num:04d}.n.png"))
        plot_vector(slcdata,ax=ax,vec="B",units=s.u.muG,scalefunc=np.log,color=(0,0.7,1,0.7))
        ax.set_xlim(extent[:2])
        ax.set_ylim(extent[2:])
        plt.savefig(os.path.join(outdir,f"{s.problem_id}.{num:04d}.nB.png"))
        plot_vector(slcdata,ax=ax,color=(1.,0,0,0.7),units=0.3)
        ax.set_xlim(extent[:2])
        ax.set_ylim(extent[2:])
        plt.savefig(os.path.join(outdir,f"{s.problem_id}.{num:04d}.nvB.png"))

    # Sigma_RGB
    if "Sigma_HII" in prjdata:
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_axes([0,0,1,1])
        ax.axis("off")

        extent = prjdata["extent"]
        H2, params = get_field(prjdata,slcdata,"Sigma_H2",my_dfi["Sigma_gas"],log=False)
        HI, params = get_field(prjdata,slcdata,"Sigma_HI",my_dfi["Sigma_gas"],log=False)
        HII, params = get_field(prjdata,slcdata,"Sigma_HII",my_dfi["Sigma_gas"],log=False)

        rgb_default = make_lupton_rgb(4*H2, HI, HII*2, minimum=0.1, stretch=30,Q=8)
        plt.imshow(rgb_default, origin='lower',extent=extent)

        ax.set_xlim(extent[:2])
        ax.set_ylim(extent[:2])

        plt.savefig(os.path.join(outdir,f"{s.problem_id}.{num:04d}.Sigma_RGB.png"))
    plt.close("all")

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
    outdir2 = os.path.join(s.basedir, "map_images")
    os.makedirs(outdir2, exist_ok=True)

    # get my nums
    if s.nums is not None:
        mynums = scatter_nums(s, s.nums, COMM)
    else:
        mynums = []

    print("[rank, mynums]:", COMM.rank, mynums)

    time0 = time.time()
    for num in mynums:
        print(num, end=" ")
        do_surfmaps(s, num, outdir)
        for proj_ax in ["x","y","z"]:
            plot_newimage(s, num, outdir2, proj_ax=proj_ax)

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
