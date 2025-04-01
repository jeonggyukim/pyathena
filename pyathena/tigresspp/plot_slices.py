import sys

sys.path.insert(0, "../")
import os
import os.path as osp
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm, SymLogNorm
import cmasher as cmr
import numpy as np

from pyathena.tigresspp.load_sim_tigresspp import LoadSimTIGRESSPP
from pyathena.fields.fields import DerivedFields
from pyathena.plt_tools.make_movie import make_movie

from mpi4py import MPI


def plot_slice_xy(sim, slc, field, dfi, vec=None, stream_kwargs=dict(color="k")):
    try:
        dfi_ = dfi[field]
        data = dfi_["func"](slc, sim.u)
    except KeyError:
        return
    if field == "Fcr3":
        data = np.abs(data)
    im = plt.pcolormesh(
        data.x,
        data.y,
        data,
        cmap=dfi_["imshow_args"]["cmap"],
        norm=dfi_["imshow_args"]["norm"],
    )
    if vec is not None:
        vx = f"{vec}1"
        vy = f"{vec}2"
        plt.streamplot(
            slc.x.data, slc.y.data, slc[vx].data, slc[vy].data, **stream_kwargs
        )
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlim(sim.domain["le"][0], sim.domain["re"][0])
    plt.ylim(sim.domain["le"][1], sim.domain["re"][1])
    return im


def plot_slice_xz(sim, slc, field, dfi, vec=None, stream_kwargs=dict(color="k")):
    try:
        dfi_ = dfi[field]
        data = dfi_["func"](slc, sim.u)
    except KeyError:
        return
    if field == "Fcr3":
        data = np.abs(data)
    im = plt.pcolormesh(
        data.x,
        data.z,
        data,
        cmap=dfi_["imshow_args"]["cmap"],
        norm=dfi_["imshow_args"]["norm"],
    )
    if vec is not None:
        vx = f"{vec}1"
        vy = f"{vec}3"
        plt.streamplot(
            slc.x.data, slc.z.data, slc[vx].data, slc[vy].data, **stream_kwargs
        )
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlim(sim.domain["le"][0], sim.domain["re"][0])
    plt.ylim(sim.domain["le"][2], sim.domain["re"][2])
    return im


def plot_slices(sim, num, savefig=True):
    slc_xy = sim.get_slice(num, "allslc.z", slc_kwargs=dict(z=0, method="nearest"))
    slc_xz = sim.get_slice(num, "allslc.y", slc_kwargs=dict(y=0, method="nearest"))
    xz_ratio = sim.domain["Lx"][2] / sim.domain["Lx"][1]
    flist = [
        "nH",
        "T",
        "vmag",
        "Vcr_mag",
        "sigma_para",
        "pok_cr_inj",
        "pok_cr",
        "pok",
        "Fcr3",
        "Bmag",
        "Zgas",
        "rret"
    ]
    vectors = [
        None,
        None,
        "velocity",
        "0-Vc",
        None,
        None,
        None,
        None,
        "0-Fc",
        "cell_centered_B",
        None,
        None
    ]
    nf = len(flist)
    fig, axes = plt.subplots(
        2,
        nf,
        figsize=(2 * nf, 2 * (1 + xz_ratio)),
        gridspec_kw=dict(height_ratios=[xz_ratio, 1]),
        constrained_layout=True,
        num=0,
    )
    for ax, field, vec in zip(axes[0, :], flist, vectors):
        plt.sca(ax)
        im = plot_slice_xz(
            sim,
            slc_xz,
            field,
            sim.dfi,
            vec=vec,
            stream_kwargs=dict(color="w" if field == "Bmag" else "k"),
        )
        if im is not None:
            cbar = plt.colorbar(
                im,
                orientation="horizontal",
                location="top",
                pad=0.01,
                shrink=0.8,
                aspect=10,
                label=sim.dfi[field]["label"],
            )
            ax.set_aspect("equal", adjustable="box")
        # ax.set_title(field)
        ax.axis("off")
    for ax, field, vec in zip(axes[1, :], flist, vectors):
        plt.sca(ax)
        im = plot_slice_xy(
            sim,
            slc_xy,
            field,
            sim.dfi,
            vec=vec,
            stream_kwargs=dict(density=0.5, color="w" if field == "Bmag" else "k"),
        )
        if im is not None:
            ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
    tMyr = slc_xy.attrs["time"] * sim.u.Myr
    axes[0, 0].annotate(
        f"t={tMyr:.2f} Myr",
        (0.5, 0.99),
        ha="center",
        va="top",
        xycoords="axes fraction",
        fontsize="x-large",
        bbox=dict(boxstyle="round,pad=0.2", fc="w", ec="k", lw=1),
    )

    if savefig:
        savdir = osp.join(sim.savdir, "cr_snapshot")
        if not osp.exists(savdir):
            os.makedirs(savdir, exist_ok=True)

        savname = osp.join(savdir, "{0:s}_{1:04d}.png".format(sim.basename, num))
        plt.savefig(savname, dpi=200, bbox_inches="tight")
    return fig
    # plt.tight_layout()

def plot_projections(sim, num, savefig=True):
    prjdata = sim.get_prj(num,"y",prefix="prj.y")
    prjkwargs,labels = sim.set_prj_dfi()
    xz_ratio = sim.domain["Lx"][2] / sim.domain["Lx"][1]
    flist = ["Sigma","mflux","teflux","keflux","creflux"]
    phlist = ["wc","hot"]
    nf = len(flist)
    nphase = len(phlist)
    vectors = [None]*nf
    size = 2
    fig, axes = plt.subplots(
        1,
        nf*nphase,
        figsize=(size * nf * nphase , size * (xz_ratio)),
        # gridspec_kw=dict(height_ratios=[xz_ratio, 1]),
        constrained_layout=True,
        num=0,
    )

    for i,phase in enumerate(phlist):
        for ax, field, vec in zip(axes[i::nphase], flist, vectors):
            plt.sca(ax)
            ax.axis("off")
            try:
                prj = prjdata[field].sel(phase=phase)
            except KeyError:
                continue
            im = plt.pcolormesh(
                prj.x,
                prj.z,
                prj,
                **prjkwargs[field]
            )
            if im is not None:
                cbar = plt.colorbar(
                    im,
                    orientation="horizontal",
                    location="top",
                    pad=0.01,
                    shrink=0.8,
                    aspect=10,
                    label=labels[field]+f"\n{phase}",
                )
                ax.set_aspect("equal", adjustable="box")
    tMyr = prjdata.attrs["time"] * sim.u.Myr
    axes[0].annotate(
        f"t={tMyr:.2f} Myr",
        (0.5, 0.99),
        ha="center",
        va="top",
        xycoords="axes fraction",
        fontsize="x-large",
        bbox=dict(boxstyle="round,pad=0.2", fc="w", ec="k", lw=1),
    )

    if savefig:
        savdir = osp.join(sim.savdir, "cr_snapshot_prj")
        if not osp.exists(savdir):
            os.makedirs(savdir, exist_ok=True)

        savname = osp.join(savdir, "{0:s}_{1:04d}.png".format(sim.basename, num))
        plt.savefig(savname, dpi=200, bbox_inches="tight")
    return fig

if __name__ == "__main__":
    spp = LoadSimTIGRESSPP(sys.argv[1])
    spp.update_derived_fields()

    COMM = MPI.COMM_WORLD
    mynums = [spp.nums[i] for i in range(len(spp.nums)) if i % COMM.size == COMM.rank]
    print(COMM.rank, mynums)

    for num in mynums:
        print(num)
        f = plot_slices(spp, num)
        plt.close(f)
        f = plot_projections(spp, num)
        plt.close(f)

# Make movies
    COMM.barrier()

    if COMM.rank == 0:
        if not osp.isdir(osp.join(spp.basedir, "movies")):
            os.mkdir(osp.join(spp.basedir, "movies"))
        fin = osp.join(spp.basedir, "cr_snapshot/*.png")
        fout = osp.join(spp.basedir, "movies/{0:s}_cr_snapshot.mp4".format(spp.basename))
        try:
            make_movie(fin, fout, fps_in=15, fps_out=15)
        except FileNotFoundError:
            pass

        if not osp.isdir(osp.join(spp.basedir, "movies")):
            os.mkdir(osp.join(spp.basedir, "movies"))
        fin = osp.join(spp.basedir, "cr_snapshot_prj/*.png")
        fout = osp.join(spp.basedir, "movies/{0:s}_cr_snapshot_prj.mp4".format(spp.basename))
        try:
            make_movie(fin, fout, fps_in=15, fps_out=15)
        except FileNotFoundError:
            pass