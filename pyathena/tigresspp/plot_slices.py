import sys

import os
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np

from pyathena.tigresspp.load_sim_tigresspp import LoadSimTIGRESSPP
from pyathena.plt_tools.make_movie import make_movie
from pyathena.plt_tools.plt_starpar import scatter_sp

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import ImageGrid
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
        # "pok_cr_inj",
        "pok_cr",
        "pok_trbz",
        "pok",
        # "Fcr3",
        "pok_mag",
        # "Zgas",
        # "rret"
    ]
    vectors = [
        None,
        None,
        "velocity",
        "0-Vs",
        None,
        # None,
        None,
        None,
        None,
        # "0-Fc",
        "cell_centered_B",
        # None,
        # None
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
            stream_kwargs=dict(color="w" if field in ["Bmag","pok_mag"] else "k"),
        )
        if im is not None:
            cbar = plt.colorbar(
                im,
                orientation="horizontal",
                location="top",
                pad=0.01,
                shrink=0.8,
                aspect=10,
                label=sim.dfi[field]["label_unit"],
            )
            ax.set_aspect("equal", adjustable="box")
        # ax.set_title(field)
            ax.annotate(sim.dfi[field]["label_name"],
                (0.5, 1.01),
                ha="center",
                va="bottom",
                xycoords="axes fraction",
                fontsize=40,
                bbox=dict(boxstyle="round,pad=0.2", fc="w", ec="k", lw=1,alpha=1),)
        ax.axis("off")
    for ax, field, vec in zip(axes[1, :], flist, vectors):
        plt.sca(ax)
        im = plot_slice_xy(
            sim,
            slc_xy,
            field,
            sim.dfi,
            vec=vec,
            stream_kwargs=dict(density=0.5, color="w" if field in ["Bmag","pok_mag"] else "k"),
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
        savdir = osp.join(sim.savdir, "cr_snapshot2")
        if not osp.exists(savdir):
            os.makedirs(savdir, exist_ok=True)

        savname = osp.join(savdir, "{0:s}_{1:04d}.png".format(sim.basename, num))
        plt.savefig(savname, dpi=200, bbox_inches="tight")
    return fig
    # plt.tight_layout()

def plot_slices_ncr(sim, num, savefig=True):
    slc_xy = sim.get_slice(num, "allslc.z", slc_kwargs=dict(z=0, method="nearest"))
    slc_xz = sim.get_slice(num, "allslc.y", slc_kwargs=dict(y=0, method="nearest"))
    xz_ratio = sim.domain["Lx"][2] / sim.domain["Lx"][1]
    flist = [
        "nH",
        "xe",
        "T",
        "pok",
        "vmag",
        "Bmag",
        "Zgas",
        "rret",
        "Lambda_cool",
        "cool_rate_cgs",
        "heat_rate_cgs",
    ]
    vectors = [
        None,
        None,
        None,
        None,
        "velocity",
        "cell_centered_B",
        None,
        None,
        None,
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
            stream_kwargs=dict(color="w"),
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
            stream_kwargs=dict(density=0.5, color="w"),
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
        savdir = osp.join(sim.savdir, "ncr_snapshot")
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


def plot_projections_ncr(sim, num, savefig=True):
    prjdata = sim.get_prj(num,"y",prefix="prj.y")
    prjkwargs,labels = sim.set_prj_dfi()
    xz_ratio = sim.domain["Lx"][2] / sim.domain["Lx"][1]
    flist = ["Sigma","Sigma_HI","Sigma_HII","EM","teflux","keflux"]
    phlist = ["whole"]
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
        savdir = osp.join(sim.savdir, "ncr_snapshot_prj")
        if not osp.exists(savdir):
            os.makedirs(savdir, exist_ok=True)

        savname = osp.join(savdir, "{0:s}_{1:04d}.png".format(sim.basename, num))
        plt.savefig(savname, dpi=200, bbox_inches="tight")
    return fig

def plot_snapshot(sim, num,
                  outid=None,
                  parnum=None,
                fields_xy=('Sigma', 'nH', 'T', 'rret', 'pok', 'Bmag',),
                fields_xz=('Sigma', 'nH', 'T', 'rret', 'pok', 'Bmag', 'vz'),
                sink_fields=('Sigma', 'nH'),
                norm_factor=5.0,
                agemax=40.0,
                savefig=False):
    """Plot 12-panel projection/slice plots in the z and y directions

    Parameters
    ----------
    num : int
        vtk snapshot number
    fields_xy: list of str
        Field names for z projections and slices
    fields_xz: list of str
        Field names for y projections and slices
    sink_fields: list of str
        Field names on which sink particles are plotted
    norm_factor : float
        Normalization factor for starpar size. Smaller norm_factor for bigger size.
    agemax : float
        Maximum age of radiation source particles [Myr]
    """
    # read slice/projection/star particle data
    slc_xy = sim.get_slice(num, "allslc.z", outid=outid,
                           slc_kwargs=dict(z=0, method="nearest"))
    slc_xz = sim.get_slice(num, "allslc.y", outid=outid,
                           slc_kwargs=dict(y=0, method="nearest"))
    prj_xy = sim.get_prj(num, "z", prefix="prj.z", outid=outid)
    prj_xz = sim.get_prj(num, "y", prefix="prj.y", outid=outid)
    if parnum is None:
        parnum = num
    sp = sim.load_parbin(parnum)
    prjkwargs,labels = sim.set_prj_dfi()

    # setup figure and axes grid
    nxy = len(fields_xy)
    nxz = len(fields_xz)
    LzoLx = sim.domain['Lx'][2]/sim.domain['Lx'][0]
    xwidth = 1.5
    ysize = LzoLx*xwidth
    xsize = ysize/nxy*4 + nxz*xwidth
    x1 = 0.90*(ysize*4/nxy/xsize)
    x2 = 0.90*(nxz*xwidth/xsize)

    fig = plt.figure(figsize=(xsize, ysize), num=0)
    g1 = ImageGrid(fig, [0.02, 0.05, x1, 0.94], (nxy//2, 2), axes_pad=0.1,
                aspect=True, share_all=True, direction='column')
    g2 = ImageGrid(fig, [x1+0.07, 0.05, x2, 0.94], (1, nxz), axes_pad=0.1,
                # cbar_mode="each",
                # cbar_location="top",
                # cbar_size="7%",
                # cbar_pad="2%",
                aspect=True, share_all=True)

    # get domain range
    le = sim.domain["le"]
    re = sim.domain["re"]

    # loop over xy plots
    for i, (ax, f) in enumerate(zip(g1, fields_xy)):
        plt.sca(ax)
        if f in prj_xy:
            prj = prj_xy[f].sel(phase="whole")
            im = plt.pcolormesh(
                prj.x,
                prj.y,
                prj,
                **prjkwargs[f]
            )
        else:
            plot_slice_xy(sim,slc_xy,f,sim.dfi)

        if f in sink_fields:
            scatter_sp(sp, ax, 'z', kind='prj', kpc=False,
                    norm_factor=norm_factor, agemax=agemax,
                    cmap=plt.cm.cool_r)
        ax.set(xlim=(le[0],re[0]), ylim=(le[1],re[1]))
        if i == 2:
            ax.set(xlabel='x [pc]', ylabel='y [pc]')
        else:
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)

    # loop over xz plots
    for i, (ax, f) in enumerate(zip(g2, fields_xz)):
        plt.sca(ax)
        ax.set_aspect("equal", adjustable="box")
        if f in prj_xz:
            prj = prj_xz[f].sel(phase="whole")
            im = plt.pcolormesh(
                prj.x,
                prj.z,
                prj,
                **prjkwargs[f]
            )
            label = labels[f]
        else:
            im = plot_slice_xz(sim,slc_xz,f,sim.dfi)
            try:
                label=sim.dfi[f]["label"]
            except KeyError:
                label = f
        cax = inset_axes(ax, width="80%", height="2%", loc="upper center")

        cbar = plt.colorbar(
            im,
            orientation="horizontal",
            location="top",
            cax=cax,
            # cax=g2.cbar_axes[i],
            pad=0.01,
            shrink=0.8,
            aspect=10,
            label=label
        )

        if f in sink_fields:
            scatter_sp(sp, ax, 'y', kind='prj', kpc=False,
                    norm_factor=norm_factor, agemax=agemax,
                    cmap=plt.cm.cool_r)
        ax.set(xlim=(le[0],re[0]), ylim=(le[2],re[2]))
        if i == 0:
            ax.set(xlabel='x [pc]', ylabel='z [pc]')
        else:
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)

    plt.sca(g1[0])
    time = slc_xy.attrs["time"]
    plt.annotate(f'Model: {sim.basename}  time={time*sim.u.Myr:6.1f} Myr',
                (0,1.),xycoords="axes fraction",
                va='bottom', ha='left')
    if savefig:
        savdir = osp.join(sim.savdir, 'snapshot')
        if outid is not None:
            savdir = osp.join(sim.savdir, f'snapshot_out{outid:d}')
        os.makedirs(savdir,exist_ok=True)
        savname = osp.join(savdir, f'snapshot_{num:05d}.png')
        plt.savefig(savname, dpi=200, bbox_inches='tight')

    return fig

if __name__ == "__main__":
    spp = LoadSimTIGRESSPP(sys.argv[1])
    spp.update_derived_fields()
    spp.dt = [spp.par[f"output{i+1}"]["dt"] for i,v in enumerate(spp.out_fmt)]

    COMM = MPI.COMM_WORLD
    mynums = [spp.nums[i] for i in range(len(spp.nums)) if i % COMM.size == COMM.rank]
    print(COMM.rank, mynums)

    for num in mynums:
        print(num)
        if spp.options["newcool"]:
            f = plot_slices_ncr(spp, num)
            plt.close(f)
            f = plot_projections_ncr(spp, num)
            plt.close(f)
            head="ncr"
        else:
            f = plot_slices(spp, num)
            plt.close(f)
            f = plot_projections(spp, num)
            plt.close(f)
            head="cr"

    for k, v in zip(spp.hdf5_outid, spp.hdf5_outvar):
        nums = spp.nums_hdf5[v]
        pardt = spp.dt[spp.parbin_outid-1]
        mydt = spp.dt[k-1]

        mynums = [nums[i] for i in range(len(nums)) if i % COMM.size == COMM.rank]
        for num in mynums:
            parnum = num//(pardt/mydt)
            if (v == spp._hdf5_outvar_def):
                f = plot_snapshot(spp,num,parnum=parnum,savefig=True)
            else:
                f = plot_snapshot(spp,num,outid=k,parnum=parnum,
                                fields_xy=('Sigma', 'nH', 'T', 'pok'),
                                fields_xz=('Sigma', 'nH', 'T', 'pok'),
                                savefig=True)
            plt.close(f)

# Make movies
    COMM.barrier()

    if COMM.rank == 0:
        if not osp.isdir(osp.join(spp.basedir, "movies")):
            os.mkdir(osp.join(spp.basedir, "movies"))
        fin = osp.join(spp.basedir, f"{head}_snapshot/*.png")
        fout = osp.join(spp.basedir, f"movies/{spp.basename}_{head}_snapshot.mp4")
        try:
            make_movie(fin, fout, fps_in=15, fps_out=15)
        except FileNotFoundError:
            pass

        if not osp.isdir(osp.join(spp.basedir, "movies")):
            os.mkdir(osp.join(spp.basedir, "movies"))
        fin = osp.join(spp.basedir, f"{head}_snapshot_prj/*.png")
        fout = osp.join(spp.basedir, f"movies/{spp.basename}_{head}_snapshot_prj.mp4")
        try:
            make_movie(fin, fout, fps_in=15, fps_out=15)
        except FileNotFoundError:
            pass

        if not osp.isdir(osp.join(spp.basedir, "movies")):
            os.mkdir(osp.join(spp.basedir, "movies"))
        fin = osp.join(spp.basedir, f"snapshot/*.png")
        fout = osp.join(spp.basedir, f"movies/{spp.basename}_snapshot.mp4")
        try:
            make_movie(fin, fout, fps_in=15, fps_out=15)
        except FileNotFoundError:
            pass