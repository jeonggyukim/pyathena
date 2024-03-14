import matplotlib.pyplot as plt
from pyathena.plt_tools.plt_starpar import scatter_sp, colorbar_sp, legend_sp
import cmasher as cmr

# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import SymLogNorm, LogNorm, Normalize
import matplotlib.gridspec as gridspec

# import pickle
import os
import numpy as np
import pyathena as pa
import pandas as pd


def decorate_ax(text, ax, iscbar=False):
    if iscbar:
        ax.annotate(
            text,
            (0.5, 1.02),
            ha="center",
            va="bottom",
            weight="bold",
            xycoords="axes fraction",
            fontsize="x-large",
        )
    ax.axis("off")


def set_axis_color(ax, xy, color):
    """
    Change spines, label, tick colors
    """
    if "x" in xy:
        ax.tick_params(axis="x", colors=color, which="both")
        ax.xaxis.label.set_color(color)
        ax.spines["bottom"].set_color(color)
        ax.spines["top"].set_color(color)
    if "y" in xy:
        ax.spines["left"].set_color(color)
        ax.spines["right"].set_color(color)
        ax.tick_params(axis="y", colors=color, which="both")
        ax.yaxis.label.set_color(color)


def get_sn(s, ds, dt=10, dz=50, z0=0):
    sn = pa.read_hst(s.files["sn"])
    sn.index = sn.time
    time = ds.domain["time"]
    snsel = sn[time - dt : time + 0.1]
    snsel = snsel.where(np.abs(z0 - snsel["x3"]) < dz).dropna()
    return snsel


# setup field attributes -- modified from JGK's field information
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
    my_dfi['chi_FUV']['cmap'] = cmr.amber
    my_dfi['chi_FUV']['norm'] = LogNorm(1.e-1, 1.e3)
    my_dfi['Erad_LyC']['cmap'] = cmr.cosmic
    my_dfi['Erad_LyC']['norm'] = LogNorm(1.e-16, 1.e-10)
    my_dfi["T"]["cmap"] = jh_cmap
    my_dfi["T"]["norm"] = LogNorm(1.0e1, 1.0e7)
    my_dfi["cool_rate"]["norm"] = LogNorm(1.0e-27, 1.0e-20)
    my_dfi["cool_rate"]["cmap"] = cmr.get_sub_cmap(cmr.freeze_r, 0.0, 0.7)
    my_dfi["heat_rate"]["norm"] = LogNorm(1.0e-27, 1.0e-20)
    my_dfi["heat_rate"]["cmap"] = cmr.get_sub_cmap(cmr.flamingo_r, 0.0, 0.7)
    my_dfi["net_cool_rate"] = add_dfi(
        r"$\mathcal{G-L}>0\;[{\rm erg}\,{\rm cm^{-3}}\,{\rm s}^{-1}]$",
        SymLogNorm(1.0e-26, vmin=-1.0e-22, vmax=1.0e-22),
        cmr.get_sub_cmap(cmr.fusion, 0.1, 0.9),
    )
    my_dfi["Uion"] = add_dfi(r"$U_{\rm ion}$", LogNorm(1.0e-6, 1.0e-2), cmr.dusk)
    return my_dfi


def paper_snapshot(
    s,
    num,
    z0=0,
    dz=15,
    dt=10,
    outdir=None,
    scratch_dir="/scratch/gpfs/changgoo/TIGRESS-NCR/",
):
    fields = [
        "nH",
        "T",
        "xHI",
        "xe",
        "xHII",
        "xH2",
        "chi_FUV",
        "cool_rate",
        "net_cool_rate",
        "Erad_LyC",
    ]
    ds = s.load_vtk(num)
    dfi = set_my_dfi(ds.dfi)
    if s.config_time < pd.to_datetime("2022-02-10 13:21:32 -0500"):
        cooling_rate_unit = 1.0
    else:
        cooling_rate_unit = (s.u.energy_density / s.u.time).cgs.value
    try:
        s.load_chunk(num)
        dd = s.get_field_from_chunk(fields)
    except IOError:
        print("Reading fields from vtk:", fields)
        dd = ds.get_field(fields)
    dd["cool_rate"] *= cooling_rate_unit
    dd["net_cool_rate"] *= cooling_rate_unit
    # dd['heat_rate'] *= cooling_rate_unit

    dslc = dd.sel(z=z0, method="nearest")

    if outdir is None:
        outdir = os.path.join(scratch_dir, s.basename, "snapshot_{:03d}".format(z0))
        if not os.path.isdir(outdir):
            os.makedirs(outdir)

    prj = s.read_prj(num)
    sp = s.load_starpar_vtk(num, force_override=True)

    # setup figure and axes
    fig = plt.figure(figsize=(18.9, 10), num=0)

    gs0 = gridspec.GridSpec(
        2,
        1,
        left=0,
        right=0.25,
        bottom=0.00,
        top=1,
        height_ratios=[1, 1],
        wspace=0,
        hspace=0,
    )

    # XZ-projections
    # gs0 = gs[0].subgridspec(2,1)
    # gs0.set_height_ratios([1,15])
    # cax = plt.subplot(gs0[0])
    # ax=plt.subplot(gs0[1])
    axes = dict()
    for i, f in enumerate(["Sigma_gas", "EM"]):
        ax = plt.subplot(gs0[i])
        axes[f] = ax
        plt.sca(ax)
        # data['surf_y'].plot(**dfi['surf'])
        im = plt.imshow(
            prj["z"][f],
            cmap=dfi[f]["cmap"],
            norm=dfi[f]["norm"],
            extent=np.array(
                [
                    ds.domain["le"][0],
                    ds.domain["re"][0],
                    ds.domain["le"][1],
                    ds.domain["re"][1],
                ]
            )
            / 1.0e3,
        )
        # plt.ylim(-2,2)
        # cbar=plt.colorbar(im,cax=cax,orientation='horizontal')
        if i == 0:
            cbar = plt.colorbar(
                im,
                pad=0.02,
                shrink=0.8,
                location="top",
                label=dfi[f]["cbar_kwargs"]["label"],
            )
        else:
            cbar = plt.colorbar(
                im,
                pad=0.02,
                shrink=0.8,
                location="bottom",
                label=dfi[f]["cbar_kwargs"]["label"],
            )
        decorate_ax(dfi[f]["cbar_kwargs"]["label"], ax, iscbar=False)
        ax.set_aspect("equal")
        # ax.axhline(z0/1.e3,ls=':',color='k',lw=2)

        ax.axis("on")
        set_axis_color(ax, "x", "none")
        ax.spines["right"].set_color("none")
        ax.spines["left"].set_position(("axes", -0.02))
        ax.yaxis.tick_left()
        ax.yaxis.set_label_position("left")

        ax.set_ylabel(r"$y\,[{\rm kpc}]$")

    ax = axes["Sigma_gas"]
    scatter_sp(
        sp, ax, "z", kpc=True, norm_factor=50, u=s.u, agemax=40, kind="slc", dist_max=50
    )
    legend_sp(ax, norm_factor=50, mass=[1.0e3, 1.0e4], location="right")
    bbox = ax.get_position()
    # lx=bbox.x1-bbox.x0
    ly = bbox.y1 - bbox.y0
    # dx = lx*0.2
    dy = ly * 0.1
    bbox = [bbox.x1 + 0.01, bbox.y0 + dy * 0.5, 0.01, ly * 0.5 - dy]
    colorbar_sp(fig, 40, bbox=bbox, orientation="vertical")

    # Z-slices
    gs1 = gridspec.GridSpec(
        2, 3, left=0.34, right=1, bottom=0.0, top=1.0, wspace=0.0, hspace=0.0
    )

    #     # z-projections
    #     axes=[]
    #     for i,f in enumerate(['Sigma_gas','EM']):
    #         ax=plt.subplot(gs1[i,0])
    #         axes.append(ax)
    #         plt.sca(ax)
    #         # data['surf_y'].plot(**dfi['surf'])
    #         im=plt.imshow(prj['z'][f],cmap=dfi[f]['cmap'],norm=dfi[f]['norm'],
    #                      extent=np.array([ds.domain['le'][0],ds.domain['re'][0],ds.domain['le'][1],ds.domain['re'][1]])/1.e3)
    #         cbloc = 'top' if (i==0) else 'bottom'
    #         cbar=plt.colorbar(im,pad=0.02,shrink=0.8,location=cbloc,label=dfi[f]['cbar_kwargs']['label'])
    #         decorate_ax(dfi[f]['cbar_kwargs']['label'],ax,iscbar=False)
    #         ax.set_aspect('equal')
    #     ax=axes[0]
    #     scatter_sp(sp,ax,'z',kpc=True,norm_factor=200,u=s.u,agemax=40)

    LyC_slc = dslc["Erad_LyC"]
    # nH_slc = dslc['nH']
    extent = (
        np.array(
            [
                ds.domain["le"][0],
                ds.domain["re"][0],
                ds.domain["le"][1],
                ds.domain["re"][1],
            ]
        )
        / 1.0e3
    )
    fields_to_draw = [
        "nH",
        "T",
        "xe",
        "chi_FUV",
        "cool_rate",
        "net_cool_rate",
    ]
    for i, f in enumerate(fields_to_draw):
        #     slc=data[f].sel(z=0,method='nearest')
        #     slc=data[f].sel(z=slice(z0-ds.domain['dx'][2],z0+ds.domain['dx'][2])).mean(dim='z')
        slc = dslc[f]
        ax = plt.subplot(gs1[i // 3, i % 3])
        axes[f] = ax
        plt.sca(ax)
        #     slc.plot(**dfi[f])
        if f == "net_cool_rate":
            im = plt.imshow(
                slc,
                cmap=dfi["cool_rate"]["cmap"],
                norm=dfi["cool_rate"]["norm"],
                extent=extent,
            )
            im = plt.imshow(
                -slc,
                cmap=dfi["heat_rate"]["cmap"],
                norm=dfi["heat_rate"]["norm"],
                extent=extent,
            )
        else:
            im = plt.imshow(
                slc, cmap=dfi[f]["cmap"], norm=dfi[f]["norm"], extent=extent
            )

        ax.set_aspect("equal")
        ax.set_title("")
        decorate_ax(dfi[f]["cbar_kwargs"]["label"], ax, iscbar=False)
        if i // 3 == 0:
            cbar = plt.colorbar(
                im,
                pad=0.02,
                shrink=0.8,
                location="top",
                label=dfi[f]["cbar_kwargs"]["label"],
            )
        elif i // 3 == 1:
            cbar = plt.colorbar(
                im,
                pad=0.02,
                shrink=0.8,
                location="bottom",
                label=dfi[f]["cbar_kwargs"]["label"],
            )
        if f == "T":
            cbar.set_ticks([100, 1.0e4, 1.0e6])
        if f in ["xHI", "chi_FUV"]:
            # if f != 'Erad_LyC':
            plt.contour(
                slc.x / 1.0e3,
                slc.y / 1.0e3,
                LyC_slc,
                levels=[1.0e-15, 1.0e-14, 1.0e-13, 1.0e-12],
                linewidths=2,
                colors=["tab:red", "tab:orange", "tab:green", "tab:blue"],
            )
        #                     cmap=dfi['Erad_LyC']['cmap'],norm=dfi['Erad_LyC']['norm'])
        if f in ["Erad_LyC"]:
            #         plt.contour(slc.x/1.e3,slc.y/1.e3,nH_slc,levels=[1.,1.e2],cmap=dfi['density']['cmap'])
            scatter_sp(sp, plt.gca(), "z", kpc=True, norm_factor=2, u=s.u)
        if f in ["nH"]:
            # spsel=sp.where(np.abs((sp['x3']-z0)<dz)).dropna()
            # scatter_sp(spsel,ax,'z',kpc=True,norm_factor=20,u=s.u,agemax=40)
            snsel = get_sn(s, ds, dt=dt, dz=dz, z0=z0)
            plt.scatter(
                snsel["x1"] / 1.0e3,
                snsel["x2"] / 1.0e3,
                s=100,
                linewidths=0,
                marker="*",
                color="deeppink",
                # alpha=np.clip((snsel['time']-ds.domain['time']+dt)/(dt),0,1)
            )

    #         plt.plot(snsel[snsel['mass']==0]['x1']/1.e3,snsel[snsel['mass']==0]['x2']/1.e3,'*',color='magenta')
    for f in ["nH", "chi_FUV"]:
        ax = axes[f]
        ax.axis("on")
        set_axis_color(ax, "x", "none")
        ax.spines["right"].set_color("none")
        ax.spines["left"].set_position(("axes", -0.02))
        ax.yaxis.tick_left()
        ax.yaxis.set_label_position("left")
        ax.set_yticks([-0.4, 0.0, 0.4])
    #     ax.set_ylabel(r'$y\,[{\rm kpc}]$')

    for f in ["xe", "net_cool_rate"]:
        ax = axes[f]
        ax.axis("on")
        set_axis_color(ax, "x", "none")
        ax.spines["left"].set_color("none")
        ax.spines["right"].set_position(("axes", 1.02))
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_yticks([-0.4, 0.0, 0.4])
        ax.set_ylabel(r"$y\,[{\rm kpc}]$")

    # plt.savefig(os.path.join(outdir,'{:s}.{:04d}.z{:03d}.snapshot.png'.format(s.basename,ds.num,z0)),
    # dpi=300,bbox_inches='tight')

    return fig
