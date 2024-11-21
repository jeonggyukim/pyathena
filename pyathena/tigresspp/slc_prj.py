# slc_prj.py

import os
import os.path as osp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# import astropy.units as au
# import astropy.constants as ac
from matplotlib.colors import Normalize, LogNorm, SymLogNorm
from mpl_toolkits.axes_grid1 import ImageGrid
import xarray as xr

from ..load_sim import LoadSim
from ..plt_tools.cmap_shift import cmap_shift
from ..plt_tools.plt_starpar import scatter_sp
from ..classic.utils import texteffect

cmap_def = dict(
    Sigma_gas=plt.cm.pink_r,
    Sigma_H2=plt.cm.pink_r,
    EM=plt.cm.plasma,
    nH=plt.cm.Spectral_r,
    T=cmap_shift(mpl.cm.RdYlBu_r, midpoint=3.0 / 7.0),
    pok=plt.cm.plasma,
    vz=plt.cm.bwr,
    vy=plt.cm.bwr,
    vx=plt.cm.bwr,
    chi_FUV=plt.cm.viridis,
    Erad_LyC=plt.cm.viridis,
    xi_CR=plt.cm.viridis,
    Bmag=plt.cm.cividis,
    rmetal=plt.cm.cool,
    rSN=plt.cm.cool,
)

norm_def = dict(
    Sigma_gas=LogNorm(1e-2, 1e2),
    Sigma_H2=LogNorm(1e-2, 1e2),
    EM=LogNorm(1e0, 1e5),
    nH=LogNorm(1e-4, 1e3),
    T=LogNorm(1e1, 1e8),
    pok=LogNorm(1.0e2, 1.0e6),
    vz=SymLogNorm(1, vmin=-1000, vmax=1000),
    vy=SymLogNorm(1, vmin=-1000, vmax=1000),
    vx=SymLogNorm(1, vmin=-1000, vmax=1000),
    chi_FUV=LogNorm(1e-2, 1e2),
    Erad_LyC=LogNorm(1e-16, 5e-13),
    xi_CR=LogNorm(5e-17, 1e-15),
    Bmag=LogNorm(1.0e-2, 1.0e2),
    rmetal=LogNorm(0.01,0.2),
    rSN=Normalize(0,1)
)

tiny = 1.0e-30

cpp_to_cc = {
    "rho": "density",
    "press": "pressure",
    "vel1": "velocity1",
    "vel2": "velocity2",
    "vel3": "velocity3",
}

cpp_to_cc_mag = {
    "Bcc1": "cell_centered_B1",
    "Bcc2": "cell_centered_B2",
    "Bcc3": "cell_centered_B3",
}


class SliceProj:
    @staticmethod
    def _get_extent(domain):
        r = dict()
        r["x"] = (domain["le"][1], domain["re"][1], domain["le"][2], domain["re"][2])
        r["y"] = (domain["le"][0], domain["re"][0], domain["le"][2], domain["re"][2])
        r["z"] = (domain["le"][0], domain["re"][0], domain["le"][1], domain["re"][1])

        return r

    @LoadSim.Decorators.check_pickle
    def read_allslc(
        self,
        num,
        axes=["x", "y", "z"],
        prefix="allslc",
        savdir=None,
        force_override=False,
    ):
        axes = np.atleast_1d(axes)

        ds = self.load_hdf5(num=num).rename(cpp_to_cc)
        if "Bcc1" in ds:
            ds = ds.rename(cpp_to_cc_mag)

        if "cell_centered_B1" in ds:
            self.mhd = True
        else:
            self.mhd = False

        res = dict()
        res["time"] = ds.attrs["Time"]
        res["extent"] = self._get_extent(self.domain)
        res["center"] = dict(
            x=self.domain["center"][0],
            y=self.domain["center"][1],
            z=self.domain["center"][2],
        )
        for ax in axes:
            dat = ds.sel({ax: res["center"][ax]}, method="nearest")
            res[ax] = dict()
            for f in dat:
                if f in ["velocity", "cell_centered_B"]:
                    for ivec in ["1", "2", "3"]:
                        res[ax][f + ivec] = dat[f + ivec].data
                else:
                    res[ax][f] = dat[f].data

        for zpos, zlab in zip(
            [-1000, -500, -200, 200, 500, 1000],
            ["zn10", "zn05", "zn02", "zp02", "zp05", "zp10"],
        ):
            dat = ds.sel(z=zpos, method="nearest")
            res[zlab] = dict()
            for f in dat:
                if f in ["velocity", "cell_centered_B"]:
                    for ivec in ["1", "2", "3"]:
                        res[zlab][f + ivec] = dat[f + ivec].data
                else:
                    res[zlab][f] = dat[f].data

        return res

    def read_slc_from_allslc(
        self, num, fields="default", savdir=None, force_override=False
    ):
        allslc = self.read_allslc(num, savdir=savdir, force_override=force_override)
        keylist = list(allslc.keys())

        if "cell_centered_B1" in allslc:
            self.mhd = True
        else:
            self.mhd = False

        if fields is None:
            return allslc

        fields_def = [
            "nH",
            "vz",
            "T",
            "cs",
            "vx",
            "vy",
            "vz",
            "pok",
        ]
        if self.mhd:
            fields_def += ["Bx", "By", "Bz", "Bmag"]

        if fields == "default":
            fields = fields_def

        newslc = dict()
        for k in keylist:
            if k in ["time", "extent", "center"]:
                newslc[k] = allslc[k]
            else:
                if k not in newslc:
                    newslc[k] = dict()
                fieldlist = list(allslc[k].keys())
                data = allslc[k]
                for f in fields:
                    if f in fieldlist:
                        newslc[k][f] = data[f]
                    elif f in self.dfi:
                        try:
                            newslc[k][f] = self.dfi[f]["func"](data, self.u)
                        except KeyError:
                            continue
                    else:
                        print("{} is not available".format(f))
        return newslc

    @LoadSim.Decorators.check_pickle
    def read_prj(
        self, num, axes=["x", "y", "z"], prefix="prj", savdir=None, force_override=False
    ):
        axtoi = dict(x=0, y=1, z=2)
        fields = ["rho"]
        axes = np.atleast_1d(axes)

        dat = self.load_hdf5(num=num, quantities=fields)

        res = dict()
        res["extent"] = self._get_extent(self.domain)

        for ax in axes:
            i = axtoi[ax]
            dx = self.domain["dx"][i] * self.u.length
            conv_Sigma = (dx * self.u.density).to("Msun/pc**2")

            res[ax] = dict()
            res[ax]["Sigma_gas"] = (np.sum(dat["rho"], axis=2 - i) * conv_Sigma).data

        return res

    @staticmethod
    def plt_slice(ax, slc, axis="z", field="density", cmap=None, norm=None, jshift=0):
        try:
            if cmap is None:
                cmap = cmap_def[field]

            if norm is None:
                norm = mpl.colors.LogNorm()
            elif norm == "linear":
                norm = mpl.colors.Normalize()

            data = slc[axis][field]
            if (jshift != 0) and (axis == "z"):
                import scipy.ndimage as sciim

                data = sciim.interpolation.shift(data, (-jshift, 0), mode="wrap")

            ax.imshow(
                data + tiny,
                cmap=cmap,
                extent=slc["extent"][axis],
                norm=norm,
                origin="lower",
                interpolation="none",
            )
        except KeyError:
            pass

    @staticmethod
    def plt_proj(
        ax,
        prj,
        axis="z",
        field="Sigma_gas",
        jshift=0,
        cmap=None,
        norm=None,
        vmin=None,
        vmax=None,
    ):
        try:
            vminmax = dict(Sigma_gas=(1e-2, 1e2))
            cmap_def = dict(Sigma_gas="pink_r")

            if cmap is None:
                try:
                    cmap = cmap_def[field]
                except KeyError:
                    cmap = plt.cm.viridis
            if vmin is None or vmax is None:
                vmin = vminmax[field][0]
                vmax = vminmax[field][1]

            if (norm is None) or (norm == "log"):
                norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
            elif norm == "linear":
                norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

            data = prj[axis][field]
            if (jshift != 0) and (axis == "z"):
                import scipy.ndimage as sciim

                data = sciim.interpolation.shift(data, (-jshift, 0), mode="wrap")
            ax.imshow(
                data,
                cmap=cmap,
                extent=prj["extent"][axis],
                norm=norm,
                origin="lower",
                interpolation="none",
            )
        except KeyError:
            pass

    def plt_snapshot(
        self,
        num,
        fields_xy=["Sigma_gas", "T", "pok", "nH", "Bmag", "rSN"],
        fields_xz=[
            "Sigma_gas",
            "nH",
            "T",
            "vz",
            "Bmag",
            "rmetal"
        ],
        xwidth=2,
        norm_factor=5.0,
        agemax=40.0,
        agemax_sn=40.0,
        runaway=False,
        suptitle=None,
        savdir_pkl=None,
        savdir=None,
        force_override=False,
        savefig=True,
    ):
        """Plot 12-panel projection, slice plots in the z and y directions

        Parameters
        ----------
        num : int
            vtk snapshot number
        fields_xy: list of str
            Field names for z projections and slices
        fields_xz: list of str
            Field names for y projections and slices
        norm_factor : float
            Normalization factor for starpar size. Smaller norm_factor for bigger size.
        agemax : float
            Maximum age of radiation source particles [Myr]
        agemax_sn : float
            Maximum age of sn particles [Myr]
        runaway : bool
            If True, show runaway star particles
        suptitle : str
            Suptitle for snapshot
        savdir_pkl : str
            Path to which save (from which load) projections and slices
        savdir : str
            Path to which save (from which load) projections and slices
        """

        label = dict(
            Sigma_gas=r"$\Sigma$",
            Sigma_H2=r"$\Sigma_{\rm H_2}$",
            EM=r"${\rm EM}$",
            nH=r"$n_{\rm H}$",
            T=r"$T$",
            pok=r"$P/k_B$",
            vz=r"$v_z$",
            vy=r"$v_y$",
            vx=r"$v_x$",
            chi_FUV=r"$\mathcal{E}_{\rm FUV}$",
            Erad_LyC=r"$\mathcal{E}_{\rm LyC}$",
            xi_CR=r"$\xi_{\rm CR}$",
            Bmag=r"$|B|$",
            rmetal=r"$Z$",
            rSN=r"$f_{\rm SN}$"
        )

        kind = dict(
            Sigma_gas="prj",
            Sigma_H2="prj",
            EM="prj",
            nH="slc",
            T="slc",
            pok="slc",
            vz="slc",
            vy="slc",
            vx="slc",
            chi_FUV="slc",
            Erad_LyC="slc",
            xi_CR="slc",
            Bmag="slc",
            rmetal="slc",
            rSN="slc"
        )
        nxy = len(fields_xy)
        nxz = len(fields_xz)
        LzoLx = self.domain["Lx"][2] / self.domain["Lx"][0]
        ysize = LzoLx * xwidth
        xsize = ysize / nxy * 4 + nxz * xwidth
        x1 = 0.90 * (ysize * 4 / nxy / xsize)
        x2 = 0.90 * (nxz * xwidth / xsize)

        fig = plt.figure(figsize=(xsize, ysize))  # , constrained_layout=True)
        g1 = ImageGrid(
            fig,
            [0.02, 0.05, x1, 0.94],
            (nxy // 2, 2),
            axes_pad=0.1,
            aspect=True,
            share_all=True,
            direction="column",
        )
        g2 = ImageGrid(
            fig,
            [x1 + 0.07, 0.05, x2, 0.94],
            (1, nxz),
            axes_pad=0.1,
            aspect=True,
            share_all=True,
        )

        slc_fields = []
        for f in fields_xy:
            if kind[f] == "slc":
                slc_fields += [f]
        for f in fields_xz:
            if kind[f] == "slc":
                slc_fields += [f]
        slc_fields = list(set(slc_fields))
        dat = dict()
        dat["slc"] = self.read_slc_from_allslc(
            num, fields=slc_fields, savdir=savdir_pkl, force_override=force_override
        )
        dat["prj"] = self.read_prj(
            num, savdir=savdir_pkl, force_override=force_override
        )
        time = dat["slc"]["time"] * self.u.Myr
        if len(self.files["parbin"]) != 0:
            sp = self.load_parbin(num)
        else:
            sp = None

        extent = dat["prj"]["extent"]["z"]

        # spiral arm model -- rolling y position
        # if self.test_spiralarm():
        #     Om = self.par["orbital_advection"]["Omega"]
        #     pattern = self.par["problem"]["pattern"]
        #     vy0 = self.par["problem"]["R0"] * (1 - pattern) * Om
        #     Ly = ds.domain["Lx"][1]
        #     dy = ds.domain["dx"][1]
        #     ymin = ds.domain["le"][1]
        #     yshift = np.mod(vy0 * ds.domain["time"], Ly)
        #     jshift = yshift / dy
        #     ynew = sp["x2"].copy()
        #     ynew -= ymin + yshift
        #     negy = ynew.loc[ynew < 0].copy()
        #     ynew.loc[ynew < 0] = negy + Ly
        #     sp["x2"] = ynew + ymin
        #     self.logger.info(
        #         "[plt_snapshot] y-position will be rolled "
        #         + "with vy0={0} and yshift={1}".format(vy0, yshift)
        #     )
        # else:
        jshift = 0

        for i, (ax, f) in enumerate(zip(g1, fields_xy)):
            ax.set_aspect(self.domain["Lx"][1] / self.domain["Lx"][0])
            self.plt_slice(
                ax,
                dat[kind[f]],
                "z",
                f,
                cmap=cmap_def[f],
                norm=norm_def[f],
                jshift=jshift,
            )

            if (i == 0) & (sp is not None):
                scatter_sp(
                    sp,
                    ax,
                    "z",
                    kind="prj",
                    kpc=False,
                    norm_factor=norm_factor / 2,
                    agemax=agemax,
                    agemax_sn=agemax_sn,
                    runaway=runaway,
                    cmap=plt.cm.cool_r,
                )
            ax.set(xlim=(extent[0], extent[1]), ylim=(extent[2], extent[3]))
            ax.text(
                0.5,
                0.92,
                label[f],
                **texteffect(fontsize="x-large"),
                ha="center",
                transform=ax.transAxes,
            )
            if i == (nxy // 2 - 1):
                ax.set(xlabel="x [pc]", ylabel="y [pc]")
            else:
                ax.axes.get_xaxis().set_visible(False)
                ax.axes.get_yaxis().set_visible(False)

        extent = dat["prj"]["extent"]["y"]
        for i, (ax, f) in enumerate(zip(g2, fields_xz)):
            ax.set_aspect(self.domain["Lx"][2] / self.domain["Lx"][0])
            self.plt_slice(ax, dat[kind[f]], "y", f, cmap=cmap_def[f], norm=norm_def[f])
            if (i == 0) & (sp is not None):
                scatter_sp(
                    sp,
                    ax,
                    "y",
                    kind="prj",
                    kpc=False,
                    norm_factor=norm_factor,
                    agemax=agemax,
                    cmap=plt.cm.cool_r,
                )
            ax.set(xlim=(extent[0], extent[1]), ylim=(extent[2], extent[3]))
            ax.text(
                0.5,
                0.97,
                label[f],
                **texteffect(fontsize="x-large"),
                ha="center",
                transform=ax.transAxes,
            )
            if i == 0:
                ax.set(xlabel="x [pc]", ylabel="z [pc]")
            else:
                ax.axes.get_xaxis().set_visible(False)
                ax.axes.get_yaxis().set_visible(False)

        if suptitle is None:
            suptitle = self.basename
        # fig.suptitle(suptitle + ' t=' + str(int(ds.domain['time'])), x=0.4, y=1.02,
        #              va='center', ha='center', **texteffect(fontsize='xx-large'))
        fig.suptitle(
            f"Model: {suptitle}  time={time:8.1f} Myr",
            x=0.4,
            y=1.02,
            va="center",
            ha="center",
            **texteffect(fontsize="xx-large"),
        )
        # plt.subplots_adjust(top=0.95)

        if savefig:
            if savdir is None:
                savdir = osp.join(self.savdir, "snapshot")
            if not osp.exists(savdir):
                os.makedirs(savdir, exist_ok=True)

            savname = osp.join(savdir, "{0:s}_{1:04d}.png".format(self.basename, num))
            plt.savefig(savname, dpi=200, bbox_inches="tight")

        return fig


def slc_to_xarray(slc, axis="z"):
    dset = xr.Dataset()
    for f in slc[axis].keys():
        x0, x1, y0, y1 = slc["extent"][axis[0]]

        Ny, Nx = slc[axis][f].shape

        xfc = np.linspace(x0, x1, Nx + 1)
        yfc = np.linspace(y0, y1, Ny + 1)
        xcc = 0.5 * (xfc[1:] + xfc[:-1])
        ycc = 0.5 * (yfc[1:] + yfc[:-1])

        dims = dict(z=["y", "x"], x=["z", "y"], y=["z", "x"])

        dset[f] = xr.DataArray(slc[axis][f], coords=[ycc, xcc], dims=dims[axis[0]])
    return dset


def slc_get_all_z(slc):
    dlist = []
    for k in slc.keys():
        if k.startswith("z"):
            slc_dset = slc_to_xarray(slc, k)
            if len(k) == 1:
                z0 = 0.0
            elif k[1] == "n":
                z0 = float(k[2:]) * (-100)
            elif k[1] == "p":
                z0 = float(k[2:]) * (100)
            else:
                raise KeyError
            slc_dset = slc_dset.assign_coords(z=z0)
            dlist.append(slc_dset)
        else:
            pass
    return xr.concat(dlist, dim="z").sortby("z")
