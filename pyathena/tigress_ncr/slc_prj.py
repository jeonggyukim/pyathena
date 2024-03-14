# slc_prj.py

import os
import os.path as osp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import astropy.units as au
import astropy.constants as ac
from matplotlib.colors import Normalize, LogNorm
from mpl_toolkits.axes_grid1 import ImageGrid
import xarray as xr

from ..load_sim import LoadSim

# from ..io.read_starpar_vtk import read_starpar_vtk
from ..plt_tools.cmap_shift import cmap_shift
from ..plt_tools.plt_starpar import scatter_sp
from ..classic.utils import texteffect

cmap_def = dict(
    Sigma_gas=plt.cm.pink_r,
    Sigma_H2=plt.cm.pink_r,
    EM=plt.cm.plasma,
    nH=plt.cm.Spectral_r,
    T=cmap_shift(mpl.cm.RdYlBu_r, midpoint=3.0 / 7.0),
    vz=plt.cm.bwr,
    vy=plt.cm.bwr,
    vx=plt.cm.bwr,
    chi_FUV=plt.cm.viridis,
    Erad_LyC=plt.cm.viridis,
    xi_CR=plt.cm.viridis,
    Bmag=plt.cm.cividis,
)

norm_def = dict(
    Sigma_gas=LogNorm(1e-2, 1e3),
    Sigma_H2=LogNorm(1e-2, 1e3),
    EM=LogNorm(1e0, 1e5),
    nH=LogNorm(1e-4, 1e3),
    T=LogNorm(1e1, 1e7),
    vz=Normalize(-200, 200),
    vx=Normalize(-200, 200),
    vy=Normalize(-200, 200),
    chi_FUV=LogNorm(1e-2, 1e2),
    Erad_LyC=LogNorm(1e-16, 5e-13),
    xi_CR=LogNorm(5e-17, 1e-15),
    Bmag=LogNorm(1.0e-2, 1.0e2),
)

tiny = 1.0e-30


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

        ds = self.load_vtk(num=num)
        fields = (
            ds.field_list
        )  # list(set(ds.field_list) - {'velocity','cell_centered_B'})
        # fields += ['velocity1','velocity2','velocity3']
        # if self.par["configure"]["gas"] == "mhd":
        #     fields += ["cell_centeredB1", "cell_centered_B2", "cell_centered_B3"]

        res = dict()
        res["time"] = ds.domain["time"]
        res["extent"] = self._get_extent(ds.domain)

        for ax in axes:
            dat = ds.get_slice(ax, fields, pos="c", method="nearest")
            res[ax] = dict()
            for f in fields:
                if f in ["velocity", "cell_centered_B"]:
                    for ivec in ["1", "2", "3"]:
                        res[ax][f + ivec] = dat[f + ivec].data
                else:
                    res[ax][f] = dat[f].data

        for zpos, zlab in zip(
            [-1000, -500, -200, 200, 500, 1000],
            ["zn10", "zn05", "zn02", "zp02", "zp05", "zp10"],
        ):
            dat = ds.get_slice("z", fields, pos=zpos, method="nearest")
            res[zlab] = dict()
            for f in fields:
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
            "specific_scalar[0]",
        ]
        try:
            if self.par["configure"]["radps"] == "ON":
                fields_def += ["nH2", "ne", "nHII"]
                if self.par["cooling"]["iCR_attenuation"]:
                    fields_def += ["xi_CR"]
                if self.par["radps"]["iPhotIon"] == 1:
                    fields_def += ["Erad_LyC"]
                if self.par["cooling"]["iPEheating"] == 1:
                    fields_def += ["chi_FUV"]
        except KeyError:
            pass
        if self.par["configure"]["gas"] == "mhd":
            fields_def += ["Bx", "By", "Bz", "Bmag"]
        if fields == "default":
            fields = fields_def

        newslc = dict()
        for k in keylist:
            if k in ["time", "extent"]:
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
                        newslc[k][f] = self.dfi[f]["func"](data, self.u)
                    else:
                        print("{} is not available".format(f))
        return newslc

    @LoadSim.Decorators.check_pickle
    def read_slc(
        self,
        num,
        axes=["x", "y", "z"],
        fields=None,
        prefix="slc",
        savdir=None,
        force_override=False,
    ):
        fields_def = [
            "nH",
            "vz",
            "T",
            "cs",
            "vx",
            "vy",
            "vz",
            "pok",
            "specific_scalar[0]",
        ]
        try:
            if self.par["configure"]["radps"] == "ON":
                fields_def += ["nH2", "ne", "nHII"]
                if self.par["cooling"]["iCR_attenuation"]:
                    fields_def += ["xi_CR"]
                if self.par["radps"]["iPhotIon"] == 1:
                    fields_def += ["Erad_LyC"]
                if self.par["cooling"]["iPEheating"] == 1:
                    fields_def += ["chi_FUV"]
        except KeyError:
            pass
        if self.par["configure"]["gas"] == "mhd":
            fields_def += ["Bx", "By", "Bz", "Bmag"]

        fields = fields_def
        axes = np.atleast_1d(axes)

        ds = self.load_vtk(num=num)
        res = dict()
        res["time"] = ds.domain["time"]
        res["extent"] = self._get_extent(ds.domain)

        for ax in axes:
            dat = ds.get_slice(ax, fields, pos="c", method="nearest")
            res[ax] = dict()
            for f in fields:
                res[ax][f] = dat[f].data

        for zpos, zlab in zip(
            [-1000, -500, 500, 1000], ["zn10", "zn05", "zp05", "zp10"]
        ):
            dat = ds.get_slice("z", fields, pos=zpos, method="nearest")
            res[zlab] = dict()
            for f in fields:
                res[zlab][f] = dat[f].data

        return res

    @LoadSim.Decorators.check_pickle
    def read_prj(
        self,
        num,
        axes=["x", "y", "z"],
        fields=["density", "xHI", "xH2", "xHII", "xe", "nesq"],
        prefix="prj",
        savdir=None,
        force_override=False,
    ):
        axtoi = dict(x=0, y=1, z=2)
        axes = np.atleast_1d(axes)

        ds = self.load_vtk(num=num)
        dat = ds.get_field(fields, as_xarray=True)
        res = dict()
        res["extent"] = self._get_extent(ds.domain)
        res["time"] = ds.domain["time"]

        for ax in axes:
            i = axtoi[ax]
            dx = ds.domain["dx"][i] * self.u.length
            conv_Sigma = (dx * self.u.muH * ac.u.cgs / au.cm**3).to("Msun/pc**2").value
            conv_EM = (dx * au.cm**-6).to("pc cm-6").value

            res[ax] = dict()
            res[ax]["Sigma_gas"] = (
                np.sum(dat["density"], axis=2 - i) * conv_Sigma
            ).data
            for field in fields:
                if field == "xH2":
                    val = (2.0 * dat["density"] * dat["xH2"]).data
                    valsum = np.sum(val, axis=2 - i) * conv_Sigma
                    res[ax]["Sigma_H2"] = valsum
                elif field in ["xHI", "xHII", "xe"]:
                    val = (dat["density"] * dat[field]).data
                    valsum = np.sum(val, axis=2 - i) * conv_Sigma
                    res[ax][f"Sigma_{field[1:]}"] = valsum
                elif field == "nesq":
                    val = (dat["nesq"]).data
                    valsum = np.sum(val, axis=2 - i) * conv_EM
                    res[ax]["EM"] = valsum
                elif field.startswith("specific_scalar"):
                    ns = field[-2:-1]
                    val = (dat["density"] * dat[f"specific_scalar[{ns}]"]).data
                    valsum = np.sum(val, axis=2 - i) * conv_Sigma
                    res[ax][f"Sigma_scalar{ns}"] = valsum

        return res

    @LoadSim.Decorators.check_pickle
    def read_prj_RMDMEM(
        self,
        num,
        axes=["x", "y", "z"],
        fields=["nH", "ne", "xHI", "cell_centered_B", "T"],
        prefix="prj2",
        savdir=None,
        force_override=False,
    ):
        axtoi = dict(x=0, y=1, z=2)
        axes = np.atleast_1d(axes)

        ds = self.load_vtk(num=num)
        dat = ds.get_field(fields, as_xarray=True)
        res = dict()
        res["extent"] = self._get_extent(ds.domain)
        res["time"] = ds.domain["time"]

        for ax in axes:
            i = axtoi[ax]
            dx = ds.domain["dx"][i]
            Blos = dat[f"cell_centered_B{i+1}"]
            Btot = np.sqrt(
                dat["cell_centered_B1"] ** 2
                + dat["cell_centered_B2"] ** 2
                + dat["cell_centered_B3"] ** 2
            )
            nHI = dat["nH"] * dat["xHI"]
            ne = dat["ne"]

            res[ax] = dict()
            res[ax]["NHI"] = (nHI * dx).sum(dim=ax)
            res[ax]["DM"] = (ne * dx).sum(dim=ax)
            res[ax]["RM"] = (ne * Blos * dx).sum(dim=ax)
            res[ax]["Blos"] = (Blos * dx).sum(dim=ax)
            res[ax]["EM"] = (ne**2 * dx).sum(dim=ax)
            res[ax]["neB"] = (ne * Btot * dx).sum(dim=ax)
            res[ax]["Btot"] = (Btot * dx).sum(dim=ax)

            warm = dat["T"] < 3.5e4
            res[ax]["NHI_w"] = (nHI * dx * warm).sum(dim=ax)
            res[ax]["DM_w"] = (ne * dx * warm).sum(dim=ax)
            res[ax]["RM_w"] = (ne * Blos * dx * warm).sum(dim=ax)
            res[ax]["Blos_w"] = (Blos * dx * warm).sum(dim=ax)
            res[ax]["EM_w"] = (ne**2 * dx * warm).sum(dim=ax)
            res[ax]["neB_w"] = (ne * Btot * dx * warm).sum(dim=ax)
            res[ax]["Btot_w"] = (Btot * dx * warm).sum(dim=ax)
            res[ax]["warm"] = (warm).sum(dim=ax)

        return res

    def read_slc_xarray(self, num, fields="default", axis="zall", force_override=False):
        slc = self.read_slc_from_allslc(
            num, fields=fields, force_override=force_override
        )
        if axis == "zall":
            slc_dset = slc_get_all_z(slc)
        else:
            slc_dset = slc_to_xarray(slc, axis)
        return slc_dset

    def read_slc_time_series(
        self,
        num1=None,
        num2=None,
        nskip=1,
        nums=None,
        fields="default",
        axis="zall",
        sfr=True,
        radiation=False,
    ):
        slc_list = []
        if nums is None:
            nums = range(num1, num2, nskip)
        for num in nums:
            try:
                slc = self.read_slc_xarray(num, fields=fields, axis=axis)
                slc_list.append(slc)
            except OSError:
                continue
        slc_dset = xr.concat(slc_list, dim="time")
        if sfr:
            hst = self.read_hst()
            for fsfr in ["sfr10", "sfr40", "sfr100"]:
                slc_dset[fsfr] = (
                    hst[fsfr]
                    .to_xarray()
                    .rename(time_code="time")
                    .interp(time=slc_dset.time)
                )
        if radiation and self.test_newcool():
            hst = self.read_hst()
            for lum in ["Ltot_PH", "Ltot_PE", "Ltot_LW"]:
                slc_dset[lum] = (
                    hst[lum]
                    .to_xarray()
                    .rename(time_code="time")
                    .interp(time=slc_dset.time)
                )
            slc_dset["Ltot_FUV"] = slc_dset["Ltot_PE"] + slc_dset["Ltot_LW"]
            for fi in ["Sigma_gas", "H_2p", "nmid", "nmid_2p"]:
                slc_dset[fi] = (
                    hst[fi]
                    .to_xarray()
                    .rename(time_code="time")
                    .interp(time=slc_dset.time)
                )

        return slc_dset

    def slc_to_flux(self, slc):
        import astropy.constants as ac

        dset = xr.Dataset()
        dset["density"] = slc["nH"]
        dset["velocity1"] = slc["vx"]
        dset["velocity2"] = slc["vy"]
        dset["velocity3"] = slc["vz"]
        Bunits = np.sqrt(self.u.energy_density.cgs.value) * np.sqrt(4.0 * np.pi) * 1e6
        dset["magnetic_field1"] = slc["Bx"] / Bunits
        dset["magnetic_field2"] = slc["By"] / Bunits
        dset["magnetic_field3"] = slc["Bz"] / Bunits
        pok_units = (self.u.energy_density / ac.k_B).cgs.value
        dset["pressure"] = slc["pok"] / pok_units

        dset["ekin"] = (
            0.5
            * dset["density"]
            * (dset["velocity1"] ** 2 + dset["velocity2"] ** 2 + dset["velocity3"] ** 2)
        )
        dset["eth"] = 1.5 * dset["pressure"]
        dset["emag"] = 0.5 * (
            dset["magnetic_field1"] ** 2
            + dset["magnetic_field2"] ** 2
            + dset["magnetic_field3"] ** 2
        )
        dset["cs"] = np.sqrt(dset["pressure"] / dset["density"])

        zsign = (np.abs(dset.z) / dset.z).fillna(1)
        dset["vout"] = dset["velocity3"] * zsign
        dset["Bout"] = dset["magnetic_field3"] * zsign
        dset["massflux"] = dset["density"] * dset["vout"]
        dset["momflux_mag"] = dset["emag"] - dset["magnetic_field3"] ** 2
        dset["momflux_kin"] = dset["density"] * dset["vout"] ** 2
        dset["momflux_th"] = dset["pressure"]
        dset["momflux"] = dset["momflux_kin"] + dset["momflux_th"] + dset["momflux_mag"]

        dset["energyflux_kin_z"] = 0.5 * dset["density"] * dset["vout"] ** 3
        dset["energyflux_kin"] = dset["vout"] * dset["ekin"]
        dset["energyflux_th"] = dset["vout"] * 2.5 * dset["pressure"]
        dset["energyflux"] = dset["energyflux_kin"] + dset["energyflux_th"]
        dset["poyntingflux"] = 2.0 * dset["vout"] * dset["emag"]
        dset["poyntingflux"] -= dset["Bout"] * (
            dset["velocity1"] * dset["magnetic_field1"]
            + dset["velocity2"] * dset["magnetic_field2"]
            + dset["velocity3"] * dset["magnetic_field3"]
        )
        if "specific_scalar0" in dset:
            dset["Z"] = dset["specific_scalar0"]
            # dset['yZ']=dset['specific_scalar0']/ZISM
            dset["metalflux"] = (
                dset["specific_scalar0"] * dset["density"] * dset["vout"]
            )

        if "sfr" in slc.attrs:
            dset.attrs["sfr"] = slc.attrs["sfr"]

        return dset

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
        fields_xy=("Sigma_gas", "Sigma_H2", "EM", "nH", "T", "chi_FUV"),
        fields_xz=(
            "Sigma_gas",
            "Sigma_H2",
            "EM",
            "nH",
            "T",
            "vz",
            "Bmag",
            "chi_FUV",
            "Erad_LyC",
        ),
        # fields_xy=('Sigma_gas', 'EM', 'xi_CR', 'nH', 'chi_FUV', 'Erad_LyC'),
        # fields_xz=('Sigma_gas', 'EM', 'nH', 'chi_FUV', 'Erad_LyC', 'xi_CR'),
        norm_factor=5.0,
        agemax=20.0,
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
            vz=r"$v_z$",
            vy=r"$v_y$",
            vx=r"$v_x$",
            chi_FUV=r"$\mathcal{E}_{\rm FUV}$",
            Erad_LyC=r"$\mathcal{E}_{\rm LyC}$",
            xi_CR=r"$\xi_{\rm CR}$",
            Bmag=r"$|B|$",
        )

        kind = dict(
            Sigma_gas="prj",
            Sigma_H2="prj",
            EM="prj",
            nH="slc",
            T="slc",
            vz="slc",
            vy="slc",
            vx="slc",
            chi_FUV="slc",
            Erad_LyC="slc",
            xi_CR="slc",
            Bmag="slc",
        )
        nxy = len(fields_xy)
        nxz = len(fields_xz)
        ds = self.load_vtk(num=num)
        LzoLx = ds.domain["Lx"][2] / ds.domain["Lx"][0]
        xwidth = 3
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

        exclude_fields = []
        if self.par["configure"]["radps"] == "ON":
            if self.par["radps"]["iPhotIon"] == 0:
                exclude_fields += ["Erad_LyC"]
            if self.par["cooling"]["iPEheating"] == 0:
                exclude_fields += ["chi_FUV"]

        slc_fields = []
        for f in fields_xy:
            if f in exclude_fields:
                continue
            if kind[f] == "slc":
                slc_fields += [f]
        for f in fields_xz:
            if f in exclude_fields:
                continue
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
        sp = self.load_starpar_vtk(num)

        extent = dat["prj"]["extent"]["z"]

        # spiral arm model -- rolling y position
        if self.test_spiralarm():
            Om = self.par["problem"]["Omega"]
            pattern = self.par["problem"]["pattern"]
            vy0 = self.par["problem"]["R0"] * (1 - pattern) * Om
            Ly = ds.domain["Lx"][1]
            dy = ds.domain["dx"][1]
            ymin = ds.domain["le"][1]
            yshift = np.mod(vy0 * ds.domain["time"], Ly)
            jshift = yshift / dy
            ynew = sp["x2"].copy()
            ynew -= ymin + yshift
            negy = ynew.loc[ynew < 0].copy()
            ynew.loc[ynew < 0] = negy + Ly
            sp["x2"] = ynew + ymin
            self.logger.info(
                "[plt_snapshot] y-position will be rolled "
                + "with vy0={0} and yshift={1}".format(vy0, yshift)
            )
        else:
            jshift = 0

        for i, (ax, f) in enumerate(zip(g1, fields_xy)):
            ax.set_aspect(ds.domain["Lx"][1] / ds.domain["Lx"][0])
            self.plt_slice(
                ax,
                dat[kind[f]],
                "z",
                f,
                cmap=cmap_def[f],
                norm=norm_def[f],
                jshift=jshift,
            )

            if i == 0:
                scatter_sp(
                    sp,
                    ax,
                    "z",
                    kind="prj",
                    kpc=False,
                    norm_factor=norm_factor,
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
            ax.set_aspect(ds.domain["Lx"][2] / ds.domain["Lx"][0])
            self.plt_slice(ax, dat[kind[f]], "y", f, cmap=cmap_def[f], norm=norm_def[f])
            if i == 0:
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
            "Model: {0:s}  time=".format(suptitle) + str(int(ds.domain["time"])),
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
                os.makedirs(savdir)

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
    return dset.assign_coords(time=slc["time"])


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
