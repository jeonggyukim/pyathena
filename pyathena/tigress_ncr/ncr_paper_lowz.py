# from .load_sim_tigress_ncr import LoadSimTIGRESSNCR,LoadSimTIGRESSNCRAll
import pyathena as pa
import numpy as np

# import pandas as pd
import xarray as xr
import astropy.constants as ac
import astropy.units as au

# import scipy
import os
import cmasher as cmr
import matplotlib.pyplot as plt

from .phase import assign_phase
from .ncr_papers import PaperData
# from ..plt_tools.utils import texteffect

from scipy.ndimage import shift


class LowZData(PaperData):
    def __init__(self, basedir="/scratch/gpfs/changgoo/TIGRESS-NCR/"):
        self.outdir = "/tigress/changgoo/public_html/TIGRESS-NCR/lowZ-figures/"
        os.makedirs(self.outdir, exist_ok=True)

        models, mlist_early = self._get_models(basedir, verbose=False)
        self.basedir = basedir
        self.sa = pa.LoadSimTIGRESSNCRAll(models)
        self.update_model_list(basedir, verbose=False)

    def update_model_list(self, basedir, verbose=True):
        models, mlist_early = self._get_models(basedir, verbose=verbose)
        self.mlist_all = list(models)
        self.mlist = list(mlist_early)
        self.mlist_early = mlist_early
        self._set_model_params()
        self._set_model_list()
        self._set_colors()
        self._set_torb()

    def _set_model_list(self):
        mgroup = dict()
        for m in self.mlist:
            if "LGR2" in m:
                head = "LGR2"
            elif "LGR4" in m:
                head = "LGR4"
            elif "LGR8" in m:
                head = "LGR8"
            elif "R8" in m:
                head = "R8"

            if "S30" in m:
                head += "-S30"
            elif "S150" in m:
                head += "-S150"
                if "Om01" in m:
                    head += "-Om100q0"
                elif "Om02" in m:
                    head += "-Om200"
                # if 'rstZ01' in m:
                #     head += 'r'
            elif "S100" in m:
                head += "-S100"
                # if 'rstZ01' in m:
                #     head += 'r'
            elif "S05" in m:
                head += "-S05"
            else:
                # additional distinction only for R8 and LGR4
                if ".b1." in m:
                    head += "-b1"
                elif ".b10." in m:
                    head += "-b10"

            print(head, m)
            if head in mgroup:
                mgroup[head].append(m)
            else:
                mgroup[head] = [m]
        try:
            mgroup["R8"] = mgroup["R8-b1"]
            mgroup["LGR4"] = mgroup["LGR4-b1"]
            mgroup["LGR2-S150"] = (
                mgroup["LGR2-S150-Om100q0"] + mgroup["LGR2-S150-Om200"]
            )
        except KeyError:
            print("not all model groups are set")

        self.mgroup = mgroup

    def _set_torb(self):
        self.torb = dict()
        self.torb_Myr = dict()
        self.torb_code = dict()
        for m in self.mlist:
            s = self.sa.set_model(m)
            torb = 2 * np.pi / s.par["problem"]["Omega"]
            s.torb_code = torb
            s.torb_Myr = torb * s.u.Myr
            self.torb[m] = torb
            if "S30" in m:
                gkey = "S30"
            elif "S05" in m:
                gkey = "S05"
            elif "S100" in m:
                gkey = "S100"
            elif "S150" in m:
                gkey = "S150"
                if "Om01" in m:
                    gkey += "-Om100q0"
                elif "Om02" in m:
                    gkey += "-Om200"
            elif m.startswith("R8"):
                gkey = "R8"
            elif m.startswith("LGR4"):
                gkey = "LGR4"
            self.torb_code[gkey] = torb
            self.torb_Myr[gkey] = torb * s.u.Myr

    @staticmethod
    def stitch_hsts(sa, m1, m2):
        s1 = sa.set_model(m1)
        h1 = s1.read_hst()
        s2 = sa.set_model(m2)
        h2 = s2.read_hst()
        t0 = h2.index[0]
        hnew = h1[:t0].append(h2).fillna(0.0)
        return hnew

    @staticmethod
    def get_model_name(s, beta=True, zonly=False):
        p = s.par["problem"]
        if "S30" in s.basename:
            head = "S30"
        elif "S150" in s.basename:
            head = "S150"
            if "Om01" in s.basename:
                head += "-Om100q0"
            elif "Om02" in s.basename:
                head += "-Om200"
        elif "S100" in s.basename:
            head = "S100"
        elif "S05" in s.basename:
            head = "S05"
        else:
            head = s.basename.split("_")[0]
            if beta:
                head += f"-b{int(p['beta'])}"
        if p["Z_gas"] == p["Z_dust"]:
            ztail = f"Z{p['Z_gas']:3.1f}"
        else:
            ztail = f"Zg{p['Z_gas']:3.1f}Zd{p['Z_dust']:5.3f}"
        if zonly:
            return ztail
        else:
            if "rstZ01" in s.basename:
                ztail += "r"
            return "{}-{}".format(head, ztail)

    def _set_model_params(self):
        sa = self.sa
        for m in sa.models:
            s = sa.set_model(m)
            pp = s.par["problem"]
            prp = s.par["radps"]
            s.Zgas = pp["Z_gas"]
            s.Zdust = pp["Z_dust"]
            s.xymax = prp["xymaxPP"]
            s.epspp = prp["eps_extinct"]
            s.beta = pp["beta"]
            s.name = self.get_model_name(s)

    @staticmethod
    def _get_models(basedir, verbose=True):
        dirs = sorted(os.listdir(basedir))[::-1]
        models = dict()
        for d in dirs:
            # if not os.path.isdir(d): continue
            if ("v3.iCR4" in d) or ("v3.iCR5" in d):
                if verbose:
                    print(d)
                models[d] = os.path.join(basedir, d)
        models["R8_8pc_NCR.full.b10.v3"] = os.path.join(
            basedir, "R8_8pc_NCR.full.b10.v3"
        )

        mskip = [
            "LGR4_4pc_NCR_S100.full.b1.v3.iCR4.Zg0.1.Zd0.1.xy1024.eps1.e-8",
            #  'LGR4_4pc_NCR_S100.full.b1.v3.iCR4.Zg1.Zd1.xy1024.eps1.e-8',
            "LGR2_4pc_NCR_S150.full.b2.Om02.v3.iCR4.Zg1.Zd1.xy1024.eps1.e-8",
            "LGR2_4pc_NCR_S150.full.b2.Om02.v3.iCR4.Zg0.1.Zd0.1.xy1024.eps1.e-8",
            "LGR2_4pc_NCR_S150.full.b2.Om02.v3.iCR4.Zg0.1.Zd0.1.xy2048.eps1.e-8",
            "LGR8_8pc_NCR_S05.full.b10.v3.iCR5.Zg1.Zd1.xy8192.eps0.0",
            "LGR8_8pc_NCR_S05.full.b10.v3.iCR4.Zg0.1.Zd0.1.xy4096.eps0.0",
            #  'LGR2_4pc_NCR_S150.full.b2.Om01.q0.v3.iCR5.Zg0.1.Zd0.1.xy2048.eps1.e-8'
        ]
        mlist = list(models)
        mlist_early = dict()
        for m in mlist:
            if m in mskip:
                continue
            if "xy" in m:
                mearly = m[: m.rfind("xy") - 1]
                for m1 in mlist:
                    if m1 == mearly:
                        mlist_early[m] = m1
                    elif m1.replace("iCR4", "iCR5") == mearly:
                        mlist_early[m] = m1
        mlist_early[
            "LGR4_4pc_NCR_S100.full.b1.v3.iCR5.Zg1.Zd1.xy1024.eps1.e-8.rstZ01"
        ] = None
        #'LGR4_4pc_NCR_S100.full.b1.v3.iCR4.Zg0.1.Zd0.1'
        # mlist_early['LGR2_4pc_NCR_S150.full.b2.Om02.v3.iCR5.Zg1.Zd1.xy1024.eps1.e-8.rstZ01'] = None
        #'LGR2_4pc_NCR_S150.full.b2.Om02.v3.iCR4.Zg0.1.Zd0.1'
        mlist_early[
            "R8_8pc_NCR.full.b10.v3.iCR4.Zg1.Zd1.xy2048.eps0.0"
        ] = "R8_8pc_NCR.full.b10.v3"
        mlist_early[
            "LGR2_4pc_NCR_S150.full.b2.Om01.q0.v3.iCR5.Zg1.Zd1.xy1024.eps1.e-8"
        ] = None
        mlist_early[
            "LGR2_4pc_NCR_S150.full.b2.Om02.v3.iCR5.Zg1.Zd1.xy1024.eps1.e-8.rstZ01"
        ] = None
        mlist_early[
            "R8_8pc_NCR.full.b1.v3.iCR4.Zg3.Zd3.xy1024.eps1.e-8"
        ] = "R8_8pc_NCR.full.b1.v3.iCR4.Zg1.Zd1"
        mlist_early[
            "LGR4_4pc_NCR.full.b1.v3.iCR4.Zg3.Zd3.xy512.eps1.e-7"
        ] = "LGR4_4pc_NCR.full.b1.v3.iCR4.Zg1.Zd1.xy1024.eps1.e-8"
        mlist_early[
            "LGR2_4pc_NCR_S150.full.b2.Om02.v3.iCR5.Zg1.Zd1.xy1024.eps1.e-8.rstZ01"
        ] = None
        # 'LGR2_4pc_NCR_S150.full.b2.Om01.q0.v3.iCR4.Zg0.1.Zd0.1'
        # mlist_early['LGR2_4pc_NCR_S150.full.b2.Om01.q0.v3.iCR5.Zg1.Zd1.xy1024.eps1.e-8'] = 'LGR2_4pc_NCR_S150.full.b2.Om01.q0.v3.iCR4.Zg0.1.Zd0.1'
        return models, mlist_early

    def _set_colors(self):
        colors1 = cmr.take_cmap_colors("cmr.guppy", 5, cmap_range=(0.55, 1.0))
        colors2 = cmr.take_cmap_colors("cmr.guppy_r", 5, cmap_range=(0.55, 1.0))
        self.plt_kwargs = dict()
        for gr in self.mgroup:
            self.plt_kwargs[gr] = dict()
            if "S05" in gr:
                colors = colors1
            elif "S30" in gr:
                colors = colors1
            elif "S100" in gr:
                colors = colors2
            elif "S150" in gr:
                colors = colors2
            elif "R8" in gr:
                colors = colors1
            elif "LGR4" in gr:
                colors = colors2

            if "b10" in gr:
                ls = "--"
            else:
                ls = "-"
            self.plt_kwargs[gr]["colors"] = colors
            self.plt_kwargs[gr]["ls"] = ls

            for m in self.mgroup[gr]:
                s = self.sa.set_model(m)
                if s.Zdust == 3.0:
                    s.color = colors[0]
                elif s.Zdust == 1.0:
                    s.color = colors[1]
                elif s.Zdust == 0.3:
                    s.color = colors[2]
                elif s.Zdust == 0.1:
                    s.color = colors[3]
                elif s.Zdust == 0.025:
                    s.color = colors[4]
                else:
                    print("{} cannot find matching color".format(m))
                if "rstZ01" in m:
                    s.ls = ":"
                elif "Om01" in m:
                    s.ls = "--"
                else:
                    s.ls = ls

    def read_hst(self, mgroup=None):
        if mgroup is None:
            mlist = self.mlist
        elif mgroup in self.mgroup:
            mlist = self.mgroup[mgroup]
        else:
            print("list of mgroup:", list(self.mgroup.keys()))
        for m in mlist:
            s = self.sa.set_model(m)
            mearly = self.mlist_early[m]
            try:
                print("reading history:", m)
                if mearly is None:
                    h = s.read_hst()
                else:
                    h = self.stitch_hsts(self.sa, mearly, m)
            except (KeyError, OSError, IOError):
                print("history reading error: ", mearly, m)
            s.h = h
            s.h["tdep40"] = s.h["Sigma_gas"] / s.h["sfr40"] * 1.0e-3

    def plot_hst(self, m, y="sfr40", full=True):
        s = self.sa.set_model(m)
        if full:
            h = s.h
        else:
            h = s.hst
        name = self.get_model_name(s)
        # torb = 2*np.pi/s.par['problem']['Omega']*s.u.Myr
        plt.plot(
            h["time"],
            get_smoothed(h[y], h["time"], 5),
            label=name,
            lw=1,
            color=s.color,
            ls=s.ls,
        )

    def collect_hst_list(self, y, tslice=None, group="R8-b10"):
        hlist = []
        namelist = []
        Zlist = []
        for m in self.mgroup[group]:
            s = self.sa.set_model(m)
            namelist.append(s.name)
            if tslice is not None:
                ysel = s.h[y][tslice]
            else:
                ysel = s.h[y]
            hlist.append(ysel)
            Zlist.append(s.Zdust)
        return namelist, hlist, Zlist

    def get_trange(self, s):
        if s.torb_Myr < 50:
            trange = slice(s.torb_Myr * 5, s.torb_Myr * 15)
        elif s.torb_Myr > 300:
            trange = slice(s.torb_Myr * 1.5, s.torb_Myr * 5)
        else:
            trange = slice(s.torb_Myr * 2, s.torb_Myr * 5)

        return trange

    def collect_zpdata(
        self,
        m,
        trange=None,
        reduce=True,
        recal=False,
        silent=False,
        func=np.mean,
        **func_kwargs,
    ):
        zpmid, zpwmid = self.get_PW_time_series(m, recal=recal, silent=silent)
        # if m in self.mlist_early:
        #     zpmid_early,zpwmid_early = self.get_PW_time_series(self.mlist_early[m],recal=recal)
        #     tmax = zpmid.time.min().data*0.999
        #     tmin = zpmid_early.time.min().data
        #     zpmid=xr.concat([zpmid_early.sel(time=slice(tmin,tmax)),zpmid],dim='time')
        #     zpwmid=xr.concat([zpwmid_early.sel(time=slice(tmin,tmax)),zpwmid],dim='time')
        s = self.sa.set_model(m)
        sfr_field = "sfr40"
        if "LGR8" in m:
            sfr_field = "sfr100"
        if trange is None:
            self.get_trange(s)
            if not silent:
                print(
                    m, s.torb_Myr, trange, zpmid.time.data.min(), zpmid.time.data.max()
                )

        zpmid = zpmid.sel(time=trange)
        zpwmid = zpwmid.sel(time=trange)

        ydata = xr.Dataset()

        yield_conv = (
            ((au.cm ** (-3) * au.K * ac.k_B) / (ac.M_sun / ac.kpc**2 / au.yr))
            .to("km/s")
            .value
        )
        for ph in ["2p", "hot"]:
            # midplane
            A = zpmid["A"].sel(phase=ph)
            Atop = zpmid["A_top"].sel(phase=ph)
            for yf in ["Ptot", "Pturb", "Pth", "Pimag", "oPimag", "dPimag", "Prad"]:
                if yf not in zpmid:
                    continue
                yfname = yf if ph == "2p" else "{}_{}".format(yf, ph)
                # midplane pressure
                y = zpmid[yf].sel(phase=ph) / A * s.u.pok
                ydata[yfname] = y

                # feedback yield
                ydata[yfname.replace("Pi", "Y").replace("P", "Y")] = (
                    y / zpmid[sfr_field] * yield_conv
                )

                # top pressure
                ytop = zpmid["{}_top".format(yf)].sel(phase=ph) / Atop * s.u.pok
                ydata["{}_top".format(yfname)] = ytop

                # difference
                ydata["d{}".format(yfname)] = y - ytop
        ydata["Ynonth"] = ydata["Yturb"] + ydata["Ymag"]
        A = zpmid["A"].sel(phase="2p")
        for yf in ["nH"]:
            y = zpmid[yf].sel(phase="2p") / A
            ydata[yf] = y
        for yf in [
            "sigma_eff_mid",
            "sigma_eff",
            "sigma_turb_mid",
            "sigma_turb",
            "sigma_th_mid",
            "sigma_th",
            "H",
        ]:
            y = zpmid[yf].sel(phase="2p")
            ydata[yf] = y
        for yf in [
            "PDE_whole_approx",
            "PDE_2p_avg_approx",
            "PDE_2p_mid_approx",
            "PDE_whole_approx_sp",
            "PDE_2p_avg_approx_sp",
            "PDE_2p_mid_approx_sp",
            "PDE_whole_full",
            "PDE_2p_avg_full",
            "PDE_2p_mid_full",
            "H_whole_full",
            "H_2p_avg_full",
            "H_2p_mid_full",
            "sfr10",
            "sfr40",
            "sfr100",
            "Sigma_gas",
        ]:
            ydata[yf] = zpmid[yf]
        ydata["sfr"] = zpmid[sfr_field]
        # full
        ydata["W"] = zpwmid["W"] * s.u.pok
        ydata["Wsg"] = zpwmid["Wsg"] * s.u.pok
        ydata["Wext"] = zpwmid["Wext"] * s.u.pok
        # top
        ydata["W_top"] = zpmid["W_top"].sum(dim="phase") * s.u.pok
        ydata["Wsg_top"] = zpmid["Wsg_top"].sum(dim="phase") * s.u.pok
        ydata["Wext_top"] = zpmid["Wext_top"].sum(dim="phase") * s.u.pok
        # difference
        ydata["dW"] = ydata["W"] - ydata["W_top"]
        ydata["dWsg"] = ydata["Wsg"] - ydata["Wsg_top"]
        ydata["dWext"] = ydata["Wext"] - ydata["Wext_top"]

        rhos = s.par["problem"]["SurfS"] / (2 * s.par["problem"]["zstar"])
        rhod = s.par["problem"]["rhodm"]
        ydata["rhotot"] = (
            ydata["nH"] * (1.4 * ac.m_p / au.cm**3).to("Msun/pc^3").value + rhos + rhod
        )
        ydata["tdep"] = zpmid["Sigma_gas"] / zpmid[sfr_field]
        ydata["tdep10"] = zpmid["Sigma_gas"] / zpmid["sfr10"]
        ydata["tdep40"] = zpmid["Sigma_gas"] / zpmid["sfr40"]
        ydata["tdep100"] = zpmid["Sigma_gas"] / zpmid["sfr100"]
        ydata["Zgas"] = s.Zgas * zpmid["Sigma_gas"] / zpmid["Sigma_gas"]
        ydata["Zdust"] = s.Zdust * zpmid["Sigma_gas"] / zpmid["Sigma_gas"]
        ydata["PDE"] = zpmid["PDE_2p_avg_full"]
        if reduce:
            # apply only on the good data points
            ydata = ydata.where(
                ~ydata["Ytot"].isin([np.nan, np.inf, -np.inf]), drop=True
            )
            ydata = ydata.reduce(func, dim="time", **func_kwargs)
        ydata = ydata.assign_coords(name=m)
        ydata = ydata.to_array().drop("phase")

        return ydata

    def add_legend(self, kind="R8", main=True, beta=True, beta_loc=5, **kwargs):
        colors = self.plt_kwargs[kind]["colors"]
        from matplotlib.lines import Line2D

        if main:
            custom_lines = []
            for c in colors:
                custom_lines.append(Line2D([0], [0], color=c))
            leg1 = plt.legend(
                custom_lines,
                ["(3,3)", "(1,1)", "(0.3,0.3)", "(0.1,0.1)", "(0.1,0.025)"],
                title=r"$(Z_{\rm g}^\prime,Z_{\rm d}^\prime)$",
                **kwargs,
            )
        if beta:
            if "loc" in kwargs:
                kwargs.pop("loc")
            beta_def = 1
            beta_alt = 10
            labels = [
                r"$\beta_0 = {}$".format(beta_def),
                r"$\beta_0 = {}$".format(beta_alt),
            ]
            custom_lines2 = [
                Line2D([0], [0], ls="-", color="k"),
                Line2D([0], [0], ls="--", color="k"),
            ]
            plt.legend(custom_lines2, labels, loc=beta_loc, **kwargs)
            if main:
                plt.gca().add_artist(leg1)

    def get_PW_time_series(
        self, m, dt=0, zrange=slice(-10, 10), recal=False, silent=False
    ):
        s = self.sa.set_model(m)

        # test needs for recalculation
        zpfiles = [
            os.path.join(s.basedir, "zprof", "{}.PWzprof.nc".format(s.problem_id)),
            os.path.join(s.basedir, "zprof", "{}.zpmid.nc".format(s.problem_id)),
            os.path.join(s.basedir, "zprof", "{}.zpwmid.nc".format(s.problem_id)),
        ]
        if not silent:
            print("Getting P, W time series for {}".format(m))

        for f in zpfiles:
            isexist = os.path.isfile(f)
            if isexist:
                isold = os.path.getmtime(f) < os.path.getmtime(s.files["zprof"][-1])
                recal = recal | isold
                if isold:
                    if not silent:
                        print("  -- {} is old".format(f))
                    break
            else:
                if not silent:
                    print("  -- {} is not available".format(f))
                recal = recal | (~isexist)
                break

        if not recal:
            if not silent:
                print("  -- read from files")
        else:
            if not silent:
                print("  -- recalculate from zprof")

        zprof = get_PW_zprof(s, recal=recal)
        zpmid, zpwmid = get_PW_time_series_from_zprof(
            s, zprof, dt=dt, zrange=zrange, recal=recal
        )
        return zpmid, zpwmid


def add_torb(ax, torb, ticklabels=True):
    # Define function and its inverse
    f = lambda x: x / torb  # noqa [E731]
    g = lambda x: torb * x  # noqa [E731]

    ax2 = ax.secondary_xaxis("top", functions=(f, g))
    plt.setp(ax2.get_xticklabels(), visible=ticklabels)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis="x", which="both", top=False)
    return ax2


def get_smoothed(y, t, dt):
    dt = 5
    window = int(dt / t.diff().median())
    return y.rolling(window=window, center=True, min_periods=1).mean()


def plot_errors(x, y, quantiles=[0.16, 0.5, 0.84], **kwargs):
    qx = list(x.quantile(quantiles))
    qy = list(y.quantile(quantiles))
    plt.errorbar(
        qx[1],
        qy[1],
        yerr=[[qy[1] - qy[0]], [qy[2] - qy[1]]],
        xerr=[[qx[1] - qx[0]], [qx[2] - qx[1]]],
        **kwargs,
    )


def add_boxplot(
    pdata,
    group="R8-b1",
    field="sfr40",
    offset=0,
    tslice=None,
    label=True,
    beta_label=False,
):
    if tslice is None:
        torb = pdata.torb_code[group.split("-")[0]]
        tslice = slice(torb * 2, torb * 5)
    namelist, ylist, Zdlist = pdata.collect_hst_list(field, group=group, tslice=tslice)
    Zd_to_idx = {3: 0, 1: 1, 0.3: 2, 0.1: 3, 0.025: 4}
    pos = np.array([Zd_to_idx[Zd] for Zd in Zdlist])
    width = 0.4

    colors = [pdata.plt_kwargs[group]["colors"][idx] for idx in pos]
    ls = pdata.plt_kwargs[group]["ls"]

    if "b10" in group:
        pos = pos.astype("float") + width * 1.1
    box = plt.boxplot(
        ylist,
        positions=pos + offset,
        widths=width,
        showfliers=False,
        patch_artist=True,
        medianprops=dict(color="k", ls=ls),
    )

    for artist, c in zip(box["boxes"], colors):
        plt.setp(artist, color=c, alpha=0.8, ls=ls)
    for element in ["whiskers", "caps"]:
        for artist1, artist2, c in zip(box[element][::2], box[element][1::2], colors):
            plt.setp(artist1, color=c, ls=ls)
            plt.setp(artist2, color=c, ls=ls)
    if label:
        labels = [
            r"$Z_{\rm g}^\prime=1$" + "\n" + r"$Z_{\rm d}^\prime =1$",
            r"$Z_{\rm g}^\prime=0.3$" + "\n" + r"$Z_{\rm d}^\prime =0.3$",
            r"$Z_{\rm g}^\prime=0.1$" + "\n" + r"$Z_{\rm d}^\prime =0.1$",
            r"$Z_{\rm g}^\prime=0.1$" + "\n" + r"$Z_{\rm d}^\prime =0.025$",
            r"$Z_{\rm g}^\prime=3$" + "\n" + r"$Z_{\rm d}^\prime =3$",
        ]
        for c, name, x, y in zip(colors, labels, pos, box["caps"][1::2]):
            y0, y1 = y.get_ydata()
            plt.annotate(
                name,
                (x, y0 * 1.05),
                xycoords="data",
                ha="center",
                va="bottom",
                fontsize="x-small",
                color=c,
            )
    if beta_label:
        for c, x, y in zip(colors, pos, box["medians"]):
            if "b10" in group:
                name = r"$\beta_0=10$"
            else:
                name = r"$\beta_0=1$"
            y0, y1 = y.get_ydata()
            plt.annotate(
                name,
                (x, y0 * 0.99),
                xycoords="data",
                ha="center",
                va="top",
                fontsize="x-small",
                color="k",
            )
    plt.ylabel(r"$\Sigma_{\rm SFR}\,[M_\odot\,{\rm kpc^{-2}\,yr^{-1}}]$")
    ax = plt.gca()
    ax.grid(visible=False)
    ax.xaxis.set_tick_params(which="both", bottom=False, top=False, labelbottom=False)


def get_PW_zprof(s, recal=False):
    fzpmid = os.path.join(s.basedir, "zprof", "{}.PWzprof.nc".format(s.problem_id))
    if not recal:
        if os.path.isfile(fzpmid):
            with xr.open_dataset(fzpmid) as zpmid:
                zpmid.load()
            return zpmid

    if hasattr(s, "newzp"):
        zp = s.newzp
    else:
        zp = PaperData.zprof_rename(s)

    zpmid = xr.Dataset()
    dm = s.domain
    dz = dm["dx"][2]
    # calculate weight first
    uWext = (
        zp["dWext"].sel(z=slice(0, dm["re"][2]))[:, ::-1].cumsum(dim="z")[:, ::-1] * dz
    )
    lWext = -zp["dWext"].sel(z=slice(dm["le"][2], 0)).cumsum(dim="z") * dz
    Wext = xr.concat([lWext, uWext], dim="z")
    if "dWsg" in zp:
        uWsg = (
            zp["dWsg"].sel(z=slice(0, dm["re"][2]))[:, ::-1].cumsum(dim="z")[:, ::-1]
            * dz
        )
        lWsg = -zp["dWsg"].sel(z=slice(dm["le"][2], 0)).cumsum(dim="z") * dz
        Wsg = xr.concat([lWsg, uWsg], dim="z")
    W = Wext + Wsg
    zpmid["Wext"] = Wext
    zpmid["Wsg"] = Wsg
    zpmid["W"] = W

    # caculate radiation pressure
    if "frad_z0" in zp:
        frad_list = ["frad_z0", "frad_z1", "frad_z2"]
        for frad in frad_list:
            uPrad = (
                zp[frad].sel(z=slice(0, dm["re"][2]))[:, ::-1].cumsum(dim="z")[:, ::-1]
                * dz
            )
            lPrad = -zp[frad].sel(z=slice(dm["le"][2], 0)).cumsum(dim="z") * dz
            zpmid[frad] = xr.concat([lPrad, uPrad], dim="z")
        zpmid["Prad"] = zpmid[frad_list].to_array().sum(dim="variable")

    # Pressures/Stresses
    zpmid["Pth"] = zp["P"]
    zpmid["Pturb"] = 2.0 * zp["Ek3"]
    zpmid["Ptot"] = zpmid["Pth"] + zpmid["Pturb"]
    if "PB1" in zp:
        zpmid["Pmag"] = zp["PB1"] + zp["PB2"] + zp["PB3"]
        zpmid["Pimag"] = zpmid["Pmag"] - 2.0 * zp["PB3"]
        zpmid["dPmag"] = zp["dPB1"] + zp["dPB2"] + zp["dPB3"]
        zpmid["dPimag"] = zpmid["dPmag"] - 2.0 * zp["dPB3"]
        zpmid["oPmag"] = zpmid["Pmag"] - zpmid["dPmag"]
        zpmid["oPimag"] = zpmid["Pimag"] - zpmid["dPimag"]
        zpmid["Ptot"] += zpmid["Pimag"]

    # scale height
    zpmid["H"] = np.sqrt((zp["d"] * zp.z**2).sum(dim="z") / zp["d"].sum(dim="z"))

    # density, area
    zpmid["nH"] = zp["d"]
    zpmid["A"] = zp["A"]

    # heat and cool
    if "cool" in zp:
        zpmid["heat"] = zp["heat"]
        zpmid["cool"] = zp["cool"]
        if "netcool" in zp:
            zpmid["net_cool"] = zp["net_cool"]
        else:
            zpmid["net_cool"] = zp["cool"] - zp["heat"]

    # Erad
    if "Erad0" in zp:
        zpmid["Erad0"] = zp["Erad0"]
    if "Erad1" in zp:
        zpmid["Erad1"] = zp["Erad1"]
    if "Erad2" in zp:
        zpmid["Erad2"] = zp["Erad2"]

    # rearrange phases
    twop = (
        zpmid.sel(phase=["CMM", "CNM", "UNM", "WNM"])
        .sum(dim="phase")
        .assign_coords(phase="2p")
    )
    hot = zpmid.sel(phase=["WHIM", "HIM"]).sum(dim="phase").assign_coords(phase="hot")
    if "WIM" in zpmid.phase:
        wim = zpmid.sel(phase=["WIM"]).sum(dim="phase").assign_coords(phase="WIM")
        zpmid = xr.concat([twop, wim, hot], dim="phase")
    else:
        zpmid = xr.concat([twop, hot], dim="phase")

    if os.path.isfile(fzpmid):
        os.remove(fzpmid)
    zpmid.to_netcdf(fzpmid)

    return zpmid


def get_PW_time_series_from_zprof(
    s, zprof, sfr=None, dt=0, zrange=slice(-10, 10), recal=False
):
    fzpmid = os.path.join(s.basedir, "zprof", "{}.zpmid.nc".format(s.problem_id))
    fzpw = os.path.join(s.basedir, "zprof", "{}.zpwmid.nc".format(s.problem_id))

    if (os.path.isfile(fzpmid) and os.path.isfile(fzpw)) and (not recal):
        zpmid = xr.open_dataset(fzpmid)
        zpmid.close()
        zpmid.attrs = PaperData.set_zprof_attr()
        # smoothing
        if dt > 0:
            window = int(dt / zpmid.time.diff(dim="time").median())
            zpmid_t = zpmid.rolling(time=window, center=True, min_periods=1).mean()
        else:
            zpmid_t = zpmid
        zpwmid = xr.open_dataset(fzpw)
        zpwmid.close()
        return zpmid_t, zpwmid

    # szeff
    szeff = np.sqrt(zprof["Ptot"].sum(dim="z") / zprof["nH"].sum(dim="z"))
    vzeff = np.sqrt(zprof["Pturb"].sum(dim="z") / zprof["nH"].sum(dim="z"))
    cseff = np.sqrt(zprof["Pth"].sum(dim="z") / zprof["nH"].sum(dim="z"))

    # select midplane
    zpmid = zprof.sel(z=zrange).mean(dim="z")
    # select top and bottom
    hzmax = s.par["domain1"]["x3max"] * 0.5
    for Pcomp in [
        "Pth",
        "Pturb",
        "Pmag",
        "Pimag",
        "dPmag",
        "dPimag",
        "oPmag",
        "oPimag",
        "Ptot",
        "Wext",
        "Wsg",
        "W",
        "Prad",
    ]:
        if Pcomp in zpmid:
            k = "{}_top".format(Pcomp)
            zpmid[k] = xr.concat(
                [
                    zprof[Pcomp].sel(z=slice(hzmax - 10, hzmax + 10)),
                    zprof[Pcomp].sel(z=slice(-hzmax - 10, -hzmax + 10)),
                ],
                dim="z",
            ).mean(dim="z")
    zpmid["A_top"] = xr.concat(
        [
            zprof["A"].sel(z=slice(hzmax - 10, hzmax + 10)),
            zprof["A"].sel(z=slice(-hzmax - 10, -hzmax + 10)),
        ],
        dim="z",
    ).mean(dim="z")
    zpwmid = zprof.sel(z=zrange).mean(dim="z").sum(dim="phase")

    # SFR from history
    vol = np.prod(s.domain["Lx"])
    area = vol / s.domain["Lx"][2]

    h = pa.read_hst(s.files["hst"])

    zpmid["sfr10"] = xr.DataArray(
        np.interp(zpmid.time_code, h["time"], h["sfr10"]), coords=[zpmid.time]
    )
    zpmid["sfr40"] = xr.DataArray(
        np.interp(zpmid.time_code, h["time"], h["sfr40"]), coords=[zpmid.time]
    )
    zpmid["sfr100"] = xr.DataArray(
        np.interp(zpmid.time_code, h["time"], h["sfr100"]), coords=[zpmid.time]
    )

    if sfr is None:
        zpmid["sfr"] = xr.DataArray(
            np.interp(zpmid.time_code, h["time"], h["sfr10"]), coords=[zpmid.time]
        )
    else:
        zpmid["sfr"] = sfr

    # sz from history
    KE = h["x1KE"] + h["x2KE"] + h["x3KE"]
    if "x1ME" in h:
        szmag = h["x1ME"] + h["x2ME"] - 2.0 * h["x3ME"]
        ME = h["x1ME"] + h["x2ME"] + h["x3ME"]
    else:
        szmag = 0.0
        ME = 0.0

    P = (h["totalE"] - KE - ME) * (5 / 3.0 - 1)
    szeff_mid = np.sqrt((2.0 * h["x3KE"] + P + szmag) / h["mass"])
    zpmid["szeff"] = xr.DataArray(
        np.interp(zpmid.time_code, h["time"], szeff_mid), coords=[zpmid.time]
    )

    # PDE

    zpmid["sigma_eff"] = szeff  # vertically integrated
    zpmid["sigma_turb"] = vzeff
    zpmid["sigma_th"] = cseff
    zpmid["sigma_eff_mid"] = np.sqrt(zpmid["Ptot"] / zpmid["nH"])  # midplane
    zpmid["sigma_turb_mid"] = np.sqrt(zpmid["Pturb"] / zpmid["nH"])
    zpmid["sigma_th_mid"] = np.sqrt(zpmid["Pth"] / zpmid["nH"])
    zpmid["Sigma_gas"] = xr.DataArray(
        np.interp(zpmid.time_code, h["time"], h["mass"] * s.u.Msun * vol / area),
        coords=[zpmid.time],
    )
    zpmid["Sigma_sp"] = xr.DataArray(
        np.interp(zpmid.time_code, h["time"], h["msp"] * s.u.Msun * vol / area),
        coords=[zpmid.time],
    )
    rhosd = (
        0.5 * s.par["problem"]["SurfS"] / s.par["problem"]["zstar"]
        + s.par["problem"]["rhodm"]
    )
    rhosd2 = (
        0.5
        * (s.par["problem"]["SurfS"] + zpmid["Sigma_sp"])
        / s.par["problem"]["zstar"]
        + s.par["problem"]["rhodm"]
    )
    zpmid["PDE1"] = (
        np.pi
        * zpmid["Sigma_gas"] ** 2
        / 2.0
        * (ac.G * (ac.M_sun / ac.pc**2) ** 2 / ac.k_B).cgs.value
    )
    zpmid["PDE2_2p"] = (
        zpmid["Sigma_gas"]
        * np.sqrt(2 * rhosd)
        * zpmid["sigma_eff"].sel(phase="2p")
        * (
            np.sqrt(ac.G * ac.M_sun / ac.pc**3)
            * (ac.M_sun / ac.pc**2)
            * au.km
            / au.s
            / ac.k_B
        ).cgs.value
    )
    zpmid["PDE2_2p_mid"] = (
        zpmid["Sigma_gas"]
        * np.sqrt(2 * rhosd)
        * zpmid["sigma_eff_mid"].sel(phase="2p")
        * (
            np.sqrt(ac.G * ac.M_sun / ac.pc**3)
            * (ac.M_sun / ac.pc**2)
            * au.km
            / au.s
            / ac.k_B
        ).cgs.value
    )
    zpmid["PDE2"] = (
        zpmid["Sigma_gas"]
        * np.sqrt(2 * rhosd)
        * zpmid["szeff"]
        * (
            np.sqrt(ac.G * ac.M_sun / ac.pc**3)
            * (ac.M_sun / ac.pc**2)
            * au.km
            / au.s
            / ac.k_B
        ).cgs.value
    )
    zpmid["PDE3_2p"] = (
        zpmid["Sigma_gas"]
        * np.sqrt(2 * rhosd2)
        * zpmid["sigma_eff"].sel(phase="2p")
        * (
            np.sqrt(ac.G * ac.M_sun / ac.pc**3)
            * (ac.M_sun / ac.pc**2)
            * au.km
            / au.s
            / ac.k_B
        ).cgs.value
    )
    zpmid["PDE3_2p_mid"] = (
        zpmid["Sigma_gas"]
        * np.sqrt(2 * rhosd2)
        * zpmid["sigma_eff_mid"].sel(phase="2p")
        * (
            np.sqrt(ac.G * ac.M_sun / ac.pc**3)
            * (ac.M_sun / ac.pc**2)
            * au.km
            / au.s
            / ac.k_B
        ).cgs.value
    )
    zpmid["PDE3"] = (
        zpmid["Sigma_gas"]
        * np.sqrt(2 * rhosd2)
        * zpmid["szeff"]
        * (
            np.sqrt(ac.G * ac.M_sun / ac.pc**3)
            * (ac.M_sun / ac.pc**2)
            * au.km
            / au.s
            / ac.k_B
        ).cgs.value
    )
    zpmid["PDE_2p_mid_approx"] = (
        zpmid["PDE1"] + zpmid["PDE2_2p_mid"]
    )  # PDE from midplane pressure and density of 2p gas
    zpmid["PDE_2p_avg_approx"] = (
        zpmid["PDE1"] + zpmid["PDE2_2p"]
    )  # PDE from mass weighted mean VD of 2p gas
    zpmid["PDE_whole_approx"] = (
        zpmid["PDE1"] + zpmid["PDE2"]
    )  # PDE from mass weighted mean VD of whole gas
    zpmid["PDE_2p_mid_approx_sp"] = (
        zpmid["PDE1"] + zpmid["PDE3_2p_mid"]
    )  # PDE from midplane pressure and density of 2p gas + added new star particle gravity
    zpmid["PDE_2p_avg_approx_sp"] = (
        zpmid["PDE1"] + zpmid["PDE3_2p"]
    )  # PDE from mass weighted mean VD of 2p gas + added new star particle gravity
    zpmid["PDE_whole_approx_sp"] = (
        zpmid["PDE1"] + zpmid["PDE3"]
    )  # PDE from mass weighted mean VD of whole gas + added new star particle gravity

    # using PRFM package
    import prfm

    Sigma_star = s.par["problem"]["SurfS"]
    H_star = s.par["problem"]["zstar"]
    rho_dm = s.par["problem"]["rhodm"]
    Sigma_gas = zpmid["Sigma_gas"]

    for sz, label in zip(
        [
            zpmid["szeff"],
            zpmid["sigma_eff"].sel(phase="2p"),
            zpmid["sigma_eff_mid"].sel(phase="2p"),
        ],
        ["whole", "2p_avg", "2p_mid"],
    ):
        model = prfm.PRFM(
            Sigma_gas=Sigma_gas,
            Sigma_star=Sigma_star,
            H_star=H_star,
            rho_dm=rho_dm,
            astro_units=True,
            sigma_eff=sz,
        )
        model.calc_self_consistent_solution()
        zpmid[f"PDE_{label}_full"] = xr.DataArray(model.Wtot, coords=[zpmid.time])
        zpmid[f"H_{label}_full"] = xr.DataArray(model.H_gas, coords=[zpmid.time])

    if os.path.isfile(fzpmid):
        os.remove(fzpmid)
    if os.path.isfile(fzpw):
        os.remove(fzpw)

    zpmid.to_netcdf(fzpmid)
    zpwmid.to_netcdf(fzpw)

    zpmid.attrs = PaperData.set_zprof_attr()

    # smoothing
    if dt > 0:
        window = int(dt / zpmid.time.diff(dim="time").median())
        zpmid = zpmid.rolling(time=window, center=True, min_periods=1).mean()

    return zpmid, zpwmid


def plot_DE(
    pdata,
    m,
    tr,
    xf,
    yf,
    sfrfield="sfr",
    label="",
    ax=None,
    fit=False,
    qr=[0.16, 0.5, 0.84],
):
    Punit_label = r"$/k_B\,[{\rm cm^{-3}\,K}]$"
    sfr_unit_label = r"$\,[M_\odot{\rm \,kpc^{-2}\,yr}]$"

    if ax is None:
        ax = plt.gca()
    plt.sca(ax)
    s = pdata.sa.set_model(m)
    zpmid, zpwmid = pdata.get_PW_time_series(m)

    zpmid = zpmid.sel(time=tr)
    zpwmid = zpwmid.sel(time=tr)

    wpdata = dict(
        sfr=zpmid[sfrfield],
        W=zpwmid["W"] * s.u.pok,
        PDE=zpmid["PDE_2p"],
        Ptot=zpmid["Ptot"].sel(phase="2p") / zpmid["A"].sel(phase="2p") * s.u.pok,
    )

    x = wpdata[xf]
    y = wpdata[yf]
    plt.plot(x, y, "o", markersize=5, markeredgewidth=0, color=s.color, alpha=0.3)
    if qr is None:
        xavg = x.mean().data
        xstd = x.std().data
        yavg = y.mean().data
        ystd = y.std().data
        plt.errorbar(
            xavg,
            yavg,
            xerr=[[xstd], [xstd]],
            yerr=[[ystd], [ystd]],
            marker="*",
            markersize=8,
            ecolor="k",
            markeredgecolor="k",
            color=s.color,
            zorder=10,
            label=label,
        )
    else:
        qx = x.quantile(qr).data
        qy = y.quantile(qr).data
        plt.errorbar(
            qx[1],
            qy[1],
            xerr=[[qx[1] - qx[0]], [qx[2] - qx[1]]],
            yerr=[[qy[1] - qy[0]], [qy[2] - qy[1]]],
            marker="o",
            markersize=8,
            ecolor="k",
            markeredgecolor="k",
            color=s.color,
            zorder=10,
            label=label,
        )
    xl = zpmid.attrs["Plabels"][xf]
    if xf != "W":
        xl += r"${}_{\rm ,2p}$"
    xl += Punit_label
    if yf == "sfr":
        yl = r"$\Sigma_{\rm SFR}$" + sfr_unit_label
    else:
        yl = zpmid.attrs["Plabels"][yf]
        if yf != "W":
            yl += r"${}_{\rm ,2p}$"
        yl += Punit_label
    plt.xlabel(xl)
    plt.ylabel(yl)
    # draw reference line
    Prange = np.logspace(2, 7)
    plt.plot(Prange, Prange, ls=":", color="k")
    plt.yscale("log")
    plt.xscale("log")
    plt.xlim(5.0e3, 5.0e5)
    if yf == "sfr":
        plt.ylim(5.0e-4, 5.0e-1)
    else:
        plt.ylim(5.0e3, 5.0e5)
        ax.set_aspect("equal")

    if fit:
        if xf == "W" and yf == "Ptot":
            plt.plot(
                Prange, 10.0 ** (0.99 * np.log10(Prange) + 0.083), ls="-", color="k"
            )
        if xf.startswith("PDE") and yf == "Ptot":
            plt.plot(
                Prange, 10.0 ** (1.03 * np.log10(Prange) - 0.199), ls="-", color="k"
            )
        if xf.startswith("PDE") and yf == "W":
            plt.plot(
                Prange, 10.0 ** (1.03 * np.log10(Prange) - 0.276), ls="-", color="k"
            )
        if xf == "W" and yf == "sfr":
            plt.plot(
                Prange, 10.0 ** (1.18 * np.log10(Prange) - 7.32), ls="-", color="k"
            )
        if xf.startswith("PDE") and yf == "sfr":
            plt.plot(
                Prange, 10.0 ** (1.18 * np.log10(Prange) - 7.32), ls="-", color="k"
            )
        if xf == "Ptot" and yf == "sfr":
            plt.plot(
                Prange, 10.0 ** (1.17 * np.log10(Prange) - 7.43), ls="-", color="k"
            )


def plot_Pcomp(
    pdata, m, tr, yf, xf="PDE", label="", ax=None, fit=False, qr=[0.16, 0.5, 0.84]
):
    s = pdata.sa.set_model(m)
    Punit_label = r"$/k_B\,[{\rm cm^{-3}\,K}]$"

    if ax is None:
        ax = plt.gca()
    plt.sca(ax)

    zpmid, zpw = pdata.get_PW_time_series(m)

    zpmid = zpmid.sel(time=tr)
    zpw = zpw.sel(time=tr)

    wpdata = dict(
        W=zpw["W"] * s.u.pok,
        PDE=zpmid["PDE_2p_avg_full"],
        Ptot=zpmid["Ptot"].sel(phase="2p") / zpmid["A"].sel(phase="2p") * s.u.pok,
    )

    Ptot = zpmid["Ptot"].sel(phase="2p")
    # A = zpmid['A'].sel(phase='2p')
    x = wpdata[xf]

    if yf not in zpmid:
        return

    y = zpmid[yf].sel(phase="2p") / Ptot
    plt.plot(x, y, "o", markersize=5, markeredgewidth=0, color=s.color, alpha=0.3)
    if qr is None:
        xavg = x.mean().data
        xstd = x.std().data
        yavg = y.mean().data
        ystd = y.std().data
        plt.errorbar(
            xavg,
            yavg,
            xerr=[[xstd], [xstd]],
            yerr=[[ystd], [ystd]],
            marker="*",
            markersize=8,
            ecolor="k",
            markeredgecolor="k",
            color=s.color,
            zorder=10,
            label=label,
        )
    else:
        qx = x.quantile(qr).data
        qy = y.quantile(qr).data
        plt.errorbar(
            qx[1],
            qy[1],
            xerr=[[qx[1] - qx[0]], [qx[2] - qx[1]]],
            yerr=[[qy[1] - qy[0]], [qy[2] - qy[1]]],
            marker="o",
            markersize=8,
            ecolor="k",
            markeredgecolor="k",
            color=s.color,
            zorder=10,
            label=label,
        )

    xl = zpmid.attrs["Plabels"][xf]
    if xf != "W":
        xl += r"${}_{\rm ,2p}$"
    xl += Punit_label
    yl = zpmid.attrs["Plabels"][yf]
    if yf != "W":
        yl += r"${}_{\rm ,2p}$"
    yl += r"$/$" + zpmid.attrs["Plabels"]["Ptot"] + r"${}_{\rm ,2p}$"

    plt.xlabel(xl)
    plt.ylabel(yl)

    # draw reference line
    Prange = np.logspace(2, 7)
    plt.yscale("log")
    plt.xscale("log")
    plt.ylim(1.0e-2, 1)
    plt.xlim(1.0e3, 1.0e6)
    if fit:
        if yf == "Pth":
            plt.plot(
                Prange, 10.0 ** (-0.275 * np.log10(Prange) + 0.517), ls="-", color="k"
            )
        if yf == "Pturb":
            plt.plot(
                Prange, 10.0 ** (0.129 * np.log10(Prange) - 0.918), ls="-", color="k"
            )


def plot_Pcomp_sfr(
    pdata, m, tr, yf, label="", ax=None, fit=False, qr=[0.16, 0.5, 0.84]
):
    s = pdata.sa.set_model(m)
    Punit_label = r"$/k_B\,[{\rm cm^{-3}\,K}]$"
    sfr_unit_label = r"$\,[M_\odot{\rm \,kpc^{-2}\,yr}]$"

    if ax is None:
        ax = plt.gca()
    plt.sca(ax)

    zpmid, zpw = pdata.get_PW_time_series(m)
    zpmid = zpmid.sel(time=tr)
    zpw = zpw.sel(time=tr)

    x = zpmid["sfr"]

    if yf not in zpmid:
        return

    A = zpmid["A"].sel(phase="2p")
    y = zpmid[yf].sel(phase="2p") / A * s.u.pok
    plt.plot(x, y, "o", markersize=5, markeredgewidth=0, color=s.color, alpha=0.3)
    qx = x.quantile(qr).data
    qy = y.quantile(qr).data
    plt.errorbar(
        qx[1],
        qy[1],
        xerr=[[qx[1] - qx[0]], [qx[2] - qx[1]]],
        yerr=[[qy[1] - qy[0]], [qy[2] - qy[1]]],
        marker="o",
        markersize=8,
        ecolor="k",
        markeredgecolor="k",
        color=s.color,
        zorder=10,
        label=label,
    )
    plt.xlabel(r"$\Sigma_{\rm SFR}$" + sfr_unit_label)
    plt.ylabel(zpmid.attrs["Plabels"][yf] + r"${}_{\rm ,2p}$" + Punit_label)

    # draw reference line
    plt.yscale("log")
    plt.xscale("log")
    plt.xlim(5.0e-4, 0.5)
    plt.ylim(1.0e3, 1.0e6)
    sfrrange = np.logspace(-4, 2)
    if fit:
        if yf == "Pth":
            plt.plot(
                sfrrange, 10.0 ** (0.603 * np.log10(sfrrange) + 4.99), ls="-", color="k"
            )
        #             plt.plot(sfrrange,10.**(0.86*np.log10(sfrrange)+5.69),ls=':',color='k')
        if yf in ["Pturb"]:  # ,'Pimag','oPimag','dPimag']:
            plt.plot(
                sfrrange, 10.0 ** (0.960 * np.log10(sfrrange) + 6.17), ls="-", color="k"
            )
        #             plt.plot(sfrrange,10.**(0.89*np.log10(sfrrange)+6.3),ls=':',color='k')
        if yf == "Ptot":
            plt.plot(
                sfrrange, 10.0 ** (0.840 * np.log10(sfrrange) + 6.26), ls="-", color="k"
            )


#             plt.plot(sfrrange,10.**(0.847*np.log10(sfrrange)+6.27),ls=':',color='k')


def plot_Upsilon_sfr(
    pdata, m, tr, xf, yf, label="", ax=None, fit=False, qr=[0.16, 0.5, 0.84]
):
    s = pdata.sa.set_model(m)
    yield_conv = (
        ((au.cm ** (-3) * au.K * ac.k_B) / (ac.M_sun / ac.kpc**2 / au.yr))
        .to("km/s")
        .value
    )
    Uunit_label = r"$\,[{\rm km/s}]$"
    Punit_label = r"$/k_B\,[{\rm cm^{-3}\,K}]$"
    sfr_unit_label = r"$\,[M_\odot{\rm \,kpc^{-2}\,yr}]$"

    if ax is None:
        ax = plt.gca()
    plt.sca(ax)

    zpmid, zpw = pdata.get_PW_time_series(m)

    zpmid = zpmid.sel(time=tr)
    zpw = zpw.sel(time=tr)

    if yf not in zpmid:
        return

    A = zpmid["A"].sel(phase="2p")
    y = zpmid[yf].sel(phase="2p") / A * s.u.pok / zpmid["sfr40"] * yield_conv
    if xf == "Zgas":
        x = s.Zgas * zpmid["time"] / zpmid["time"]
    elif xf == "Zdust":
        x = s.Zdust * zpmid["time"] / zpmid["time"]
    else:
        x = zpmid[xf]
    plt.plot(x, y, "o", markersize=5, markeredgewidth=0, color=s.color, alpha=0.3)

    qx = x.quantile(qr).data
    qy = y.quantile(qr).data
    plt.errorbar(
        qx[1],
        qy[1],
        xerr=[[qx[1] - qx[0]], [qx[2] - qx[1]]],
        yerr=[[qy[1] - qy[0]], [qy[2] - qy[1]]],
        marker="o",
        markersize=8,
        ecolor="k",
        markeredgecolor="k",
        color=s.color,
        zorder=10,
        label=label,
    )
    if xf == "sfr":
        plt.xlabel(r"$\Sigma_{\rm SFR}$" + sfr_unit_label)
    elif xf.startswith("PDE_whole"):
        plt.xlabel(r"$P_{\rm DE}$" + Punit_label)
    elif xf.startswith("PDE_2p"):
        plt.xlabel(r"$P_{\rm DE,2p}$" + Punit_label)
    elif xf == "Zgas":
        plt.xlabel(r"$Z_{\rm gas}^\prime$")
    elif xf == "Zdust":
        plt.xlabel(r"$Z_{\rm dust}^\prime$")
    plt.ylabel(
        r"$\Upsilon_{{\rm {}}}$".format(yf[2:] if yf.startswith("Pi") else yf[1:])
        + Uunit_label
    )

    # draw reference line
    plt.yscale("log")
    plt.xscale("log")
    plt.ylim(10, 3000)
    if xf == "sfr":
        plt.xlim(5.0e-4, 0.5)
        sfrrange = np.logspace(-4, 2)
        if fit:
            if yf == "Pth":
                #                 plt.plot(sfrrange,200*(sfrrange/1.e-2)**(-0.14),ls=':',color='k')
                plt.plot(
                    sfrrange, 110 * (sfrrange / 1.0e-2) ** (-0.4), ls="-", color="k"
                )
            if yf in ["Pturb", "Pimag", "oPimag", "dPimag"]:
                #                 plt.plot(sfrrange,700*(sfrrange/1.e-2)**(-0.11),ls=':',color='k')
                plt.plot(
                    sfrrange, 330 * (sfrrange / 1.0e-2) ** (-0.05), ls="-", color="k"
                )
            if yf == "Ptot":
                #                 plt.plot(sfrrange,770*(sfrrange/1.e-2)**(-0.15),ls=':',color='k')
                plt.plot(
                    sfrrange, 740 * (sfrrange / 1.0e-2) ** (-0.2), ls="-", color="k"
                )
    if xf.startswith("PDE"):
        plt.xlim(1.0e3, 1.0e6)
        Prange = np.logspace(3, 6)
        if fit:
            if yf == "Pth":
                plt.plot(
                    Prange,
                    10.0 ** (-0.506 * np.log10(Prange) + 4.45),
                    ls="-",
                    color="k",
                )
            if yf in ["Pturb"]:  # ,'Pimag','oPimag','dPimag']:
                plt.plot(
                    Prange,
                    10.0 ** (-0.060 * np.log10(Prange) + 2.81),
                    ls="-",
                    color="k",
                )
            if yf == "Ptot":
                plt.plot(
                    Prange,
                    10.0 ** (-0.212 * np.log10(Prange) + 3.86),
                    ls="-",
                    color="k",
                )


def plot_PWYstack(
    pdata,
    m,
    tr,
    i,
    plabel=True,
    Upsilon=False,
    label="",
    ax=None,
    qr=[0.16, 0.5, 0.84],
    errorbar_color="k",
):
    s = pdata.sa.set_model(m)
    Punit_label = r"$/k_B\,[10^4{\rm cm^{-3}\,K}]$"
    Punit = 1.0e-4 * s.u.pok
    Pcolors = cmr.get_sub_cmap("cmr.cosmic", 0.3, 1, N=4).colors
    Wcolors = cmr.get_sub_cmap("cmr.ember", 0.3, 1, N=4).colors
    yield_conv = (
        ((au.cm ** (-3) * au.K * ac.k_B) / (ac.M_sun / ac.kpc**2 / au.yr))
        .to("km/s")
        .value
    )
    Uunit_label = r"$\,[{\rm km/s}]$"
    if ax is None:
        ax = plt.gca()
    plt.sca(ax)

    zpmid, zpw = pdata.get_PW_time_series(m)
    zpmid = zpmid.sel(time=tr)
    zpw = zpw.sel(time=tr)

    # Ptot=zpmid['Ptot'].sel(phase='2p')
    A = zpmid["A"].sel(phase="2p")

    # pressure
    f0 = 0
    for yf, c, cl in zip(
        ["oPimag", "dPimag", "Pth", "Pturb"], Pcolors, ["w", "w", "k", "k"]
    ):
        if yf not in zpmid:
            continue
        if Upsilon:
            y = zpmid[yf].sel(phase="2p") / A * s.u.pok / zpmid["sfr40"] * yield_conv
            width = 0.8
            offset = 0.0
            align = "center"
        else:
            y = zpmid[yf].sel(phase="2p") / A * Punit
            width = -0.4
            offset = -0.2
            align = "edge"
        qy = y.quantile(qr).data
        plt.bar(label, qy[1], bottom=f0, color=c, width=width, align=align)

        if plabel:
            plt.annotate(
                zpmid.attrs["Ulabels"][yf] if Upsilon else zpmid.attrs["Plabels"][yf],
                (i + offset, f0 + 0.5 * qy[1]),
                color=cl,
                ha="center",
                va="center",
            )
        f0 = f0 + qy[1]
    if "Prad" in zpw:
        y = zpw["Prad"] * Punit
        qy = y.quantile(qr).data
        plt.bar(label, qy[1], bottom=f0, color=Wcolors[-1], width=width, align=align)
    if Upsilon:
        y = zpmid["Ptot"].sel(phase="2p") / A * s.u.pok / zpmid["sfr40"] * yield_conv
    else:
        y = zpmid["Ptot"].sel(phase="2p") / A * Punit
    qy = y.quantile(qr).data
    if errorbar_color is not None:
        plt.plot([i + offset, i + offset], [qy[0], qy[2]], color=errorbar_color)
    #     plt.plot([i+offset,i+offset],[qy[1],qy[1]],'or')

    # weight
    if not Upsilon:
        f0 = 0
        for yf, c, cl in zip(["Wsg", "Wext"], Wcolors[1:], ["w", "k"]):
            if yf not in zpmid:
                continue
            y = zpw[yf] * Punit
            qy = y.quantile(qr).data
            plt.bar(label, qy[1], bottom=f0, color=c, width=-width, align=align)
            if plabel:
                plt.annotate(
                    zpmid.attrs["Plabels"][yf],
                    (i - offset, f0 + 0.5 * qy[1]),
                    color=cl,
                    ha="center",
                    va="center",
                )
            f0 = f0 + qy[1]
        y = (zpw["W"]) * Punit
        qy = y.quantile(qr).data
        if errorbar_color is not None:
            plt.plot([i - offset, i - offset], [qy[0], qy[2]], color=errorbar_color)
        plt.ylabel(r"$P$" + Punit_label)
    else:
        plt.ylabel(r"$\Upsilon$" + Uunit_label)
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)


def plot_PWYbox(
    pdata,
    m,
    tr,
    i,
    nmodels,
    legend=True,
    Upsilon=False,
    Pcomps=["Ptot", "Pturb", "Pth", "dPimag", "oPimag"],
    label="",
    ax=None,
    edge_color="k",
):
    s = pdata.sa.set_model(m)

    Punit_label = r"$/k_B\,[10^4{\rm cm^{-3}\,K}]$"
    # Punit = 1.e-4*s.u.pok
    Pcolors = cmr.get_sub_cmap("cmr.cosmic", 0.3, 1, N=4).colors
    Pcolors = cmr.get_sub_cmap("cmr.neutral", 0.3, 0.9, N=4).colors
    # Wcolors=cmr.get_sub_cmap('cmr.ember',0.3,1,N=4).colors
    yield_conv = (
        ((au.cm ** (-3) * au.K * ac.k_B) / (ac.M_sun / ac.kpc**2 / au.yr))
        .to("km/s")
        .value
    )
    Uunit_label = r"$\,[{\rm km/s}]$"
    if ax is None:
        ax = plt.gca()
    plt.sca(ax)

    zpmid, zpw = pdata.get_PW_time_series(m)
    zpmid = zpmid.sel(time=tr)
    zpw = zpw.sel(time=tr)

    # Ptot=zpmid['Ptot'].sel(phase='2p')
    A = zpmid["A"].sel(phase="2p")
    ncomp = len(Pcomps)
    w = 0.8 / nmodels

    Ycomp = []
    pos = []
    Ylabels = []
    for j, yf in enumerate(Pcomps):
        if yf == "sfr":
            y = zpmid["sfr40"]
            Ylabels.append(r"$\Sigma_{\rm SFR}$")
        else:
            if Upsilon:
                Ylabels.append(zpmid.attrs["Ulabels"][yf])
            else:
                Ylabels.append(zpmid.attrs["Plabels"][yf])
            if yf not in zpmid:
                continue
            if Upsilon:
                y = (
                    zpmid[yf].sel(phase="2p")
                    / A
                    * s.u.pok
                    / zpmid["sfr40"]
                    * yield_conv
                )
            else:
                y = zpmid[yf].sel(phase="2p") / A * s.u.pok

        Ycomp.append(y.dropna(dim="time"))

        offset = (i) / nmodels * 0.8 - 0.4 + 0.5 * w
        pos.append(j + offset)
    box = plt.boxplot(
        Ycomp,
        positions=pos,
        widths=w,
        whis=[16, 84],
        showfliers=False,
        patch_artist=True,
        #                       showmeans=True,
        #                       meanprops=dict(markerfacecolor='tab:orange',markeredgecolor='w',
        #                                      markersize=5,markeredgewidth=0.5,marker='*')
    )
    for element in ["boxes", "whiskers", "fliers", "means", "medians", "caps"]:
        plt.setp(box[element], color=edge_color)
    plt.setp(box["boxes"], facecolor=s.color)

    plt.xticks(ticks=np.arange(ncomp), labels=Ylabels, fontsize="medium")
    plt.xlim(-0.5, ncomp - 0.5)
    for c, yf, xc in zip(Pcolors, Pcomps[::-1], np.arange(ncomp)[::-1]):
        plt.axvspan(xc - 0.5, xc + 0.5, color=c, alpha=0.1, lw=0)
    if Upsilon:
        plt.ylabel(r"$\Upsilon$" + Uunit_label)
    else:
        plt.ylabel(r"$P$" + Punit_label)
    plt.yscale("log")
    if legend:
        x0 = 0.85
        dx = 0.02
        y0 = 0.95
        dy = 0.05

        c = s.color
        plt.annotate(
            "        ",
            (x0, y0 - dy * i),
            xycoords="axes fraction",
            ha="right",
            va="top",
            size="xx-small",
            backgroundcolor=c,
            color=c,
        )
        plt.annotate(
            label,
            (x0 + dx, y0 - dy * i),
            xycoords="axes fraction",
            size="xx-small",
            ha="left",
            va="top",
        )
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)


class athena_data(object):
    def __init__(self, s, data):
        self.sim = s
        self.data = data
        self.time = data.time
        if s.par["configure"]["ShearingBox"] == "yes":
            self.shear = True
            self.qshear = s.par["problem"]["qshear"]
            self.Omega = s.par["problem"]["Omega"]
            self.Lx = s.par["domain1"]["x1max"] - s.par["domain1"]["x1min"]
            self.qOmL = self.qshear * self.Omega * self.Lx

    def __repr__(self):
        return self.data.__repr__()

    def keys(self):
        return list(self.data.keys())

    def derived_keys(self):
        return list(self.sim.dfi.keys())

    def __getitem__(self, field):
        if field in self.data:
            return self.data[field]
        elif field == "T":
            if "temperature" in self.data:
                return self.data["temperature"]
            elif "xe" in self.data:
                d = self.data
                u = self.sim.u
                return (
                    d["pressure"]
                    / (d["density"] * (1.1 + d["xe"] - d["xH2"]))
                    / (ac.k_B / u.energy_density).cgs.value
                )
        elif field in self.sim.dfi:
            return self.sim.dfi[field]["func"](self.data, self.sim.u)
        elif field == "charging":
            return 1.7 * self["chi_FUV"] * np.sqrt(self["T"]) / (self["ne"]) + 50.0
        elif field == "eps_PE":
            CPE_ = np.array([5.22, 2.25, 0.04996, 0.00430, 0.147, 0.431, 0.692])
            T = self["T"]
            x = self["charging"]
            eps = (CPE_[0] + CPE_[1] * np.power(T, CPE_[4])) / (
                1.0
                + CPE_[2]
                * np.power(x, CPE_[5])
                * (1.0 + CPE_[3] * np.power(x, CPE_[6]))
            )
            return eps
        elif field == "heat_PE":
            return 1.7e-26 * self["chi_PE"] * self.sim.Zdust * self["eps_PE"]
        elif field == "phase":
            self.data[field] = assign_phase(self.sim, self, kind="six")
            return self.data[field]
        else:
            raise KeyError("{} is not available".format(field))

    def __setitem__(self, field, value):
        self.data[field] = value

    def recenter(self, fields=None, x0=0, y0=0, z0=0):
        self.recentered = True
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0

        if fields is None:
            fields = self.keys()
        for k in fields:
            d = self.data[k]
            if self.shear:
                d = shear_periodic_roll(
                    d,
                    xshift=-x0,
                    yshift=-y0,
                    zshift=-z0,
                    qOmL=self.qOmL,
                    time=self.time,
                    vy=k in ["velocity2", "vy"],
                )
            else:
                d = periodic_roll(d, xshift=-x0, yshift=-y0, zshift=-z0)
            self.data[k] = d

    def roll(self, xshift=0, yshift=0, zshift=0):
        self.recenter(x0=-xshift, y0=-yshift, z0=-zshift)


def periodic_roll(data, xshift=0, yshift=0, zshift=0):
    ndim = len(data.dims)
    dx = float(data.x[1] - data.x[0])
    dy = float(data.y[1] - data.y[0])
    if ndim == 2:
        newdata = shift(data, (yshift / dy, xshift / dx), mode="grid-wrap", order=1)
    elif ndim == 3:
        dz = float(data.z[1] - data.z[0])
        newdata = shift(data, (0, yshift / dy, xshift / dx), mode="grid-wrap", order=1)
        newdata = shift(newdata, (zshift / dz, 0, 0), mode="constant", cval=0)
    return xr.DataArray(newdata, data.coords, data.dims, name=data.name)


def shear_periodic_roll(data, xshift=0, yshift=0, zshift=0, qOmL=0, time=0, vy=False):
    ndim = len(data.dims)
    dx = float(data.x[1] - data.x[0])
    dy = float(data.y[1] - data.y[0])
    if ndim == 3:
        dz = float(data.z[1] - data.z[0])

    ishift = int(xshift / dx)
    # iresidual = xshift/dx - ishift

    data_shifted = np.zeros_like(data.data)
    # shape=data.shape
    Nx = data.shape[-1]
    # Ny = data.shape[-2]

    isign = ishift / abs(ishift) if ishift != 0 else 1
    qOmLt = qOmL * time
    vyshear = qOmL * isign if vy else 0

    if ndim == 2:
        yshear = (qOmLt / dy * isign, 0)
    elif ndim == 3:
        yshear = (0, qOmLt / dy * isign, 0)
    imin = max(0, abs(ishift))
    imax = min(Nx, Nx - abs(ishift))
    if ishift > 0:
        data_shifted[..., imin:] = data[..., :imax]
        data_shifted[..., :imin] = shift(
            data[..., imax:] + vyshear, yshear, mode="grid-wrap", order=1
        )
    else:
        data_shifted[..., :imax] = data[..., imin:]
        data_shifted[..., imax:] = shift(
            data[..., :imin] + vyshear, yshear, mode="grid-wrap", order=1
        )
    if ndim == 2:
        data_shifted = shift(data_shifted, (yshift / dy, 0), mode="grid-wrap", order=1)
    elif ndim == 3:
        data_shifted = shift(
            data_shifted, (zshift / dz, yshift / dy, 0), mode="grid-wrap", order=1
        )

    return xr.DataArray(data_shifted, data.coords, data.dims, name=data.name)