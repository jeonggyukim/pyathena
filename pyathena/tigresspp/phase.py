import matplotlib as mpl
import numpy as np
import xarray as xr
import sys
import os
import matplotlib.pyplot as plt
import cmasher as cmr
import pyathena as pa
import astropy.units as au
import tqdm

sys.path.insert(0, "../../")


def print_blue(txt):
    print("\033[34m", txt, "\033[0m", sep="", end=" ")


def print_red(txt):
    print("\033[31m", txt, "\033[0m", sep="", end=" ")


# recalculate PDFs with finer bins
def recal_nP(dchunk, xf="nH", yf="pok", NCR=True, Nx=601, Ny=501):
    dset = xr.Dataset()
    alist = [None]
    if NCR:
        alist += ["xHI", "xHII", "neu", "pion", "ion"]
    wlist = ["vol", "nH"]
    if NCR:
        wlist += ["net_cool_rate"]
    for xs in alist:
        for wf in wlist:
            if xf == "nH":
                xbins = np.logspace(-8, 6, Nx)
            elif xf == "T":
                xbins = np.logspace(0, 8, Nx)
            ybins = np.logspace(-4, 10, Ny)
            if xs is None:
                cond = dchunk["nH"] > 0.0
            elif xs == "xHI":
                cond = dchunk["xHII"] < 0.5
            elif xs == "xHII":
                cond = dchunk["xHII"] > 0.5
            elif xs == "neu":
                cond = dchunk["xe"] < 0.1
            elif xs == "pion":
                cond = (dchunk["xe"] > 0.1) & (dchunk["xe"] < 0.9)
            elif xs == "ion":
                cond = dchunk["xe"] > 0.9

            x = (
                dchunk[xf]
                .where(cond)
                # .stack(xyz=["x", "y", "z"])
                # .dropna(dim="xyz")
                .data.flatten()
            )
            y = dchunk[yf].where(cond).data.flatten()#stack(xyz=["x", "y", "z"]).dropna(dim="xyz").data
            if wf == "vol":
                w = None
            else:
                w = (
                    dchunk[wf]
                    .where(cond)
                    # .stack(xyz=["x", "y", "z"])
                    # .dropna(dim="xyz")
                    .data.flatten()
                )
            h, b1, b2 = np.histogram2d(x, y, weights=w, bins=[xbins, ybins])
            dx = np.log10(b1[1] / b1[0])
            dy = np.log10(b2[1] / b2[0])
            pdf = h.T / dx / dy
            xbins = np.log10(xbins)
            ybins = np.log10(ybins)

            if wf == "net_cool_rate":
                total = dchunk["cool_rate"].sum().data
            elif wf == "vol":
                total = np.prod(dchunk["nH"].shape)
            else:
                total = dchunk[wf].sum().data
            da = xr.DataArray(
                pdf,
                coords=[0.5 * (ybins[1:] + ybins[:-1]), 0.5 * (xbins[1:] + xbins[:-1])],
                dims=[yf, "n_H" if xf == "nH" else xf],
            )
            dset["{}-{}".format("all" if xs is None else xs, wf)] = da
            dset = dset.assign_coords({wf: total})
    return dset.assign_coords(time=dchunk.time)


def recal_xT(dchunk):
    dset = xr.Dataset()
    T = ["T", "T1"]
    xs = ["xHI", "xHII", "xe"]
    log = [False, True]
    for T_ in T:
        try:
            dchunk[T_]
        except KeyError:
            continue
        for log_ in log:
            for xs_ in xs:
                if log_:
                    label = f"{T_}-log{xs_}"
                else:
                    label = f"{T_}-{xs_}"
                x, y, w = (
                    np.log10(dchunk[T_].data.flatten()),
                    dchunk[xs_].data.flatten(),
                    dchunk["nH"].data.flatten(),
                )
                yr = [0, 1.3]
                if log_:
                    y = np.log10(y)
                    yr = [-6, 0.3]

                h, b1, b2 = np.histogram2d(
                    x, y, weights=w, range=[[1, 8], yr], bins=[300, 150]
                )
                dx = b1[1] - b1[0]
                dy = b2[1] - b2[0]
                pdf = h.T / dx / dy
                da = xr.DataArray(
                    pdf,
                    coords=[0.5 * (b2[1:] + b2[:-1]), 0.5 * (b1[1:] + b1[:-1])],
                    dims=["logx" if log_ else "x", "T"],
                )
                dset[label] = da
    dset = dset.assign_coords({"nH": dchunk["nH"].data.sum()})
    return dset.assign_coords(time=dchunk.time)


def add_phase_cuts(
    Tlist=[500, 6000, 15000, 35000, 5.0e5],
    xs="xHI",
    xs_axis="y",
    T1=False,
    xHIIcut=0.5,
    xH2cut=0.25,
    xHI_CIE=True,
    log=False,
    lkwargs=dict(color="b", ls="--", lw=1),
    color_cie="r",
):
    phcolors = get_phcolor_dict(
        cmap=None if T1 else cmr.pride,
        T1=T1,
        cmin=0.1 if T1 else 0.0,
        cmax=0.9 if T1 else 0.85,
    )
    # deviding lines
    for T0 in Tlist:
        ymin = 0
        ymax = 1
        if (T0 == Tlist[0]) and (not T1):
            ymin = 0.5
            if log:
                ymin = (6 + np.log10(ymin)) / 6.0
        else:
            ymin = 0

        if (T0 == Tlist[2]) and (not T1):
            ymax = 0.5
            if log:
                ymax = (6 + np.log10(ymax)) / 6.0
        else:
            ymax = 1

        if xs == "xHI":
            if xs_axis == "y":
                plt.axvline(np.log10(T0), ymin=ymin, ymax=ymax, **lkwargs)
            elif xs_axis == "x":
                plt.axhline(np.log10(T0), xmax=ymax, **lkwargs)
        elif xs == "xHII":
            if xs_axis == "y":
                plt.axvline(np.log10(T0), ymin=1 - ymax, ymax=1 - ymin, **lkwargs)
            elif xs_axis == "x":
                plt.axhline(np.log10(T0), xmin=ymin, **lkwargs)
    # abundance cuts
    # xHIIcut = 0.5
    # xH2cut = 0.25
    if T1:
        pass
    else:
        if xs == "xHI":
            ion_cut = 1 - np.array([xHIIcut, xHIIcut])
            mol_cut = 1 - 2.0 * np.array([xH2cut, xH2cut])
            if log:
                ion_cut = np.log10(ion_cut)
                mol_cut = np.log10(mol_cut)
            if xs_axis == "y":
                plt.plot([np.log10(Tlist[1]), np.log10(Tlist[3])], ion_cut, **lkwargs)
                plt.plot([1, np.log10(Tlist[1])], mol_cut, **lkwargs)
            elif xs_axis == "x":
                plt.plot(ion_cut, [np.log10(Tlist[1]), np.log10(Tlist[3])], **lkwargs)
                plt.plot(mol_cut, [1, np.log10(Tlist[1])], **lkwargs)
        elif xs == "xHII":
            ion_cut = np.array([xHIIcut, xHIIcut])
            if log:
                ion_cut = np.log10(ion_cut)
            if xs_axis == "y":
                plt.plot([0, np.log10(Tlist[3])], ion_cut, **lkwargs)
            elif xs_axis == "x":
                plt.plot(ion_cut, [0, np.log10(Tlist[3])], **lkwargs)
        elif xs == "xH2":
            mol_cut = np.array([xH2cut, xH2cut])
            if log:
                mol_cut = np.log10(mol_cut)
            if xs_axis == "y":
                plt.plot([0, np.log10(Tlist[3])], mol_cut, **lkwargs)
            elif xs_axis == "x":
                plt.plot(mol_cut, [0, np.log10(Tlist[3])], **lkwargs)

    # annotate
    yl = 0.05
    yr = 0.95
    if log:
        yl = -2.95
        yr = np.log10(yr)
    if xs == "xHI":
        label_infos = dict()
        label_infos["HIM"] = dict(
            xy=(np.log10(1.1 * Tlist[4]), yr), va="top", ha="left"
        )
        label_infos["WHIM"] = dict(
            xy=(np.log10(0.9 * Tlist[4]), yr), va="top", ha="right"
        )
        label_infos["WCIM"] = dict(
            xy=(np.log10(0.9 * Tlist[3]), yl), va="bottom", ha="right", rotation=90
        )
        label_infos["WPIM"] = dict(
            xy=(np.log10(1.1 * Tlist[1]), yl), va="bottom", ha="left", rotation=90
        )
        label_infos["UIM"] = dict(
            xy=(np.log10(0.9 * Tlist[1]), yl), va="bottom", ha="right"
        )
        if not log:
            label_infos["WNM"] = dict(
                xy=(np.log10(0.9 * Tlist[3]), yr), va="top", ha="right", rotation=90
            )
        label_infos["UNM"] = dict(
            xy=(np.log10(0.9 * Tlist[1]), yr), va="top", ha="right"
        )
        label_infos["CNM"] = dict(xy=(np.log10(20), yr), va="top", ha="left")
        label_infos["CMM"] = dict(
            xy=(np.log10(20), yl), va="bottom", ha="left", label="CMM"
        )
        if T1:
            if not log:
                label_infos.pop("WNM")
            label_infos.pop("CMM")
            label_infos.pop("UIM")
            label_infos["WNM"] = label_infos.pop("WPIM")
            label_infos["WIM"] = label_infos.pop("WCIM")

    elif xs == "xHII":
        label_infos = dict()
        label_infos["HIM"] = dict(
            xy=(np.log10(1.1 * Tlist[4]), yl), va="bottom", ha="left"
        )
        label_infos["WHIM"] = dict(
            xy=(np.log10(0.9 * Tlist[4]), yl), va="bottom", ha="right"
        )
        if not log:
            label_infos["WCIM"] = dict(
                xy=(np.log10(0.9 * Tlist[3]), yr), va="top", ha="right", rotation=90
            )
            label_infos["WPIM"] = dict(
                xy=(np.log10(1.1 * Tlist[1]), yr), va="top", ha="left", rotation=90
            )
        label_infos["UIM"] = dict(
            xy=(np.log10(0.9 * Tlist[1]), yr), va="top", ha="right"
        )
        label_infos["WNM"] = dict(
            xy=(np.log10(0.9 * Tlist[3]), yl), va="bottom", ha="right", rotation=90
        )
        label_infos["UNM"] = dict(
            xy=(np.log10(0.9 * Tlist[1]), yl), va="bottom", ha="right"
        )
        label_infos["CNM"] = dict(
            xy=(np.log10(0.9 * Tlist[0]), yl), va="bottom", ha="right", label="CM+NM"
        )
        if T1:
            if not log:
                label_infos.pop("WNM")
                label_infos["WNM"] = label_infos.pop("WPIM")
                label_infos["WIM"] = label_infos.pop("WCIM")
            else:
                label_infos["WIM"] = dict(
                    xy=(np.log10(0.9 * Tlist[3]), yl),
                    va="bottom",
                    ha="right",
                    rotation=90,
                )
                label_infos["WNM"] = dict(
                    xy=(np.log10(1.1 * Tlist[1]), yl),
                    va="bottom",
                    ha="left",
                    rotation=90,
                )
            label_infos.pop("UIM")

    for k in label_infos:
        info = label_infos[k]
        if "label" in info:
            label = info.pop("label")
        else:
            label = k
        if xs_axis == "x":
            info["xy"] = (info["xy"][1], info["xy"][0])
            va = "bottom" if info["ha"] == "left" else "top"
            ha = "left" if info["va"] == "bottom" else "right"
            info["va"] = va
            info["ha"] = ha
            info["rotation"] = 0

        fg = "k" if k in ["WNM", "WIM", "UIM", "WCIM", "WPIM"] else "w"
        te = pa.classic.texteffect(fontsize="xx-small", foreground=fg, linewidth=2)
        plt.annotate(label, color=phcolors[k], weight="bold", **info, **te)

    #
    if xHI_CIE:
        T = np.logspace(3, 6, 100)
        kcoll = pa.microphysics.cool.coeff_kcoll_H(T)
        krec = pa.microphysics.cool.coeff_alpha_rr_H(T)
        xHI = krec / (kcoll + krec)
        if T1:
            logT = np.log10(T * (2.1 - xHI))
        else:
            logT = np.log10(T)
        if xs == "xHII":
            xHI = 1 - xHI
        if log:
            xHI = np.log10(xHI)
        if xs_axis == "y":
            plt.plot(logT, xHI, c=color_cie, ls="--", label="CIE", lw=2, dashes=[6, 2])
        elif xs_axis == "x":
            plt.plot(xHI, logT, c=color_cie, ls="--", label="CIE", lw=2, dashes=[6, 2])


def get_dchunk(s, num, scratch_dir="/scratch/gpfs/changgoo/TIGRESS-NCR/"):
    scratch_dir += os.path.join(s.basename, "midplane_chunk")
    chunk_file = os.path.join(scratch_dir, "{:s}.{:04d}.nc".format(s.problem_id, num))
    if not os.path.isfile(chunk_file):
        raise IOError("File does not exist: {}".format(chunk_file))
    with xr.open_dataset(chunk_file) as chunk:
        chunk["Uion"] = chunk["Erad_LyC"] / (
            (s.par["radps"]["hnu_PH"] * au.eV).cgs.value * chunk["nH"]
        )
        chunk["xHII"] = 1 - chunk["xHI"] - chunk["xH2"] * 2
        chunk["T1"] = chunk["pok"] / chunk["nH"]
    return chunk, scratch_dir


def define_phase(s, kind="full", verbose=False):
    """Phase definition

    kind : str
      'full' for full 9 phases
      'four' for CU, WNM, WIM, Hot
      'five1' for CNM, UNM, WNM, WIM, Hot
      'five2' for CU, WNM, WIM, WHIM, HIM
      'six' for CNM, UNM, WNM, WIM, WHIM, HIM
      'classic' for CNM, UNM, WNM, WHIM, HIM
    """

    pcdict = get_phcolor_dict(cmap=cmr.pride, cmin=0.0, cmax=0.85)
    phdef = []
    Tlist = s.get_phase_Tlist(kind="classic" if kind == "classic" else "ncr")
    Tlist = list(np.concatenate([[0], Tlist, [np.inf]]))
    i = 1
    if kind in ["four", "five2"]:
        # cold+unstable
        phdef.append(
            dict(
                idx=i,
                name="CU",
                Tmin=Tlist[0],
                Tmax=Tlist[2],
                abundance=None,
                amin=0.0,
                c=pcdict["CNM"],
            )
        )
    elif kind == "full":
        # cold molecular (xH2>0.25, T<6000)
        phdef.append(
            dict(
                idx=i,
                name="CMM",
                Tmin=Tlist[0],
                Tmax=Tlist[2],
                abundance="xH2",
                amin=0.25,
                c=pcdict["CMM"],
            )
        )
        # cold neutral (xHI>0.5, T<500)
        i += 1
        phdef.append(
            dict(
                idx=i,
                name="CNM",
                Tmin=Tlist[0],
                Tmax=Tlist[1],
                abundance="xHI",
                amin=0.5,
                c=pcdict["CNM"],
            )
        )
        # unstable neutral (xHI>0.5, 500<T<6000)
        i += 1
        phdef.append(
            dict(
                idx=i,
                name="UNM",
                Tmin=Tlist[1],
                Tmax=Tlist[2],
                abundance="xHI",
                amin=0.5,
                c=pcdict["UNM"],
            )
        )
        # unstable ionized (xHII>0.5, T<6000)
        i += 1
        phdef.append(
            dict(
                idx=i,
                name="UIM",
                Tmin=Tlist[0],
                Tmax=Tlist[2],
                abundance="xHII",
                amin=0.5,
                c=pcdict["UIM"],
            )
        )
    else:
        # cold
        phdef.append(
            dict(
                idx=i,
                name="CNM",
                Tmin=Tlist[0],
                Tmax=Tlist[1],
                abundance=None,
                amin=0.0,
                c=pcdict["CNM"],
            )
        )
        # Unstable
        i += 1
        phdef.append(
            dict(
                idx=i,
                name="UNM",
                Tmin=Tlist[1],
                Tmax=Tlist[2],
                abundance="xHI",
                amin=0.5,
                c=pcdict["UNM"],
            )
        )

    # warm neutral (xHI>0.5, 6000<T<35000)
    if kind == "classic":
        i += 1
        phdef.append(
            dict(
                idx=i,
                name="WNM",
                Tmin=Tlist[2],
                Tmax=Tlist[4],
                abundance=None,
                amin=0.0,
                c=pcdict["WNM"],
            )
        )
    else:
        i += 1
        phdef.append(
            dict(
                idx=i,
                name="WNM",
                Tmin=Tlist[2],
                Tmax=Tlist[4],
                abundance="xHI",
                amin=0.5,
                c=pcdict["WNM"],
            )
        )

    if kind == "full":
        # warm photo-ionized (xHII>0.5, 6000<T<15000)
        i += 1
        phdef.append(
            dict(
                idx=i,
                name="WPIM",
                Tmin=Tlist[2],
                Tmax=Tlist[3],
                abundance="xHII",
                amin=0.5,
                c=pcdict["WPIM"],
            )
        )
        # warm collisonally-ionized (xHII>0.5, 15000<T<35000)
        i += 1
        phdef.append(
            dict(
                idx=i,
                name="WCIM",
                Tmin=Tlist[3],
                Tmax=Tlist[4],
                abundance="xHII",
                amin=0.5,
                c=pcdict["WCIM"],
            )
        )
    elif kind != "classic":
        # combined warm ionzied
        i += 1
        phdef.append(
            dict(
                idx=i,
                name="WIM",
                Tmin=Tlist[1],
                Tmax=Tlist[4],
                abundance="xHII",
                amin=0.5,
                c=pcdict["WPIM"],
            )
        )

    if kind in ["four", "five1"]:
        # hot ionzied
        i += 1
        phdef.append(
            dict(
                idx=i,
                name="HIM",
                Tmin=Tlist[4],
                Tmax=Tlist[6],
                abundance=None,
                amin=0.0,
                c=pcdict["HIM"],
            )
        )
    else:
        # warm-hot ionized (35000<T<5.e5)
        i += 1
        phdef.append(
            dict(
                idx=i,
                name="WHIM",
                Tmin=Tlist[4],
                Tmax=Tlist[5],
                abundance=None,
                amin=0.0,
                c=pcdict["WHIM"],
            )
        )
        # hot ionized (5.e5<T)
        i += 1
        phdef.append(
            dict(
                idx=i,
                name="HIM",
                Tmin=Tlist[5],
                Tmax=Tlist[6],
                abundance=None,
                amin=0.0,
                c=pcdict["HIM"],
            )
        )

    if verbose:
        for ph in phdef:
            T1 = ph["Tmin"]
            T2 = ph["Tmax"]
            i = ph["idx"]
            a = ph["abundance"]
            amin = ph["amin"]
            print(
                "{:5s}".format(ph["name"]),
                "{:3d}".format(i),
                "{:>10}".format(T1),
                "<T<",
                "{:10}".format("{}".format(T2)),
                "{:^10s}".format(
                    "{:4s}>{}".format(a, amin) if a is not None else "..."
                ),
            )
    return phdef


def assign_phase(s, dslc, kind="full", verbose=False):
    """Assign phase to the data chunk

    Parameters
    ==========
    s : object, LoadSimTIGRESSNCR
    dslc : xarray.Dataset
        must have nH, T, xHI, xHII, xH2
    kind : str
        phase definition type
        'full' for full 9 phases
        'four' for CU, WNM, WIM, Hot
        'five1' for CNM, UNM, WNM, WIM, Hot
        'five2' for CU, WNM, WIM, WHIM, HIM
        'six' for CNM, UNM, WNM, WIM, WHIM, HIM
    """
    from matplotlib.colors import ListedColormap

    phslc = xr.zeros_like(dslc["nH"]).astype("int") - 1
    phslc.name = "phase"

    phdef = define_phase(s, kind=kind)
    phlist = []
    phcmap = []
    for ph in phdef:
        T1 = ph["Tmin"]
        T2 = ph["Tmax"]
        i = ph["idx"]
        a = ph["abundance"]
        amin = ph["amin"]
        cond = (dslc["T"] > T1) * (dslc["T"] <= T2)
        phlist.append(ph["name"])
        phcmap.append(ph["c"])
        if verbose:
            print(ph["name"], i, T1, T2, a, amin)
        if (a is not None) & (s.test_newcool()):
            cond *= dslc[a] > amin
        phslc += cond * i
    phslc.attrs["phdef"] = phdef
    phslc.attrs["phlist"] = phlist
    phslc.attrs["phcmap"] = ListedColormap(phcmap)

    return phslc


def get_phcmap(T1=False, cmin=0, cmax=1, cmap=None):
    if T1:
        phlist = ["CNM", "UNM", "WNM", "WIM", "WHIM", "HIM"]
    else:
        phlist = ["CMM", "CNM", "UNM", "UIM", "WNM", "WPIM", "WCIM", "WHIM", "HIM"]
    if cmap is None:
        from pyathena.plt_tools import cmap

        cmap = cmap.get_cmap_jh_colors()
    nph = len(phlist)
    phcmap = cmr.get_sub_cmap(cmap, cmin, cmax, N=nph)

    return phlist, phcmap


def get_phcolor_dict(cmap=None, T1=False, cmin=0, cmax=1):
    phlist, phcmap = get_phcmap(cmap=cmap, T1=T1, cmin=cmin, cmax=cmax)
    phcolors = dict()
    for phname, c in zip(phlist, phcmap.colors):
        phcolors[phname] = c
    return phcolors


def draw_phase(ph):
    phlist, phcmap = get_phcmap()
    if "phlist" in ph.attrs:
        phlist = ph.attrs["phlist"]
    if "phcmap" in ph.attrs:
        phcmap = ph.attrs["phcmap"]
    nph = len(phlist)
    image_style = {
        "axes.grid": False,
        "image.interpolation": "nearest",
        "ytick.minor.visible": False,
    }
    with mpl.rc_context(image_style):
        ph.plot(cmap=phcmap, vmin=0, vmax=nph, cbar_kwargs=dict(extend=None))
        axes = plt.gcf().axes
        axes[1].get_yaxis().set_ticks([])
        for j, lab in enumerate(phlist):
            color = "w" if lab in ["CMM", "CNM", "HIM"] else "k"
            axes[1].annotate(
                lab,
                (0.5, (2 * j + 1) / (2 * nph)),
                xycoords="axes fraction",
                color=color,
                ha="center",
                va="center",
                rotation=90,
                fontsize="x-small",
            )
        axes[1].set_ylabel("Phase")
        axes[0].set_aspect("equal")


class PDF1D:
    def __init__(self, s, scrbase="/scratch/gpfs/changgoo/TIGRESS-NCR/"):
        self.scrbase = scrbase
        self.phdef = define_phase(s, kind="full")
        self.sim = s
        self.basedir = s.basedir

    @staticmethod
    def Tpdf(data, tf="T", wf=None, Tbin=0.01, Tmin=0, Tmax=9, normed=True):
        Tbins = 10.0 ** np.arange(Tmin, Tmax + Tbin, Tbin)
        h, b = np.histogram(
            data[tf].data.flatten(),
            bins=Tbins,
            weights=data[wf].data.flatten() if wf in data else None,
        )
        if normed:
            h = h / h.sum() / Tbin
        else:
            h = h / Tbin

        return h, b

    def recal_1Dpdfs(self, num, force_override=False, to_scratch=False):
        s = self.sim
        if to_scratch:
            out_dir = os.path.join(self.scrbase, s.basename)
        else:
            out_dir = s.basedir

        # check files first
        all_exist = True
        for tf in ["T", "T1", "pok", "nH"]:
            pdfdir = os.path.join(out_dir, "1D-pdfs", "{}-pdf".format(tf))
            os.makedirs(pdfdir, exist_ok=True)
            fname = os.path.join(pdfdir, "{}.{:04d}.nc".format(s.problem_id, num))
            if os.path.isfile(fname):
                continue
            all_exist = False
        if not force_override:
            if all_exist:
                return True
        flist = ["nH", "pok", "T"]
        wflist = ["vol", "nH"]
        if s.test_newcool():
            flist += [
                "xe",
                "xHI",
                "xHII",
                "xH2",
                "cool_rate",
                "heat_rate",
                "net_cool_rate",
            ]
            wflist += [
                "ne",
                "nHI",
                "nHII",
                "nH2",
                "net_cool_rate",
                "cool_rate",
                "heat_rate",
                "netcool",
            ]
            phkind = "full"
        else:
            phkind = "classic"
        dchunk = s.get_data(num,fields=flist)

        dchunk["T1"] = dchunk["pok"] / dchunk["nH"] / self.sim.u.muH
        if s.test_newcool():
            dchunk["ne"] = dchunk["nH"] * dchunk["xe"]
            dchunk["nH2"] = dchunk["nH"] * dchunk["xH2"] * 2
            dchunk["nHI"] = dchunk["nH"] * dchunk["xHI"]
            dchunk["nHII"] = dchunk["nH"] * dchunk["xHII"]
            dchunk["netcool"] = dchunk["cool_rate"] - dchunk["heat_rate"]
        dchunk = dchunk.sel(z=slice(-300, 300))

        ph = assign_phase(s, dchunk, kind=phkind, verbose=False)
        bin_params = dict(
            nH=(-6, 4, 0.05), T=(0, 9, 0.05), T1=(0, 9, 0.05), pok=(0, 7, 0.05)
        )

        for tf in ["T", "T1", "pok", "nH"]:
            pdfdir = os.path.join(out_dir, "1D-pdfs", "{}-pdf".format(tf))
            os.makedirs(pdfdir, exist_ok=True)
            fname = os.path.join(pdfdir, "{}.{:04d}.nc".format(s.problem_id, num))
            if not force_override:
                if os.path.isfile(fname):
                    continue

            Tmin, Tmax, Tbin = bin_params[tf]
            pdf = xr.Dataset()
            for wf in wflist:
                ph_pdf = xr.Dataset()
                for i, phinfo in enumerate(ph.attrs["phdef"]):
                    # c = phinfo["c"]
                    phname = phinfo["name"]
                    h, b = self.Tpdf(
                        dchunk.where(ph == i),
                        wf=wf,
                        tf=tf,
                        Tmin=Tmin,
                        Tmax=Tmax,
                        Tbin=Tbin,
                        normed=False,
                    )
                    ph_pdf[phname] = xr.DataArray(h, coords=[b[:-1]], dims=[tf])
                pdf[wf if wf != "nH" else "mass"] = ph_pdf.to_array("phase")
            try:
                pdf.to_array("weight").to_netcdf(fname)
            except PermissionError:
                if force_override:
                    os.remove(fname)
                    pdf.to_array("weight").to_netcdf(fname)
                else:
                    pdf.close()
                    return False
            pdf.close()
        return True

    def recal_all(self, nmin=200, force_override=False):
        import gc

        s = self.sim
        passed = []
        created = []
        for num in tqdm.tqdm(s.nums):
            if num < nmin:
                continue
            if self.recal_1Dpdfs(num, force_override=force_override):
                created.append(num)
            else:
                passed.append(num)
            gc.collect()

        for num in s.nums:
            if num in created:
                print_blue(num)
            else:
                print_red(num)

    def load_1Dpdf(self, s, num, tf, from_scratch=False):
        if from_scratch:
            out_dir = os.path.join(self.scrbase, s.basename)
        else:
            out_dir = s.basedir

        pdfdir = os.path.join(out_dir, "1D-pdfs", "{}-pdf".format(tf))
        fname = os.path.join(pdfdir, "{}.{:04d}.nc".format(s.problem_id, num))
        if os.path.isfile(fname):
            with xr.open_dataarray(fname) as dset:
                return dset.to_dataset("weight")
        else:
            raise OSError

    def loader(self, s, tf, save=False, read_from_file=True):
        out_dir = s.basedir
        pdfdir = os.path.join(out_dir, "1D-pdfs", "{}-pdf".format(tf))
        fname = os.path.join(pdfdir, "{}.nc".format(s.problem_id))
        if read_from_file:
            try:
                with xr.open_dataset(fname) as dset:
                    setattr(self, tf, dset)
                    return
            except OSError:
                pass

        dt = s.par["output2"]["dt"]
        pdflist = []
        for num in tqdm.tqdm(s.nums):
            try:
                pdflist.append(
                    self.load_1Dpdf(s, num, tf).assign_coords(time=num * dt * s.u.Myr)
                )
            except OSError:
                pass
        setattr(self, tf, xr.concat(pdflist, dim="time"))

        if save:
            getattr(self, tf).to_netcdf(fname)

    def load_all(self, save=True, read_from_file=True):
        for tf in ["T", "T1", "pok", "nH"]:
            self.loader(self.sim, tf, save=save, read_from_file=read_from_file)


if __name__ == "__main__":
    s = pa.LoadSimTIGRESSNCR(
        "/tigress/changgoo/TIGRESS-NCR/R8_4pc_NCR.full/", verbose=True
    )

    for num in s.nums:
        try:
            dchunk, scratch_dir = get_dchunk(s, num)
        except IOError:
            continue

        phase_dir = scratch_dir.replace("midplane_chunk", "phase")
        if not os.path.isdir(phase_dir):
            os.makedirs(phase_dir)

        phase_file = os.path.join(
            phase_dir, "{:s}.{:04d}.phase.nc".format(s.basename, num)
        )

        ph = define_phase(s, dchunk)

        ph.to_netcdf(phase_file)
        ph.close()
        print(phase_file)
