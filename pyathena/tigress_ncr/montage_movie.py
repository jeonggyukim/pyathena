import pyathena as pa
import matplotlib.pyplot as plt
from pyathena.plt_tools.make_movie import make_movie
import glob, os, sys
from shutil import copyfile
import os.path as osp
import argparse
import numpy as np

def get_time_from_zprof(s, inum):
    zpf = glob.glob(os.path.join(s.basedir, "zprof/", f"*.{inum:04d}.whole.zprof"))
    with open(zpf[0], "r") as fp:
        l = fp.readline()
        time = float(l[l.rfind("=") + 1 :])
    return time


def stitch_number_densities(s, i, tf="layer", ncrsp=False):
    fields = [
        "H_nuclei_density",
        "H_p0_number_density",
        "H_p1_number_density",
        "H2_number_density",
    ]
    field_names = ["total", "atomic (H$^0$)", "ionized (H$^+$)", "molecular (H$_2$)"]
    fig_ysize = 10
    if ncrsp:
        main_x1 = 0.25
        main_x2 = 1.3
        sub_x1 = 0.25
        sub_x2 = 0.95
    else:
        Nx, Ny, Nz = s.domain['Nx']
        if Nz/Nx == 6:
            main_x1 = 0.35
            main_x2 = 1
            sub_x1 = 0.35
            sub_x2 = 0.65
        elif Nz/Nx == 3:
            main_x1 = 0.25
            main_x2 = 1
            sub_x1 = 0.25
            sub_x2 = 0.75
        else:
            print(f"Layout for Nz/Nx = {Nz/Nx} is not specified")
    main_dx = main_x2 - main_x1
    sub_dx = sub_x2 - sub_x1
    Nsub = 3
    fig_xsize = fig_ysize*main_dx + fig_ysize*Nsub*sub_dx
    wratios = [main_dx] + [sub_dx]*Nsub
    with plt.style.context("dark_background", {"figure.dpi": 300}):
        fig, axes = plt.subplots(
            1,
            4,
            figsize=(fig_xsize,fig_ysize),
            gridspec_kw=dict(wspace=0, hspace=0, width_ratios=wratios),
        )

        for field, name, ax in zip(fields, field_names, axes.flatten()):
            if tf == "linramp":
                head = f"{field}_linramp"
            else:
                head = field
            files = sorted(
                glob.glob(os.path.join(s.basedir, "volume", f"{head}_time_????.png"))
            )
            inum = int(files[i].split("_")[-1].split(".")[0])
            time = get_time_from_zprof(s, inum)
            plt.sca(ax)
            img = plt.imread(files[i])
            Nx, Ny, Nch = img.shape
            plt.imshow(img)
            plt.annotate(
                name,
                (0.5, 0.0),
                ha="center",
                va="top",
                xycoords="axes fraction",
                weight="bold",
                fontsize="xx-large"
            )
            if name == "total":
                plt.xlim(Nx * main_x1, Nx * main_x2)
            else:
                plt.xlim(Nx * sub_x1, Nx * sub_x2)
            plt.axis("off")
        axes[0].annotate(
            f"t={time*s.u.Myr:5.1f}Myr",
            (0.1, 1.0),
            va="top",
            ha="left",
            xycoords="axes fraction",
            weight="bold",
            fontsize="xx-large",
        )

        fig.savefig(
            os.path.join(
                s.basedir, "volume", f"number_density_{tf}_time_{inum:04d}.png"
            ),
            bbox_inches="tight",
        )


def stitch_observables(s, i, tf="layer", ncrsp=False):
    fields = ["CII_luminosity",
              "HI_21cm_luminosity",
              "H_alpha_luminosity",
              "xray_luminosity_0.5_7.0_keV"]
    field_names = ["CII",
                   "HI 21cm",
                   r"${\bf H_\alpha}$",
                   "X-ray (0.5-7.0 keV)"]
    fig_ysize = 10
    if ncrsp:
        main_x1 = 0.25
        main_x2 = 0.95
        cbar_x1 = 1.05
        cbar_x2 = 1.3
    else:
        Nx, Ny, Nz = s.domain['Nx']
        if Nz/Nx == 6:
            main_x1 = 0.35
            main_x2 = 0.65
            cbar_x1 = 0.75
            cbar_x2 = 1
        elif Nz/Nx == 3:
            main_x1 = 0.25
            main_x2 = 0.75
            cbar_x1 = 0.75
            cbar_x2 = 1
        else:
            print(f"Layout for Nz/Nx = {Nz/Nx} is not specified")

    main_dx = main_x2 - main_x1
    cbar_dx = cbar_x2 - cbar_x1
    Nfield = len(fields)
    fig_xsize = fig_ysize*(main_dx+cbar_dx)*Nfield
    wratios = [main_dx,cbar_dx]*Nfield
    with plt.style.context("dark_background", {"figure.dpi": 200}):
        fig, axes = plt.subplots(
            1, Nfield*2, figsize=(fig_xsize, fig_ysize),
            gridspec_kw=dict(wspace=0, hspace=0, width_ratios=wratios)
        )

        for field, name, ax in zip(fields, field_names, axes.reshape(Nfield,2)):
            if tf == "linramp":
                head = f"{field}_linramp"
            else:
                head = field
            files = sorted(
                glob.glob(os.path.join(s.basedir, "volume", f"{head}_time_????.png"))
            )
            inum = int(files[i].split("_")[-1].split(".")[0])
            time = get_time_from_zprof(s, inum)

            img = plt.imread(files[i])
            Nx, Ny, Nch = img.shape
            plt.sca(ax[0])
            plt.imshow(img)
            plt.xlim(Nx * main_x1, Nx * main_x2)
            plt.axis("off")

            plt.sca(ax[1])
            plt.imshow(img)
            # plt.imshow(np.flip(img.swapaxes(0,1),axis=1))
            plt.xlim(Nx * cbar_x1, Nx * cbar_x2)
            plt.axis("off")
        axes[0].annotate(
            f"t={time*s.u.Myr:5.1f}Myr",
            (0.1, 1.0),
            va="top",
            ha="left",
            xycoords="axes fraction",
            weight="bold",
            fontsize='xx-large'
        )
        fig.savefig(
            os.path.join(s.basedir, "volume", f"luminosity_{tf}_time_{inum:04d}.png"),
            bbox_inches="tight",
        )


def stitch_rgb(s, i, ncrsp=False):
    fields = ["luminosity_RGB",
              "luminosity_R",
              "luminosity_G",
              "luminosity_B"]
    fig_ysize = 10
    if ncrsp:
        main_x1 = 0.25
        main_x2 = 0.95
        cbar_x1 = 1.05
        cbar_x2 = 1.3
    else:
        Nx, Ny, Nz = s.domain['Nx']
        if Nz/Nx == 6:
            main_x1 = 0.35
            main_x2 = 0.65
            cbar_x1 = 0.75
            cbar_x2 = 1
        elif Nz/Nx == 3:
            main_x1 = 0.25
            main_x2 = 0.75
            cbar_x1 = 0.75
            cbar_x2 = 1
        else:
            print(f"Layout for Nz/Nx = {Nz/Nx} is not specified")
    main_dx = main_x2 - main_x1
    cbar_dx = cbar_x2 - cbar_x1
    Nfield = len(fields)
    fig_xsize = fig_ysize*main_dx+(main_dx+cbar_dx)*(Nfield-1)*fig_ysize
    wratios = [main_dx]+[main_dx,cbar_dx]*(Nfield-1)
    with plt.style.context("dark_background", {"figure.dpi": 200}):
        fig, axes = plt.subplots(
            1, 7, figsize=(fig_xsize, fig_ysize),
            gridspec_kw=dict(wspace=0, hspace=0, width_ratios=wratios)
        )

        field = fields[0]
        files = sorted(
            glob.glob(os.path.join(s.basedir, "volume", f"{field}_time_????.png"))
        )
        inum = int(files[i].split("_")[-1].split(".")[0])
        time = get_time_from_zprof(s, inum)
        ax = axes[0]
        plt.sca(ax)
        img = plt.imread(files[i])
        Nx, Ny, Nch = img.shape
        plt.imshow(img)
        plt.xlim(Nx * main_x1, Nx * main_x2)
        plt.axis("off")

        for field, ax in zip(fields[1:], axes[1:].reshape(3,2)):
            files = sorted(
                glob.glob(os.path.join(s.basedir, "volume", f"{field}_time_????.png"))
            )

            img = plt.imread(files[i])
            Nx, Ny, Nch = img.shape
            plt.sca(ax[0])
            plt.imshow(img)
            plt.xlim(Nx * main_x1, Nx * main_x2)
            plt.axis("off")
            plt.sca(ax[1])
            plt.imshow(img)
            plt.xlim(Nx * cbar_x1, Nx * cbar_x2)
            plt.axis("off")
        axes[0].annotate(
            f"t={time*s.u.Myr:5.1f}Myr",
            (0.1, 1.0),
            va="top",
            ha="left",
            xycoords="axes fraction",
            weight="bold",
            fontsize='xx-large'
        )
        fig.savefig(
            os.path.join(s.basedir, "volume", f"luminosity_RGB_channel_time_{inum:04d}.png"),
            bbox_inches="tight",
        )

def stitch_rgb2(s, i, ncrsp=True):
    from matplotlib.gridspec import GridSpec
    files = sorted(
        glob.glob(os.path.join(s.basedir, "volume", f"luminosity_RGB_time_????.png"))
    )
    inum = int(files[i].split("_")[-1].split(".")[0])
    time = get_time_from_zprof(s, inum)

    fig_ysize=12
    if ncrsp:
        main_x1 = 0.25
        main_x2 = 0.95
        sub_x1 = 0.25
        sub_x2 = 1.3
    else:
        Nx, Ny, Nz = s.domain['Nx']
        if Nz/Nx == 6:
            main_x1 = 0.35
            main_x2 = 0.65
            sub_x1 = main_x1
            sub_x2 = 1
        elif Nz/Nx == 3:
            main_x1 = 0.25
            main_x2 = 0.75
            sub_x1 = main_x1
            sub_x2 = 1
        else:
            print(f"Layout for Nz/Nx = {Nz/Nx} is not specified")

    main_dx = main_x2 - main_x1
    sub_dx = sub_x2 - sub_x1

    fig_xsize = fig_ysize*main_dx+(sub_dx)*fig_ysize/3.
    wratios = [main_dx, sub_dx/3]

    with plt.style.context("dark_background",{'figure.dpi':300}):
        fig = plt.figure(figsize=(fig_xsize,fig_ysize))
        gs = GridSpec(3,2,fig, 0,0,1,1,0.1,0,width_ratios=wratios)
        ax = fig.add_subplot(gs[:,0])
        img = plt.imread(os.path.join(s.basedir,'volume',f'luminosity_RGB_time_{inum:04d}.png'))
        Nx,Ny,Nch = img.shape
        plt.imshow(img)
        plt.xlim(Nx*main_x1,Nx*main_x2)
        plt.axis('off')
        ax.annotate(
            f"t={time*s.u.Myr:5.1f}Myr",
            (0.05, 1.0),
            va="top",
            ha="left",
            xycoords="axes fraction",
            weight="bold",
            fontsize='xx-large'
        )
        for i,ch in enumerate('RGB'):
            ax=fig.add_subplot(gs[i,1])
            img = plt.imread(os.path.join(s.basedir,'volume',f'luminosity_{ch}_time_{inum:04d}.png'))
            Nx,Ny,Nch = img.shape
            plt.imshow(img)
            plt.xlim(Nx*sub_x1,Nx*sub_x2)
            plt.axis('off')

        fig.savefig(
            os.path.join(s.basedir, "volume", f"luminosity_RGB_channel_time_{inum:04d}.png"),
            bbox_inches="tight",
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--basedir", type=str, default=None, help="Name of the basedir."
    )
    args = vars(parser.parse_args())
    locals().update(args)
    print(basedir)

    fields = [
        "H_nuclei_density",
        "H_p0_number_density",
        "H_p1_number_density",
        "H2_number_density",
        "HI_21cm_luminosity",
        "CII_luminosity",
        "H_alpha_luminosity",
        "xray_luminosity_0.5_7.0_keV",
    ]

    s = pa.LoadSimTIGRESSNCR(basedir)

    # check files
    nfiles = None
    all_identical = True
    for f in fields:
        for tf in ['_linramp','']:
            files = sorted(
                glob.glob(
                    osp.join(s.basedir, "volume", f"{f}{tf}_time_????.png")
                )
            )
            print(f'found {len(files)} images for {f}{tf}')
            if nfiles is not None:
                if nfiles != len(files):
                    all_identical = False
            else:
                nfiles = len(files)

    if not all_identical: sys.exit()

    ncrsp = s.test_spiralarm()
    # montage
    for i in range(nfiles):
        print(f"montage [{i}/{nfiles}]")
        stitch_number_densities(s, i, tf="linramp", ncrsp=ncrsp)
        stitch_number_densities(s, i, tf="layer", ncrsp=ncrsp)

        stitch_observables(s, i, tf="linramp", ncrsp=ncrsp)
        stitch_observables(s, i, tf="layer", ncrsp=ncrsp)

        # if ncrsp:
        #     stitch_rgb2(s, i, ncrsp=ncrsp)
        # else:
        stitch_rgb(s, i, ncrsp=ncrsp)
        plt.close("all")

    # movies
    for f in fields:
        fin = os.path.join(s.basedir, "volume", f"{f}_linramp_time_????.png")
        fout = osp.join(s.basedir, f"{s.basename}_{f}_linramp_time.mp4")
        make_movie(fin, fout, fps_in=15, fps_out=15)
        fin = os.path.join(s.basedir, "volume", f"{f}_time_????.png")
        fout = osp.join(s.basedir, f"{s.basename}_{f}_time.mp4")
        make_movie(fin, fout, fps_in=15, fps_out=15)
    fields = ["number_density", "luminosity"]
    for f in fields:
        for tf in ["linramp", "layer"]:
            fin = os.path.join(s.basedir, "volume", f"{f}_{tf}_time_????.png")
            fout = osp.join(s.basedir, f"{s.basename}_{f}_{tf}_time.mp4")
            make_movie(fin, fout, fps_in=15, fps_out=15)
            copyfile(
                fout,
                f"/tigress/changgoo/public_html/temporary_movies/TIGRESS-NCR_volume/{osp.basename(fout)}",
            )
    fin = os.path.join(s.basedir, "volume", f"luminosity_RGB_channel_time_????.png")
    fout = osp.join(s.basedir, f"{s.basename}_luminosity_RGB_channel_time.mp4")
    make_movie(fin, fout, fps_in=15, fps_out=15)
    copyfile(
        fout,
        f"/tigress/changgoo/public_html/temporary_movies/TIGRESS-NCR_volume/{osp.basename(fout)}",
    )
