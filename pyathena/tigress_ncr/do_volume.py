#!/usr/bin/env python

import os
import os.path as osp
import gc
import time
from mpi4py import MPI
import matplotlib.pyplot as plt
import pprint
import argparse
import sys
import numpy as np

import pyathena as pa
from pyathena.util.split_container import split_container
from pyathena.plt_tools.make_movie import make_movie

import astropy.constants as ac
import astropy.units as au
import yt
from yt.visualization.volume_rendering.api import Scene, create_volume_source

from pyathena.microphysics.cool import get_xCII, q10CII_

def add_fields(s, ds, xray=True, CII=True):
    # yt standard abundance fields
    def _nHI(field, data):
        xHI = data[("athena", "xHI")]
        return data[("gas", "H_nuclei_density")] * xHI * (xHI>0.01)

    def _nHII(field, data):
        xHII =  1 - data[("athena", "xHI")] - 2 * data[("athena", "xH2")]
        return data[("gas", "H_nuclei_density")] * xHII * (xHII>0.01)

    def _nH2(field, data):
        xH2 = data[("athena", "xH2")]
        return 2.0 * data[("gas", "H_nuclei_density")] * xH2 * (xH2>0.01)

    def _jHalpha(field, data):
        T4 = data[("athena", "temperature")].v / 1.0e4
        idx = (T4 > 0.1) * (T4 < 3)
        alpha_eff = 1.17e-13 * T4 ** (-0.942 - 0.031 * np.log(T4))
        hnu_halpha = 3.0269e-13
        return (
            alpha_eff
            * hnu_halpha
            * idx
            * data[("gas", "El_number_density")]
            * data[("gas", "H_p1_number_density")]
            * (yt.units.erg * yt.units.cm**3 / yt.units.s)
        )

    def _LHalpha(field, data):
        return data[("gas","H_alpha_emissivity")]*data[("gas", "cell_volume")]

    def _jHI21(field, data):
        C = 4*np.pi*1.6201623e-33*yt.units.erg/yt.units.s # (3/16pi)h*nu*A (KKO14)
        return C*data[("gas","H_p0_number_density")]

    def _LHI21(field, data):
        return data[("gas","HI_21cm_emissivity")]*data[("gas", "cell_volume")]

    # add/override fields
    ds.add_field(
        ("gas", "H_p0_number_density"),
        function=_nHI,
        force_override=True,
        units="cm**(-3)",
        display_name=r"$n_{\rm H^0}$",
        sampling_type="cell",
    )
    ds.add_field(
        ("gas", "H_p1_number_density"),
        function=_nHII,
        force_override=True,
        units="cm**(-3)",
        display_name=r"$n_{\rm H^+}$",
        sampling_type="cell",
    )
    ds.add_field(
        ("gas", "H2_number_density"),
        function=_nH2,
        force_override=True,
        units="cm**(-3)",
        display_name=r"$n_{\rm H_2}$",
        sampling_type="cell",
    )

    # Halpha
    ds.add_field(
        ("gas", "H_alpha_emissivity"),
        function=_jHalpha,
        force_override=True,
        units="erg/s/cm**3",
        display_name=r"$4\pi j_{\rm H\alpha}$",
        sampling_type="cell",
    )

    ds.add_field(
        ("gas", "H_alpha_luminosity"),
        function=_LHalpha,
        force_override=True,
        units="erg/s",
        display_name=r"$L_{\rm H\alpha}$",
        sampling_type="cell",
    )

    # HI_21cm
    ds.add_field(
        ("gas", "HI_21cm_emissivity"),
        function=_jHI21,
        force_override=True,
        units="erg/s/cm**3",
        display_name=r"$4\pi j_{\rm HI}$",
        sampling_type="cell",
    )

    ds.add_field(
        ("gas", "HI_21cm_luminosity"),
        function=_LHI21,
        force_override=True,
        units="erg/s",
        display_name=r"$L_{\rm HI}$",
        sampling_type="cell",
    )

    # xray
    if xray:
        import pyxsim

        emin = 0.1
        emax = 10
        nbins = 1000
        model = "spex"
        binscale = "log"
        Zmet = s.par["problem"]["Z_gas"]
        srcmdl = pyxsim.CIESourceModel(
            model, emin, emax, nbins, Zmet, binscale=binscale, abund_table="aspl"
        )
        xray_fields = srcmdl.make_source_fields(ds, 0.5, 7.0, force_override=True)

    if CII:
        # set total C, O abundance
        # xOstd = s.par["cooling"]["xOstd"]
        xCstd = s.par["cooling"]["xCstd"]
        # set metallicities
        Z_g = s.par["problem"]["Z_gas"]
        Z_d = s.par["problem"]["Z_dust"]
        # calculate normalized radiation fields
        Erad_PE0 = s.par["cooling"]["Erad_PE0"] / s.u.energy_density.cgs.value
        Erad_LW0 = s.par["cooling"]["Erad_LW0"] / s.u.energy_density.cgs.value
        # set flags
        try:
            CRphotC = True if s.par["cooling"]["iCRPhotC"] == 1 else False
        except KeyError:
            CRphotC = False
        try:
            iCII_rec_rate = True if s.par["cooling"]["iCRPhotC"] == 1 else False
        except KeyError:
            iCII_rec_rate = False

        def _xCII(field, data):
            nH = data[("athena", "density")].v
            xe = data[("athena", "xe")].v
            xH2 = data[("athena", "xH2")].v
            T = data[("athena", "temperature")].v
            G_PE = data[("athena", "rad_energy_density_PE")].v / Erad_PE0
            G_CI = data[("athena", "rad_energy_density_LW")].v / Erad_LW0
            CR_rate = data[("athena", "CR_ionization_rate")].v
            return get_xCII(
                nH,
                xe,
                xH2,
                T,
                Z_d,
                Z_g,
                CR_rate,
                G_PE,
                G_CI,
                xCstd=xCstd,
                gr_rec=True,
                CRphotC=CRphotC,
                iCII_rec_rate=iCII_rec_rate,
            )

        def _Lambda_CII(field, data):
            nH = data[("athena", "density")].v
            xe = data[("athena", "xe")].v
            xHI = data[("athena", "xHI")].v
            xH2 = data[("athena", "xH2")].v
            xCII = data[("gas", "xCII")]
            T = data[("athena", "temperature")].v

            g0CII_ = 2.0
            g1CII_ = 4.0

            A10CII_ = 2.3e-6
            E10CII_ = 1.26e-14
            kB_cgs = ac.k_B.cgs.value

            q10 = q10CII_(nH, T, xe, xHI, xH2)
            q01 = (g1CII_ / g0CII_) * q10 * np.exp(-E10CII_ / (kB_cgs * T))
            T4 = data[("athena", "temperature")].v / 1.0e4
            idx = T4 < 3.5
            return (
                (q01 / (q01 + q10 + A10CII_) * A10CII_ * E10CII_ * xCII / nH)
                * idx
                * yt.units.erg
                / yt.units.s
                * yt.units.cm**3
            )

        def _jCII(field, data):
            return (
                data[("gas", "Lambda_CII")]
                * data[("gas", "H_nuclei_density")] ** 2
            )

        def _LCII(field, data):
            return data[("gas", "CII_emissivity")] * data[("gas", "cell_volume")]

        ds.add_field(
            ("gas", "xCII"),
            function=_xCII,
            force_override=True,
            display_name=r"$x_{\rm C^+}$",
            sampling_type="cell",
        )
        ds.add_field(
            ("gas", "Lambda_CII"),
            function=_Lambda_CII,
            force_override=True,
            units="erg*cm**3/s",
            display_name=r"$\Lambda_{\rm C^+}$",
            sampling_type="cell",
        )
        ds.add_field(
            ("gas", "CII_emissivity"),
            function=_jCII,
            force_override=True,
            units="erg/cm**3/s",
            display_name=r"$4\pi j_{\rm CII}$",
            sampling_type="cell",
        )
        ds.add_field(
            ("gas", "CII_luminosity"),
            function=_LCII,
            force_override=True,
            units="erg/s",
            display_name=r"$L_{\rm C^+}$",
            sampling_type="cell",
        )

    # more fields
    def _specific_enthalphy(field,data):
        return data["gas", "specific_thermal_energy"] + data["gas", "pressure"]/data["gas", "density"]
    def _specific_kinetic_energy(field,data):
        return 0.5*data["gas", "velocity_magnitude"]**2
    def _total_energy_density(field,data):
        return data["gas", "specific_total_energy"]*data["gas", "density"]
    def _total_energy_flux_z(field,data):
        return (data["gas", "specific_enthalphy"] + data["gas", "specific_kinetic_energy"])*data["gas", f"momentum_density_z"]
    def _vzout(field,data):
        return data["gas", "velocity_z"]*(data["gas","z"]/data["gas","z"])
    def _vzin(field,data):
        return data["gas", "velocity_z"]*(-data["gas","z"]/data["gas","z"])

    ds.add_field(("gas","specific_enthalphy"),
                function=_specific_enthalphy,
                units='(km/s)**2',
                sampling_type="cell",force_override=True)
    ds.add_field(("gas","specific_kinetic_energy"),
                function=_specific_kinetic_energy,
                units='(km/s)**2',
                sampling_type="cell",force_override=True)
    ds.add_field(("gas","total_energy_density"),
                function=_total_energy_density,
                units='erg/cm**3',
                sampling_type="cell",force_override=True)
    ds.add_field(("gas","total_energy_flux_z"),
                function=_total_energy_flux_z,
                units='erg/s/cm**2',
                sampling_type="cell",force_override=True)
    ds.add_field(("gas","vzout"),
                function=_vzout,
                units='km/s',
                sampling_type="cell",force_override=True)
    ds.add_field(("gas","vzin"),
                function=_vzin,
                units='km/s',
                sampling_type="cell",force_override=True)

    return ds

def get_mytf(b, c, nlayer=0):
    def linramp(vals, minval, maxval):
        return 0.9 * (vals - vals.min()) / (vals.max() - vals.min()) + 0.1
    tf = yt.ColorTransferFunction((np.log10(b[0]), np.log10(b[1])))
    if nlayer == 0:
        tf.map_to_colormap(
            np.log10(b[0]), np.log10(b[1]), scale=20, scale_func=linramp, colormap=c
        )
    else:
        tf.add_layers(nlayer, w=0.3, alpha=np.linspace(10,40,nlayer), colormap=c)

    return tf

def render_volume(ds, f, b, c, nlayer=0, render = True):
    sc = yt.create_scene(ds, field=f)

    tf = get_mytf(b, c, nlayer=nlayer)
    sc[0].set_transfer_function(tf)

    cam = sc.camera
    cam.set_position([1024,-1024,1024],north_vector=[0,0,1])
    cam.zoom(1.5)
    cam.set_resolution(1024)
    sc.annotate_domain(ds,color=[1,1,1,1])
    if render:
        im = sc.render()
    else:
        im = None

    return im, tf, sc

def add_volume_source(sc, ds, f, b, c, nlayer=0, render=True):
    vol = create_volume_source(ds, field=f)
    tf = get_mytf(b, c, nlayer=nlayer)
    vol.set_transfer_function(tf)

    sc.add_source(vol)
    if render:
        im = sc.render()
    else:
        im = None

    return im, tf

def add_tf_to_image(fig, ds, f, tf, xoff=0.1):
    ax2 = fig.add_axes([1-xoff,0.1,0.05,0.8])
    tf.vert_cbar(256,False,ax2,label_fmt="%d")
    if f[1].startswith('xray'):
        label = f"log ${ds.field_info[f].display_name.replace('$','')}\,[{ds.field_info[f].units}]$"
    else:
        label = f"log {ds.field_info[f].display_name} [{ds.field_info[f].units}]"
    ax2.set_ylabel(label,weight='bold',fontsize=15)

def save_with_tf(ds, f, im, tf, fout = None, xoff = 0.1):
    fig = plt.figure(figsize=(5,5),dpi=200)
    ax = fig.add_axes([0,0,1,1])
    ax.axis('off')
    ax.imshow(im.swapaxes(0,1))
    add_tf_to_image(fig, ds, f, tf, xoff=xoff)
    ax.annotate(f"t={ds.current_time.to('Myr').v:5.1f} Myr",(xoff,0.95),
                xycoords='axes fraction',ha='left',va='top',weight='bold')
    if fout is not None:
        fig.savefig(fout,dpi=200,bbox_inches='tight')
    print(f'file saved: {fout}')

    return fig

def make_many_volumes(s, ds, num):
    ncr = s.par['configure']['new_cooling'] == 'ON'
    ncrsp = s.par['configure']['SpiralArm'] == 'yes'
    xoff = -0.1 if ncrsp else 0.1

    foutdir = osp.join(os.fspath(s.basedir), "volume")
    fields = [
        ("gas", "HI_21cm_luminosity"),
        ("gas", "H_alpha_luminosity"),
        ("gas", "xray_luminosity_0.5_7.0_keV"),
        ("gas", "CII_luminosity"),
        #   ("athena","rad_energy_density_PE"),
        #   ("athena","rad_energy_density_PH"),
        # ("athena", "specific_scalar[0]"),
        # ("athena", "specific_scalar[1]"),
        ("gas", "H_nuclei_density"),
        ("gas", "H_p0_number_density"),
        ("gas", "H_p1_number_density"),
        ("gas", "H2_number_density"),
    ]
    bounds = [
        (1.0e20, 1.0e30),
        (1.0e24, 1.0e34),
        (1.0e24, 1.0e34),
        (1.0e22, 1.0e32),
        #   (1.e-3,1.e3),(1.e-15,1.e4),
        # (0.02, 0.2),
        # (0.01, 1),
        (1.0e-4, 1.0e2),
        (1.0e-4, 1.0e2),
        (1.0e-4, 1.0e2),
        (1.0e-4, 1.0e2),
    ]
    cmaps = [
        "cool_r",
        "plasma",
        "winter",
        "cividis",
        # "cool",
        # "cool",
        "viridis",
        "viridis",
        "viridis",
        "viridis",
    ]

    for f, b, c in zip(fields, bounds, cmaps):
        im, tf, sc = render_volume(ds, f, b, c, nlayer=0)
        fout = osp.join(
            foutdir,
            f"{f[1].replace('[','').replace(']','')}_linramp_time_{num:04d}.png",
        )
        save_with_tf(ds, f, im, tf, fout=fout, xoff=xoff)

        im, tf, sc = render_volume(ds, f, b, c, nlayer=7)
        fout = osp.join(
            foutdir, f"{f[1].replace('[','').replace(']','')}_time_{num:04d}.png"
        )
        save_with_tf(ds, f, im, tf, fout=fout, xoff=xoff)

    # separately
    # Halpha,HI,Xray
    r,g,b = [1,0,2]
    fields_rgb = [fields[r],fields[g],fields[b]]
    bounds_rgb = [bounds[r],bounds[g],bounds[b]]
    cmaps_rgb = ['Reds','Greens','Blues']

    for channel,f,b,c in zip(['R','G','B'],fields_rgb,bounds_rgb,cmaps_rgb):
        im, tf, sc = render_volume(ds, f, b, c, nlayer=0)
        fout = osp.join(
                foutdir, f"luminosity_{channel}_time_{num:04d}.png"
            )
        save_with_tf(ds, f, im, tf, fout=fout, xoff=xoff)

    # combined
    im, tf_r, sc = render_volume(ds, fields_rgb[0], bounds_rgb[0], cmaps_rgb[0],
                                 nlayer=0, render=False)
    im, tf_g = add_volume_source(sc, ds, fields_rgb[1], bounds_rgb[1], cmaps_rgb[1],
                                 render=False)
    im, tf_b = add_volume_source(sc, ds, fields_rgb[2], bounds_rgb[2], cmaps_rgb[2],
                                 render=True)
    fig = save_with_tf(ds, fields_rgb[0], im, tf_r, xoff=xoff)
    add_tf_to_image(fig, ds, fields_rgb[1], tf_g, xoff=xoff-0.2)
    add_tf_to_image(fig, ds, fields_rgb[2], tf_b, xoff=xoff-0.4)
    fout = osp.join(foutdir, f"luminosity_RGB_time_{num:04d}.png")
    fig.savefig(fout,dpi=200,bbox_inches='tight')


def make_joint_pdfs(s, ds):
    foutdir = osp.join(os.fspath(s.basedir), "volume")

    Nx, Ny, Nz = ds.domain_dimensions
    le = ds.domain_left_edge.v
    re = ds.domain_right_edge.v
    profile = yt.create_profile(
        data_source=ds.all_data(),
        bin_fields=[("gas", "H_nuclei_density"), ("gas", "temperature")],
        fields=[
            ("gas", "cell_volume"),
            ("gas", "cell_mass"),
            ("gas", "H_alpha_luminosity"),
            ("athena", "rad_energy_density_PE"),
            ("athena", "rad_energy_density_PH"),
            ("athena", "specific_scalar[0]"),
            ("athena", "specific_scalar[1]"),
            ("gas", "xray_luminosity_0.5_7.0_keV"),
        ],
        n_bins=(256, 256),
        # units=dict(z="pc", velocity_z="km/s", volume="pc**3", mass="Msun"),
        weight_field=None,
        extrema=dict(H_nuclei_density=(1.0e-6, 1.0e4), temperature=(10, 1.0e9)),
    )


def make_volume(ds):
    from yt.visualization.volume_rendering.api import Scene, create_volume_source

    box = ds.box(ds.domain_left_edge, ds.domain_right_edge)

    # Add density
    sc = yt.create_scene(box, field=("athena", "density"))

    bounds = (1.0e-2, 1.0e2)
    tf = yt.ColorTransferFunction((np.log10(bounds[0]), np.log10(bounds[1])))
    tf.add_layers(5, w=0.01, alpha=np.logspace(1, 3, 5), colormap="winter")

    sc[0].set_use_ghost_zones(True)
    sc[0].set_transfer_function(tf)
    # sc[0].tfh.plot("transfer_function.png", profile_field="density")

    # Add temperature

    field = ("athena", "temperature")

    vol2 = create_volume_source(box, field=field)
    vol2.set_use_ghost_zones(True)

    bounds = (1.0e5, 1.0e7)
    tf = yt.ColorTransferFunction((np.log10(bounds[0]), np.log10(bounds[1])))
    tf.add_layers(3, 0.02, alpha=[10] * 3, colormap="autumn_r")

    vol2.set_transfer_function(tf)

    sc.add_source(vol2)

    cam = sc.camera
    cam.set_position([1024, -1024, 3072], north_vector=[0, 0, 1])
    cam.zoom(1.5)

    sc.annotate_domain(ds, color=[1, 1, 1, 1])
    return sc


if __name__ == "__main__":
    COMM = MPI.COMM_WORLD

    basedir_def = "/tigress/changgoo/TIGRESS-NCR/R8_4pc_NCR"

    savdir = None
    savdir_pkl = None

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--basedir", type=str, default=basedir_def, help="Name of the basedir."
    )
    args = vars(parser.parse_args())
    locals().update(args)

    s = pa.LoadSimTIGRESSNCR(basedir, verbose=True, load_method="yt")

    nums = s.nums

    if COMM.rank == 0:
        print("basedir, nums", s.basedir, nums)
        nums = split_container(nums, COMM.size)
    else:
        nums = None

    mynums = COMM.scatter(nums, root=0)
    print("[rank, mynums]:", COMM.rank, mynums)
    foutdir = osp.join(os.fspath(s.basedir), "volume")
    os.makedirs(foutdir, exist_ok=True)
    time0 = time.time()
    if True:
        for num in mynums:
            ds = s.ytload(num)
            ds = add_fields(s,ds,xray=True,CII=True)

            with plt.style.context("dark_background"):
                make_many_volumes(s, ds, num)

            n = gc.collect()
            print("Unreachable objects:", n, end=" ")
            print("Remaining Garbage:", end=" ")
            pprint.pprint(gc.garbage)
            sys.stdout.flush()
