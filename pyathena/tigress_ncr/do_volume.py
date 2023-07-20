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

import yt

def add_fields(ds,xray=True):
    # yt standard abundance fields
    def _nHI(field, data):
        return data[("gas","H_nuclei_density")]*data[("athena","xHI")]
    def _nHII(field, data):
        return data[("gas","H_nuclei_density")]*(1-data[("athena","xHI")]-2*data[("athena","xH2")])
    def _nH2(field, data):
        return 2.0*data[("gas","H_nuclei_density")]*data[("athena","xH2")]
    def _LHalpha(field, data):
        T4 = data[("athena","temperature")].v / 1.0e4
        idx = (T4>0.1) * (T4<3)
        alpha_eff = 1.17e-13 * T4 ** (-0.942 - 0.031 * np.log(T4))
        hnu_halpha = 3.0268267464328148e-12
        xHII = (1-data[("athena","xHI")]-2*data[("athena","xH2")])
        return (alpha_eff * hnu_halpha * idx *
                data[("gas","El_number_density")] *
                data[("gas","H_p1_number_density")] *
                data[("gas","cell_volume")] *
                (yt.units.erg* yt.units.cm**3/yt.units.s)
                )
    # add/override fields
    ds.add_field(("gas","H_p0_number_density"),function=_nHI, force_override=True,
                units='cm**(-3)',display_name=r'$n_{\rm H^0}$', sampling_type="cell")
    ds.add_field(("gas","H_p1_number_density"),function=_nHII, force_override=True,
                units='cm**(-3)',display_name=r'$n_{\rm H^+}$', sampling_type="cell")
    ds.add_field(("gas","H2_number_density"),function=_nH2, force_override=True,
                units='cm**(-3)',display_name=r'$n_{\rm H_2}$', sampling_type="cell")
    ds.add_field(("gas","H_alpha_luminosity"),function=_LHalpha, force_override=True,
                units="erg/s",display_name=r'$L_{\rm H\alpha}$', sampling_type="cell")

    # xray
    if xray:
        import pyxsim
        emin=0.1
        emax=10
        nbins=1000
        model='spex'
        binscale='log'
        Zmet = 1.0
        srcmdl=pyxsim.CIESourceModel(model, emin, emax, nbins, Zmet,
                                    binscale=binscale, abund_table='aspl')
        xray_fields = srcmdl.make_source_fields(ds,0.5,7.0,force_override=True)
    return ds

def make_many_volumes(s,ds):
    foutdir = osp.join(os.fspath(s.basedir), "volume")
    fields = [("gas","xray_luminosity_0.5_7.0_keV"),
              ("gas","H_alpha_luminosity"),
            #   ("athena","rad_energy_density_PE"),
            #   ("athena","rad_energy_density_PH"),
              ("athena","specific_scalar[0]"),
              ("athena","specific_scalar[1]"),
              ("gas","H_nuclei_density"),
              ("gas","H_p0_number_density"),
              ("gas","H_p1_number_density"),
              ("gas","H2_number_density")
             ]
    bounds = [(1.e24,1.e32),(1.e24,1.e30),
            #   (1.e-3,1.e3),(1.e-15,1.e4),
              (0.02,0.2),(0.01,1),
              (1.e-4,1.e2),(1.e-4,1.e2),(1.e-4,1.e2),(1.e-4,1.e2)
              ]
    cmaps = ['winter','plasma',
            #  'inferno','inferno',
             'cool','cool',
             'viridis','viridis','viridis','viridis'
             ]
    alphas = [np.linspace(20,50,5),np.linspace(20,50,5),
              np.logspace(-1,2,5),np.logspace(-1,2,5),
              np.linspace(20,50,5),np.linspace(20,50,5),np.linspace(20,50,5),np.linspace(50,100,5)
              ]
    for f, b, c, a in zip(fields,bounds, cmaps, alphas):
        sc = yt.create_scene(ds,field=f)
        # tfout = osp.join(foutdir, f"{f[1].replace('[','').replace(']','')}_tf_{num:04d}.png")
        # sc[0].tfh.plot(tfout, profile_field=f)
        tf = yt.ColorTransferFunction((np.log10(b[0]),np.log10(b[1])))
        tf.add_layers(5, w=0.1, alpha=a, colormap=c)

        sc[0].set_transfer_function(tf)

        cam = sc.camera
        cam.set_position([1024,-1024,1024],north_vector=[0,0,1])
        cam.zoom(1.5)
        sc.annotate_domain(ds,color=[1,1,1,1])

        fout = osp.join(foutdir, f"{f[1].replace('[','').replace(']','')}_time_{num:04d}.png")
        sc.save(fout)

def make_joint_pdfs(s,ds):
    foutdir = osp.join(os.fspath(s.basedir), "volume")

    Nx, Ny, Nz = ds.domain_dimensions
    le = ds.domain_left_edge.v
    re = ds.domain_right_edge.v
    profile = yt.create_profile(
        data_source=ds.all_data(),
        bin_fields=[("gas", "H_nuclei_density"), ("gas", "temperature")],
        fields=[("gas","cell_volume"), ("gas", "cell_mass"),
                ("gas","H_alpha_luminosity"),
                ("athena","rad_energy_density_PE"),
                ("athena","rad_energy_density_PH"),
                ("athena","specific_scalar[0]"),
                ("athena","specific_scalar[1]"),
                ("gas","xray_luminosity_0.5_7.0_keV")
               ],
        n_bins=(256,256),
        # units=dict(z="pc", velocity_z="km/s", volume="pc**3", mass="Msun"),
        weight_field=None,
        extrema=dict(H_nuclei_density=(1.e-6,1.e4),temperature=(10,1.e9)),
    )
def make_volume(ds):
    from yt.visualization.volume_rendering.api import Scene, create_volume_source
    box = ds.box(ds.domain_left_edge,ds.domain_right_edge)

    # Add density
    sc = yt.create_scene(box,field=("athena","density"))

    bounds = (1.e-2,1.e2)
    tf = yt.ColorTransferFunction((np.log10(bounds[0]),np.log10(bounds[1])))
    tf.add_layers(5, w=0.01, alpha=np.logspace(1,3,5), colormap="winter")

    sc[0].set_use_ghost_zones(True)
    sc[0].set_transfer_function(tf)
    # sc[0].tfh.plot("transfer_function.png", profile_field="density")

    # Add temperature

    field = ("athena","temperature")

    vol2 = create_volume_source(box, field=field)
    vol2.set_use_ghost_zones(True)

    bounds = (1.e5,1.e7)
    tf = yt.ColorTransferFunction((np.log10(bounds[0]),np.log10(bounds[1])))
    tf.add_layers(3, 0.02, alpha=[10]*3, colormap="autumn_r")

    vol2.set_transfer_function(tf)

    sc.add_source(vol2)

    cam = sc.camera
    cam.set_position([1024,-1024,3072],north_vector=[0,0,1])
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

    s=pa.LoadSimTIGRESSNCR(basedir,verbose=True,load_method='yt')

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
            ds = add_fields(ds)
            # sc = make_volume(ds)
            # fout = osp.join(foutdir, f"time_{num:04d}.png")
            # sc.save(fout)

            sc = make_many_volumes(s,ds)

            n = gc.collect()
            print("Unreachable objects:", n, end=" ")
            print("Remaining Garbage:", end=" ")
            pprint.pprint(gc.garbage)
            sys.stdout.flush()