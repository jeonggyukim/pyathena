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

    time0 = time.time()
    if True:
        for num in mynums:
            ds = s.ytload(num)
            sc = make_volume(ds)
            foutdir = osp.join(os.fspath(s.basedir), "volume")
            os.makedirs(foutdir, exist_ok=True)
            fout = osp.join(foutdir, f"time_{num:04d}.png")
            sc.save(fout)

            n = gc.collect()
            print("Unreachable objects:", n, end=" ")
            print("Remaining Garbage:", end=" ")
            pprint.pprint(gc.garbage)
            sys.stdout.flush()