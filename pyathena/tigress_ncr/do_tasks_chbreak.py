#!/usr/bin/env python

# import os
# import os.path as osp
import gc
import time
from mpi4py import MPI

# import matplotlib.pyplot as plt
import pprint
import argparse
import sys
# import pickle

import pyathena as pa

# from pyathena.plt_tools.make_movie import make_movie
# from pyathena.tigress_ncr.phase import *
from pyathena.tigress_ncr.cooling_breakdown import draw_Tpdf
from pyathena.tigress_ncr.do_tasks import scatter_nums

if __name__ == "__main__":
    COMM = MPI.COMM_WORLD

    basedir = "/tigress/changgoo/TIGRESS-NCR/R8_4pc_NCR"

    # savdir = '/tigress/jk11/tmp4/'
    # savdir_pkl = '/tigress/jk11/tmp3/'
    savdir = None
    savdir_pkl = None

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--basedir", type=str, default=basedir, help="Name of the basedir."
    )
    args = vars(parser.parse_args())
    locals().update(args)

    s = pa.LoadSimTIGRESSNCR(basedir, verbose=False)
    # get my nums
    if s.nums is not None:
        mynums = scatter_nums(s, s.nums, COMM)
    else:
        mynums = []

    print("[rank, mynums]:", COMM.rank, mynums)

    time0 = time.time()
    for num in mynums:
        print(num, end=" ")
        f1 = draw_Tpdf(s, num)

        n = gc.collect()
        print("Unreachable objects:", n, end=" ")
        print("Remaining Garbage:", end=" ")
        pprint.pprint(gc.garbage)
        sys.stdout.flush()

    # Make movies
    COMM.barrier()
