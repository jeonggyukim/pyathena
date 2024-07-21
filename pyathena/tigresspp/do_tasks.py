#!/usr/bin/env python

import os
import os.path as osp
import gc
import time
from mpi4py import MPI
import matplotlib.pyplot as plt
import pprint
import argparse

# import sys
import pickle

import pyathena as pa
from pyathena.util.split_container import split_container
from pyathena.plt_tools.make_movie import make_movie

if __name__ == "__main__":
    COMM = MPI.COMM_WORLD

    basedir_def = "/tigress/changgoo/TIGRESS-NCR/R8_4pc_NCR"
    basedir = basedir_def

    # savdir = '/tigress/jk11/tmp4/'
    # savdir_pkl = '/tigress/jk11/tmp3/'
    savdir = None
    savdir_pkl = None

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--basedir", type=str, default=basedir_def, help="Name of the basedir."
    )
    args = vars(parser.parse_args())
    locals().update(args)

    s = pa.LoadSimTIGRESSPP(basedir, verbose=False)
    nums = s.nums

    plt.rcParams["figure.dpi"] = 200
    if COMM.rank == 0:
        f1 = s.plt_hst()
        f1.savefig("sfr.png", bbox_inches="tight")
        f2 = s.plt_timing()
        f2.savefig("timing.png", bbox_inches="tight")

        print("basedir, nums", s.basedir, nums)
        nums = split_container(nums, COMM.size)
    else:
        nums = None

    mynums = COMM.scatter(nums, root=0)
    print("[rank, mynums]:", COMM.rank, mynums)

    time0 = time.time()
    for num in mynums:
        print(num, end=" ")

        try:
            fig = s.plt_snapshot(
                num, savdir_pkl=savdir_pkl, savdir=savdir, force_override=True
            )
            plt.close(fig)
            # fig = s.plt_pdf2d_all(num, plt_zprof=False, savdir_pkl=savdir_pkl, savdir=savdir)
            # plt.close(fig)
        except (EOFError, KeyError, pickle.UnpicklingError):
            fig = s.plt_snapshot(
                num, savdir_pkl=savdir_pkl, savdir=savdir, force_override=True
            )
            plt.close(fig)
            # fig = s.plt_pdf2d_all(num, plt_zprof=False, savdir_pkl=savdir_pkl, savdir=savdir, force_override=True)
            # plt.close(fig)

        n = gc.collect()
        print("Unreachable objects:", n, end=" ")
        print("Remaining Garbage:", end=" ")
        pprint.pprint(gc.garbage)

    # Make movies
    COMM.barrier()

    if COMM.rank == 0:
        if not osp.isdir(osp.join(s.basedir, "movies")):
            os.mkdir(osp.join(s.basedir, "movies"))
        fin = osp.join(s.basedir, "snapshot/*.png")
        fout = osp.join(s.basedir, "movies/{0:s}_snapshot.mp4".format(s.basename))
        make_movie(fin, fout, fps_in=15, fps_out=15)
        # from shutil import copyfile
        # copyfile(fout, osp.join('/tigress/changgoo/public_html/temporary_movies/TIGRESS-NCR',
        # osp.basename(fout)))
        # fin = osp.join(s.basedir, 'pdf2d/*.png')
        # fout = osp.join(s.basedir, 'movies/{0:s}_pdf2d.mp4'.format(s.basename))
        # make_movie(fin, fout, fps_in=15, fps_out=15)
        # from shutil import copyfile
        # copyfile(fout, osp.join('/tigress/changgoo/public_html/temporary_movies/TIGRESS-NCR',
        # osp.basename(fout)))

        print("")
        print("################################################")
        print("# Do tasks")
        print("# Execution time [sec]: {:.1f}".format(time.time() - time0))
        print("################################################")
        print("")
