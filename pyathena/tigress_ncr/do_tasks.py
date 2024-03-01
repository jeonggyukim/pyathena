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
import pickle

import pyathena as pa
from pyathena.util.split_container import split_container
from pyathena.plt_tools.make_movie import make_movie
from pyathena.tigress_ncr.phase import PDF1D
from pyathena.tigress_ncr.cooling_breakdown import draw_Tpdf
from .do_tasks_phase import process_one_file_phase


def scatter_nums(s, nums):
    if COMM.rank == 0:
        print("basedir, nums", s.basedir, nums)
        nums = split_container(nums, COMM.size)
    else:
        nums = None

    mynums = COMM.scatter(nums, root=0)
    print("[rank, mynums]:", COMM.rank, mynums)

    return mynums


def process_tar(s):
    if s.nums_rawtar is not None:
        mynums = scatter_nums(s, s.nums_rawtar)

        for num in mynums:
            s.create_tar(num=num, kind="vtk", remove_original=True, overwrite=True)
            gc.collect()
        COMM.barrier()

        # reading it again
        s = pa.LoadSimTIGRESSNCR(basedir, verbose=False)
    return s


def process_one_file_slc_prj(s, num):
    try:
        if s.test_newcool():
            fig = s.plt_snapshot(
                num,
                savdir_pkl=savdir_pkl,
                savdir=savdir,
                force_override=False,
                norm_factor=2,
            )
            plt.close(fig)
            fig = s.plt_pdf2d_all(
                num, plt_zprof=False, savdir_pkl=savdir_pkl, savdir=savdir
            )
            plt.close(fig)
        else:
            fig = s.plt_snapshot(
                num,
                savdir_pkl=savdir_pkl,
                savdir=savdir,
                force_override=True,
                fields_xy=("Sigma_gas", "nH", "T", "Bmag"),
                fields_xz=("Sigma_gas", "nH", "T", "vz", "Bmag"),
                norm_factor=4,
                agemax=40,
            )
    except (EOFError, KeyError, pickle.UnpicklingError):
        if s.test_newcool():
            fig = s.plt_snapshot(
                num,
                savdir_pkl=savdir_pkl,
                savdir=savdir,
                force_override=True,
                norm_factor=2,
            )
            plt.close(fig)
            fig = s.plt_pdf2d_all(
                num,
                plt_zprof=False,
                savdir_pkl=savdir_pkl,
                savdir=savdir,
                force_override=True,
            )
            plt.close(fig)
        else:
            fig = s.plt_snapshot(
                num,
                savdir_pkl=savdir_pkl,
                savdir=savdir,
                force_override=True,
                fields_xy=("Sigma_gas", "nH", "T", "Bmag"),
                fields_xz=("Sigma_gas", "nH", "T", "vz", "Bmag"),
                norm_factor=4,
                agemax=40,
            )


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
    # tar vtk files
    s = process_tar(s)

    # set PDF1D
    s.pdf = PDF1D(s)

    # get my nums
    mynums = scatter_nums(s, s.nums)

    time0 = time.time()
    for num in mynums:
        print(num, end=" ")

        # slc prj
        process_one_file_slc_prj(s, num)
        # 2d pdf
        process_one_file_phase(s, num)

        # 1d pdfs
        s.pdf.recal_1Dpdfs(num, force_override=False)

        # coolheat breakdown
        if s.test_newcool():
            f1 = draw_Tpdf(s, num)

        n = gc.collect()
        print("Unreachable objects:", n, end=" ")
        print("Remaining Garbage:", end=" ")
        pprint.pprint(gc.garbage)
        sys.stdout.flush()

    # Make movies
    COMM.barrier()

    if COMM.rank == 0:
        if not osp.isdir(osp.join(s.basedir, "movies")):
            os.mkdir(osp.join(s.basedir, "movies"))
        fin = osp.join(s.basedir, "snapshot/*.png")
        fout = osp.join(s.basedir, "movies/{0:s}_snapshot.mp4".format(s.basename))
        make_movie(fin, fout, fps_in=15, fps_out=15)
        from shutil import copyfile

        copyfile(
            fout,
            osp.join(
                "/tigress/changgoo/public_html/temporary_movies/TIGRESS-NCR",
                osp.basename(fout),
            ),
        )
        fin = osp.join(s.basedir, "pdf2d/*.png")
        fout = osp.join(s.basedir, "movies/{0:s}_pdf2d.mp4".format(s.basename))
        make_movie(fin, fout, fps_in=15, fps_out=15)
        from shutil import copyfile

        copyfile(
            fout,
            osp.join(
                "/tigress/changgoo/public_html/temporary_movies/TIGRESS-NCR",
                osp.basename(fout),
            ),
        )

        print("")
        print("################################################")
        print("# Do tasks")
        print("# Execution time [sec]: {:.1f}".format(time.time() - time0))
        print("################################################")
        print("")
