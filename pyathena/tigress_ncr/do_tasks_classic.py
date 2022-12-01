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
from pyathena.tigress_ncr.phase import *

if __name__ == "__main__":

    COMM = MPI.COMM_WORLD

    basedir_def = "/tigress/changgoo/TIGRESS-NCR/R8_4pc_NCR"

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

    s = pa.LoadSimTIGRESSNCR(basedir, verbose=False)
    s.pdf = PDF1D(s)
    nums = s.nums

    if COMM.rank == 0:
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
                num,
                savdir_pkl=savdir_pkl,
                savdir=savdir,
                force_override=True,
                fields_xy=("Sigma_gas", "nH", "T", "Bmag"),
                fields_xz=("Sigma_gas", "nH", "T", "vz", "Bmag"),
            )
            plt.close(fig)
        except (EOFError, KeyError, pickle.UnpicklingError):
            fig = s.plt_snapshot(
                num,
                savdir_pkl=savdir_pkl,
                savdir=savdir,
                force_override=True,
                fields_xy=("Sigma_gas", "nH", "T", "Bmag"),
                fields_xz=("Sigma_gas", "nH", "T", "vz", "Bmag"),
            )
            plt.close(fig)
        # 2d pdf
        try:
            npfile=os.path.join(s.basedir,'np_pdf',
                                '{}.{:04d}.np_pdf.nc'.format(s.basename,num))
            if not os.path.isdir(os.path.dirname(npfile)):
                os.makedirs(os.path.dirname(npfile))
            if not os.path.isfile(npfile):
                ds=s.load_vtk(num)
                flist = ['nH','pok','T']
                if s.test_newcool():
                    flist.append(['xe','xHI','xHII','xH2',
                                  'cool_rate','net_cool_rate'])
                dchunk=ds.get_field(flist)
                dchunk['T1'] = dchunk['pok']/dchunk['nH']
                dchunk=dchunk.sel(z=slice(-300,300))
                print(" creating nP ", end=" ")
                pdf_dset = recal_nP(dchunk,NCR=s.test_newcool())
                pdf_dset.to_netcdf(npfile)
            else:
                print(" skipping nP ", end=" ")
        except IOError:
            print(" passing nP ", end=" ")

        # 1d pdfs
        s.pdf.recal_1Dpdfs(num,force_override=False)

        n = gc.collect()
        print("Unreachable objects:", n, end=" ")
        print("Remaining Garbage:", end=" ")
        pprint.pprint(gc.garbage)

    # Make movies
    COMM.barrier()
