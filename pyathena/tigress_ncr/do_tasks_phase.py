#!/usr/bin/env python

import os
import gc
import time
from mpi4py import MPI
import pprint
import argparse
import sys

import pyathena as pa
from pyathena.tigress_ncr.phase import recal_nP, recal_xT
from .do_tasks import process_tar, scatter_nums


def process_one_file_phase(s, num):
    npfile = os.path.join(
        s.basedir, "np_pdf", "{}.{:04d}.np_pdf-z300.nc".format(s.basename, num)
    )
    nTfile = os.path.join(
        s.basedir, "nT_pdf", "{}.{:04d}.nT_pdf-z300.nc".format(s.basename, num)
    )
    xTfile = os.path.join(
        s.basedir, "xT_pdf", "{}.{:04d}.xT_pdf-z300.nc".format(s.basename, num)
    )
    os.makedirs(os.path.dirname(npfile), exist_ok=True)
    os.makedirs(os.path.dirname(nTfile), exist_ok=True)
    os.makedirs(os.path.dirname(xTfile), exist_ok=True)
    if not os.path.isfile(xTfile):
        ds = s.load_vtk(num)
        flist = ["nH", "pok", "T"]
        if s.test_newcool():
            flist += ["xe", "xHI", "xHII", "xH2", "cool_rate", "net_cool_rate"]
        dchunk = ds.get_field(flist)
        dchunk["T1"] = dchunk["pok"] / dchunk["nH"]
        print(" creating nP ")
        for zmax in [300, 1000, None]:
            if zmax is not None:
                dchunk = dchunk.sel(z=slice(-zmax, zmax))
                npfile_ = npfile.replace("-z300.nc", f"-z{zmax}.nc")
                nTfile_ = nTfile.replace("-z300.nc", f"-z{zmax}.nc")
                xTfile_ = xTfile.replace("-z300.nc", f"-z{zmax}.nc")
            else:
                npfile_ = npfile.replace("-z300.nc", ".nc")
                nTfile_ = nTfile.replace("-z300.nc", ".nc")
                xTfile_ = xTfile.replace("-z300.nc", ".nc")
            pdf_dset = recal_nP(dchunk, NCR=s.test_newcool())
            pdf_dset.to_netcdf(npfile_)
            pdf_dset = recal_nP(dchunk, yf="T", NCR=s.test_newcool())
            pdf_dset.to_netcdf(nTfile_)
            hist_bin = recal_xT(dchunk)
            hist_bin.to_netcdf(xTfile_)

        # close
        ds.close()
    else:
        print(" skipping nP ")


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

    s = pa.LoadSimTIGRESSNCR(basedir, verbose=False)  # noqa
    # tar vtk files
    if s.nums_rawtar is not None:
        s = process_tar(s)

    # get my nums
    if s.nums is not None:
        mynums = scatter_nums(s, s.nums)
    else:
        mynums = []

    print("[rank, mynums]:", COMM.rank, mynums)

    time0 = time.time()
    for num in mynums:
        print(num, end=" ")

        # 2d pdf
        try:
            process_one_file_phase(s, num)
        except IOError:
            print(" passing nP ")

        n = gc.collect()
        print("Unreachable objects:", n, end=" ")
        print("Remaining Garbage:", end=" ")
        pprint.pprint(gc.garbage)
        sys.stdout.flush()
