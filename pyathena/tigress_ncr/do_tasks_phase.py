#!/usr/bin/env python

import os
import gc
import time
from mpi4py import MPI
import pprint
import argparse
import sys

import pyathena as pa
from pyathena.util.split_container import split_container
from pyathena.tigress_ncr.phase import PDF1D, recal_nP, recal_xT
# from pyathena.tigress_ncr.cooling_breakdown import

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
        nums = s.nums_rawtar
        if COMM.rank == 0:
            print("basedir, nums", s.basedir, nums)
            nums = split_container(nums, COMM.size)
        else:
            nums = None

        mynums = COMM.scatter(nums, root=0)
        for num in mynums:
            s.create_tar(num=num, kind="vtk", remove_original=True, overwrite=True)
            gc.collect()
        COMM.barrier()

        # reading it again
        s = pa.LoadSimTIGRESSNCR(basedir, verbose=False)  # noqa

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

        # 2d pdf
        try:
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

            else:
                print(" skipping nP ")
        except IOError:
            print(" passing nP ")

        n = gc.collect()
        print("Unreachable objects:", n, end=" ")
        print("Remaining Garbage:", end=" ")
        pprint.pprint(gc.garbage)
        sys.stdout.flush()
