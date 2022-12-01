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
    # tar vtk files
    if s.nums_rawtar is not None:
        nums = s.nums_rawtar
        if COMM.rank == 0:
            print("basedir, nums", s.basedir, nums)
            nums = split_container(nums, COMM.size)
            os.makedirs(os.path.join(s.savdir,'xprof'),exist_ok=True)
        else:
            nums = None

        mynums = COMM.scatter(nums, root=0)
        for num in mynums:
            s.create_tar(num=num,kind='vtk',remove_original=True)
        COMM.barrier()

        # reading it again
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
        ds = s.load_vtk(num)

        data = ds.get_field(field=['nH','pok','T','xHI','xHII','xH2','vz'])

        ph = assign_phase(s,data,kind='six')

        phclist = [phdict['c'] for phdict in ph.attrs['phdef']]
        vz_ph = []
        for i,(phname,phcolor) in enumerate(zip(ph.attrs['phlist'],phclist)):
            phsel=data.where(ph==i)
            vz=xr.Dataset()
            vz['rhovz2']=(phsel['nH']*phsel['vz']**2).sum(dim=['y','z'])
            vz['rho']=(phsel['nH']).sum(dim=['y','z'])
            vz['pok']=(phsel['pok']).sum(dim=['y','z'])
            vz_ph.append(vz.assign_coords(phase=phname))
        vz_ph = xr.concat(vz_ph,dim='phase')
        vz_ph = vz_ph.assign_coords(time=ds.domain['time'])
        vz_ph.to_netcdf(os.path.join(s.savdir,'xprof',
                                    '{}.{:04d}.xprof.nc'.format(s.basename,num)))
        n = gc.collect()
        print("Unreachable objects:", n, end=" ")
        print("Remaining Garbage:", end=" ")
        pprint.pprint(gc.garbage)
        sys.stdout.flush()

    # Make movies
    COMM.barrier()