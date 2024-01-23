#!/usr/bin/env python

import os
import os.path as osp
import gc
import time
from mpi4py import MPI
import matplotlib.pyplot as plt
import pprint
import sys
import pickle

import pyathena as pa
from pyathena.util.split_container import split_container

if __name__ == '__main__':
    COMM = MPI.COMM_WORLD

    basedir = '/projects/EOSTRIKE/TIGRESS-NCR'
    # All subdirectories
    pp = [osp.join(basedir, p) for p in next(os.walk(basedir))[1]]

    idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
    basedir = pp[idx]
    model = osp.basename(basedir)
    savdir = osp.join('/tigress/jk11/NCR-RAD-LOWZ', model, 'pionized')
    if not osp.exists(savdir):
        os.makedirs(savdir)

    s = pa.LoadSimTIGRESSNCR(basedir, savdir=savdir, verbose=False)
    nums = s.nums

    if COMM.rank == 0:
        print('basedir, nums', s.basedir, nums)
        nums = split_container(nums, COMM.size)
    else:
        nums = None

    mynums = COMM.scatter(nums, root=0)
    print('[rank, mynums]:', COMM.rank, mynums)

    time0 = time.time()
    for num in mynums:
        print(num, end=' ')
        rr = s.read_zprof_partially_ionized(num, savdir=savdir)
        n = gc.collect()
        print('Unreachable objects:', n, end=' ')
        print('Remaining Garbage:', end=' ')
        pprint.pprint(gc.garbage)

    COMM.barrier()
    if COMM.rank == 0:
        print('')
        print('################################################')
        print('# Do tasks')
        print('# Execution time [sec]: {:.1f}'.format(time.time()-time0))
        print('################################################')
        print('')
