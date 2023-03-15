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

if __name__ == '__main__':

    COMM = MPI.COMM_WORLD

    basedir_def = '/tigress/changgoo/TIGRESS-NCR/R8_8pc_NCR.full.xy2048.eps0.0'
    savdir = '/tigress/jk11/ncr_tmp/R8_8pc_NCR.full.xy2048.eps0.0/pionized'

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--basedir', type=str,
                        default=basedir_def,
                        help='Name of the basedir.')
    args = vars(parser.parse_args())
    locals().update(args)

    s = pa.LoadSimTIGRESSNCR(basedir, verbose=False)
    nums = s.nums

    if COMM.rank == 0:
        print('basedir, nums', s.basedir, nums)
        nums = split_container(nums, COMM.size)
    else:
        nums = None

    mynums = COMM.scatter(nums, root=0)
    print('[rank, mynums]:', COMM.rank, mynums)
    COMM.barrier()

    time0 = time.time()
    for num in mynums:
        print(num, end=' ')
        try:
            s.read_zprof_partially_ionized(num, savdir=savdir, force_override=True)
        except (EOFError, KeyError, pickle.UnpicklingError):
            s.read_zprof_partially_ionized(num, savdir=savdir, force_override=True)

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
