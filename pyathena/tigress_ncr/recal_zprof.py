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

import pyathena as pa
from pyathena.util.split_container import split_container
from pyathena.io.read_zprof import recal_zprof

if __name__ == '__main__':

    COMM = MPI.COMM_WORLD

    basedir_def = '/tigress/changgoo/TIGRESS-NCR/R8_4pc_NCR'

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

    time0 = time.time()
    for num in mynums:
        print(num, end=' ')
        recal_zprof(s,num,write=True)
        n = gc.collect()
        print('Unreachable objects:', n, end=' ')
        print('Remaining Garbage:', end=' ')
        pprint.pprint(gc.garbage)

    COMM.barrier()
    if COMM.rank == 0:
        print('')
        print('################################################')
        print('# Done recal_zprof')
        print('# Execution time [sec]: {:.1f}'.format(time.time()-time0))
        print('################################################')
        print('')
