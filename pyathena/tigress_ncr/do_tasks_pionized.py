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

    basedir_def = '/tigress/changgoo/TIGRESS-NCR/R8_4pc_NCR.full.xy2048.eps0.np768.has'
    savdir = '/tigress/jk11/NCR-RAD/R8_4pc_NCR.full.xy2048.eps0.np768.has/'

    # basedir = '/projects/EOSTRIKE/TIGRESS-NCR/'
    # for p in next(os.walk(basedir))[1]:
    #     print(p)

    # Low metallicity models
    # Z01_Zd0025_b1 = 'R8_8pc_NCR.full.b10.v3.iCR4.Zg0.1.Zd0.025.xy4096.eps0.0'
    # Z01 = 'R8_8pc_NCR.full.b10.v3.iCR4.Zg0.1.Zd0.1.xy4096.eps0.0'
    # Z03 = 'R8_8pc_NCR.full.b10.v3.iCR4.Zg0.3.Zd0.3.xy4096.eps0.0'
    # basedir_def = osp.join('/tigress/changgoo/TIGRESS-NCR', model)
    # savdir = osp.join('/tigress/jk11/NCR-RAD-LOWZ', model)

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--basedir', type=str,
                        default=basedir_def, help='Name of the basedir.')
    args = vars(parser.parse_args())
    locals().update(args)

    s = pa.LoadSimTIGRESSNCR(basedir, savdir=savdir, verbose=False)
    nums = [num for num in range(255,459)]
    force_override = True

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
        rr = s.read_zprof_Erad_LyC(num, force_override=force_override)
        rr2 = s.read_zprof_partially_ionized(num, force_override=force_override)
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
