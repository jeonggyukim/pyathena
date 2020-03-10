#!/usr/bin/env python

import os
import os.path as osp
import time
import gc
from mpi4py import MPI
import matplotlib.pyplot as plt
import pprint

import pyathena as pa
from ..util.split_container import split_container
from ..plt_tools.make_movie import make_movie

if __name__ == '__main__':

    COMM = MPI.COMM_WORLD

    basedir = '/perseus/scratch/gpfs/jk11/GMC/M1E5R20.R.mu2.A2.S1.N512.test/'
    s = pa.LoadSimSFCloud(basedir, verbose=False)
    nums = s.nums[53:]
    
    if COMM.rank == 0:
        print('basedir, nums', s.basedir, nums)
        nums = split_container(nums, COMM.size)
    else:
        nums = None

    mynums = COMM.scatter(nums, root=0)
    print('[rank, mynums]:', COMM.rank, mynums)

    for num in mynums:
        print(num, end=' ')

        print('read_slc_prj', end=' ')
        # slc = s.read_slc(num, force_override=False)
        prj = s.read_prj(num, force_override=False)
        n = gc.collect()
        print('Unreachable objects:', n, end=' ')
        print('Remaining Garbage:', end=' ')
        pprint.pprint(gc.garbage)

        print('read_IQU', end=' ')
        r = s.read_IQU(num, force_override=False)
        n = gc.collect()
        print('Unreachable objects:', n, end=' ')
        print('Remaining Garbage:', end=' ')
        pprint.pprint(gc.garbage)

        print('plt_Bproj', end=' ')        
        fig = s.plt_Bproj(num)

        print('plt_snapshot')
        try:
            fig = s.plt_snapshot(num)
        except KeyError:
            fig = s.plt_snapshot(num, force_override=True)
        plt.close(fig)

    # # Make movies
    # if COMM.rank == 0:
    #     fin = osp.join(s.basedir, 'snapshots/*.png')
    #     fout = osp.join(s.basedir, 'movies/{0:s}_snapshots.mp4'.format(s.basename))
    #     make_movie(fin, fout, fps_in=15, fps_out=15)
    #     from shutil import copyfile
    #     copyfile(fout, osp.join('/tigress/jk11/public_html/movies',
    #                             osp.basename(fout)))
        
    COMM.barrier()
    if COMM.rank == 0:
        print('')
        print('################################################')
        print('# Do tasks')
        print('# Execution time [sec]: {:.1f}'.format(time.time()-time0))
        print('################################################')
        print('')
