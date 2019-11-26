#!/usr/bin/env python

import os
import os.path as osp
import time
from mpi4py import MPI
import matplotlib.pyplot as plt

import pyathena as pa
from ..util.split_container import split_container
from ..plt_tools.make_movie import make_movie

if __name__ == '__main__':
    COMM = MPI.COMM_WORLD

    basedir = '/scratch/gpfs/jk11/TIGRESS-RT/R4_4pc.RT.nowind/'
    s = pa.LoadSimTIGRESSRT(basedir, verbose=False)
    nums = s.nums
    
    if COMM.rank == 0:
        print('basedir, nums', s.basedir, nums)
        nums = split_container(nums, COMM.size)
    else:
        nums = None

    mynums = COMM.scatter(nums, root=0)
    print('[rank, mynums]:', COMM.rank, mynums)

    # for num in mynums:
    #     print(num, end=' ')
    #     # prj = s.read_prj(num, force_override=False)
    #     # slc = s.read_slc(num, force_override=False)
    #     try:
    #         fig = s.plt_snapshot2(num)
    #     except KeyError:
    #         fig = s.plt_snapshot2(num, force_override=True)
    #     plt.close(fig)

    # Make movies
    if COMM.rank == 0:
        fin = osp.join(s.basedir, 'snapshots2/*.png')
        fout = osp.join(s.basedir, 'movies/{0:s}_snapshots2.mp4'.format(s.basename))
        make_movie(fin, fout, fps_in=15, fps_out=15)
        from shutil import copyfile
        copyfile(fout, osp.join('/tigress/jk11/public_html/movies',
                                osp.basename(fout)))

        
    COMM.barrier()
    if COMM.rank == 0:
        print('')
        print('################################################')
        print('# Do tasks')
        print('# Execution time [sec]: {:.1f}'.format(time.time()-time0))
        print('################################################')
        print('')
