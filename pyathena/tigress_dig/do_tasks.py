#!/usr/bin/env python

import os
import os.path as osp
import time
import gc
import pprint
from mpi4py import MPI
import matplotlib.pyplot as plt

import pyathena as pa
from pyathena.util.split_container import split_container
#from ..util.split_container import split_container
# from ..plt_tools.make_movie import make_movie

if __name__ == '__main__':

    COMM = MPI.COMM_WORLD

    basedir = '/projects/EOSTRIKE/TIGRESS-DIG/R8_4pc_newacc.xymax1024'
    s = pa.LoadSimTIGRESSDIG(basedir, verbose=False, load_method='pyathena_classic')

    nums = s.nums
    # nums = s.nums[0:5]
    # nums = s.nums[5:10]
    # nums = s.nums[10:15]
    # nums = s.nums[15:20]
    # nums = s.nums[20:25]
    # nums = s.nums[25:30]
    # nums = s.nums[30:35]
    # nums = s.nums[35:40]

    # nums = s.nums[0:200]
    # nums = s.nums[150:300]
    # nums = s.nums[301:450]
    # nums = s.nums[450:571]

    # nums = s.nums[208:209]
    # nums = s.nums[300:350]
    # nums = s.nums[350:400]
    # nums = s.nums[400:450]
    # nums = s.nums[450:500]
    # nums = s.nums[500:550]
    # nums = s.nums[550:571]

    time0 = time.time()
    if COMM.rank == 0:
        print('basedir, nums', s.basedir, nums)
        nums = split_container(nums, COMM.size)
    else:
        nums = None

    mynums = COMM.scatter(nums, root=0)
    print('[rank, mynums]:', COMM.rank, mynums)

    for num in mynums:
        print(num, end=' ')
        # res = s.read_EM_pdf(num, force_override=True)
        #res = s.read_phot_dust_U_pdf(num, force_override=True)
        # res = s.read_VFF_Peters17(num, force_override=True)

        res = s.read_zprof_partially_ionized(num, force_override=True)
        n = gc.collect()
        print('Unreachable objects:', n)
        print('Remaining Garbage:', end=' ')
        pprint.pprint(gc.garbage)

    # if COMM.rank == 0:
    #     fin = osp.join(s.basedir, 'snapshots2/*.png')
    #     fout = osp.join(s.basedir, 'movies/{0:s}_snapshots2.mp4'.format(s.basename))
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
