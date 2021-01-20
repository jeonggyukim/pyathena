#!/usr/bin/env python

import os
import os.path as osp
import gc
import time
from mpi4py import MPI
import matplotlib.pyplot as plt
import pprint
import argparse

import pyathena as pa
from ..util.split_container import split_container
from ..plt_tools.make_movie import make_movie

if __name__ == '__main__':

    movie = False
    COMM = MPI.COMM_WORLD

    #basedir_def = '/tigress/jk11/TIGRESS-NCR/R8_8pc_NCR_test6/'
    #basedir_def = '/perseus/scratch/gpfs/jk11/TIGRESS-NCR/LGR4_4pc_NCR_oldvl/'

    #basedir_def = '/tigress/changgoo/TIGRESS-NCR/R8_4pc_NCR.cori'
    basedir_def = '/tigress/jk11/TIGRESS-NCR/R8_4pc_NCR.cori'
    #basedir_def = '/tigress/jk11/TIGRESS-NCR/LGR4_2pc_NCR.cori'
    #basedir_def = '/tiger/scratch/gpfs/jk11/TIGRESS-NCR/LGR4_4pc_NCR'

    # savdir = '/tigress/jk11/tmp4/'
    # savdir_pkl = '/tigress/jk11/tmp3/'
    savdir = None
    savdir_pkl = None
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--basedir', type=str,
                        default=basedir_def,
                        help='Name of the basedir.')
    args = vars(parser.parse_args())
    locals().update(args)

    s = pa.LoadSimTIGRESSNCR(basedir, verbose=False)
    nums = s.nums[::5]
    
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

        # # prj = s.read_prj(num, force_override=False)
        # # slc = s.read_slc(num, force_override=False)
        # try:
        #     fig = s.plt_snapshot(num, savdir_pkl=savdir_pkl, savdir=savdir)
        #     plt.close(fig)
        #     fig = s.plt_pdf2d_all(num, plt_zprof=False, savdir_pkl=savdir_pkl, savdir=savdir)
        #     plt.close(fig)
        # except KeyError:
        #     fig = s.plt_snapshot(num, savdir_pkl=savdir_pkl, savdir=savdir, force_override=True)
        #     plt.close(fig)
        #     fig = s.plt_pdf2d_all(num, plt_zprof=False, savdir_pkl=savdir_pkl, savdir=savdir, force_override=True)
        #     plt.close(fig)

        # n = gc.collect()
        # print('Unreachable objects:', n, end=' ')
        # print('Remaining Garbage:', end=' ')
        # pprint.pprint(gc.garbage)

        print('read_H2eq', end=' ')
        r = s.read_H2eq(num, force_override=False)
        n = gc.collect()
        print('Unreachable objects:', n, end=' ')
        print('Remaining Garbage:', end=' ')
        pprint.pprint(gc.garbage)

       
    # Make movies
    if COMM.rank == 0 and movie:
        fin = osp.join(s.basedir, 'snapshots/*.png')
        fout = osp.join(s.basedir, 'movies/{0:s}_snapshots.mp4'.format(s.basename))
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
