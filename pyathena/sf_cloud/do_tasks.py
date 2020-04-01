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
from .load_sim_sf_cloud import load_all_alphabeta


if __name__ == '__main__':

    COMM = MPI.COMM_WORLD

    #basedir = '/scratch/gpfs/jk11/GMC/M1E5R20.R.mu2.A2.S1.N512.test/'
    #basedir = '/scratch/gpfs/jk11/GMC/M1E5R20.R.Binf.A2.S1.N256.Lconst.kexpo2/'
    #basedir = '/scratch/gpfs/jk11/GMC/M1E5R20.R.Binf.A2.S1.N256.Lconst.kexpo3/'
    #basedir = '/scratch/gpfs/jk11/GMC/M1E5R20.R.Binf.A2.S1.N256/'
    # basedir = '/tigress/jk11/GMC/M1E5R20.R.B2.A2.S4.N256/'
    # s = pa.LoadSimSFCloud(basedir, verbose=False)
    # nums = s.nums
    # nums = range(0,get_num_max_virial(s))

    sa, r = load_all_alphabeta()
    models = ['B1S4']
    models = ['B2S4']
    models = ['B4S4']
    models = ['B8S4']
    models = ['A1S4']
    models = ['A3S4']
    models = ['A4S4']
    models = ['A5S4']
    for mdl in models:
        print(mdl)
        s = sa.set_model(mdl)
        nums = range(0,s.get_num_max_virial())
        
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

            print('virial', end=' ')
            r = s.read_virial(num, force_override=True)
            n = gc.collect()
            print('Unreachable objects:', n, end=' ')
            print('Remaining Garbage:', end=' ')
            pprint.pprint(gc.garbage)

            # print('read_slc_prj', end=' ')
            # # slc = s.read_slc(num, force_override=False)
            # prj = s.read_prj(num, force_override=False)
            # n = gc.collect()
            # print('Unreachable objects:', n, end=' ')
            # print('Remaining Garbage:', end=' ')
            # pprint.pprint(gc.garbage)

            # print('read_IQU', end=' ')
            # r = s.read_IQU(num, force_override=False)
            # n = gc.collect()
            # print('Unreachable objects:', n, end=' ')
            # print('Remaining Garbage:', end=' ')
            # pprint.pprint(gc.garbage)

            # print('plt_Bproj', end=' ')        
            # fig = s.plt_Bproj(num)

            # print('plt_snapshot')
            # try:
            #     fig = s.plt_snapshot(num)
            # except KeyError:
            #     fig = s.plt_snapshot(num, force_override=True)
            # plt.close(fig)

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

            
