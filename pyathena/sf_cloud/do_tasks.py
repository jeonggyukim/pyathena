#!/usr/bin/env python

import os
import os.path as osp
import time
import numpy as np
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

    fpkl = '/tigress/jk11/SF-CLOUD/pickles/sf-cloud-tests-new.p'
    models = dict(
        M1E5S0100_PHRP='/tigress/jk11/SF-CLOUD/M1E5S0100-PHRP-A4-B2-S4-N256/',
        M1E5S0200_PHRP='/scratch/gpfs/jk11/SF-CLOUD/M1E5S0200-PHRP-A4-B2-S4-N256/',
        M1E5S0400_PHRP='/scratch/gpfs/jk11/SF-CLOUD/M1E5S0400-PHRP-A4-B2-S4-N256/',
        M1E5S0800_PHRP='/scratch/gpfs/jk11/SF-CLOUD/M1E5S0800-PHRP-A4-B2-S4-N256/',
        M1E5S1600_PHRP='/scratch/gpfs/jk11/SF-CLOUD/M1E5S1600-PHRP-A4-B2-S4-N256/',
    )

    sa, df = load_all_sf_cloud(models, fpkl=fpkl, force_override=True)

    models = sa.models
    # models = ['M1E5S0100_WN_redV5']

    # name = models[0]
    for mdl in models:
        print(mdl)
        s = sa.set_model(mdl)
        # nums = range(0, s.get_num_max_virial())
        nums = s.nums[0::1]
        #nums = range(0,1000,1)

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

            # print('read_virial', end=' ')
            # res = s.read_virial(num, force_override=True)
            # n = gc.collect()

            # print('Unreachable objects:', n, end=' ')
            # print('Remaining Garbage:', end=' ')
            # pprint.pprint(gc.garbage)

            # print('read_virial2', end=' ')
            # res = s.read_virial2(num, force_override=True)
            # n = gc.collect()

            # print('Unreachable objects:', n, end=' ')
            # print('Remaining Garbage:', end=' ')
            # pprint.pprint(gc.garbage)

            # print('read_outflow', end=' ')
            # of = s.read_outflow(num, force_override=True)
            # n = gc.collect()

            #print('Unreachable objects:', n, end=' ')
            #print('Remaining Garbage:', end=' ')
            #pprint.pprint(gc.garbage)

            # print('read_density_pdf', end=' ')
            # res = s.read_density_pdf(num, force_override=True)
            # n = gc.collect()

            # print('Unreachable objects:', n, end=' ')
            # print('Remaining Garbage:', end=' ')
            # pprint.pprint(gc.garbage)


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

            # print('plt_snapshot_2panel')
            # fig = s.plt_snapshot_2panel(num, name=name)
            # plt.close(fig)

            print('plt_snapshot_combined')
            fig,d,dd = s.plt_snapshot_combined(num, zoom=1.0, savfig=True);
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
            print('# Do tasks model: {0:s}'.format(mdl))
            print('# Execution time [sec]: {:.1f}'.format(time.time()-time0))
            print('################################################')
            print('')
