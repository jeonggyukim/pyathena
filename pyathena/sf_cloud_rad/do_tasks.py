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

    sa, r = load_all_alphabeta()

    # # alpha series
    # seed = 4
    # models = list(r[(r['seed'] == seed) & (r['mu'] == 2.0) & (r['mu'] != np.inf)].index)
    # models.remove('B2S{0:d}'.format(seed))
    # print(models)

    # # beta series
    # seed = 4
    # models = list(r[(r['seed'] == seed) & (r['alpha_vir'] == 2.0) & (r['mu'] != np.inf)].index)
    # models.remove('A2S{0:d}'.format(seed))
    # print(models)

    # models = list(r[(r['seed'] == 4) & (r['mu'] == 2.0)].index)

    seed = 1
    sstr = r'S{0:d}'.format(seed)
    #models = ['B2'+sstr, 'B05'+sstr, 'B8'+sstr, 'B1'+sstr, 'B4'+sstr] # beta only
    models = ['Binf'+sstr]
    # models = ['A5'+sstr, 'A1'+sstr, 'A4'+sstr, 'A3'+sstr] # alpha only except for fiducial
    # models = ['B2'+sstr, 'B8'+sstr, 'B1'+sstr, 'B05'+sstr, 'B4'+sstr,
    #           'A1'+sstr, 'A5'+sstr, 'A4'+sstr, 'A3'+sstr] # alpha and beta

    # models = ['B2S4_N512']
    # models = ['A1S4', 'A4S4', 'A3S4', 'A5S4']

    # models = ['B2S1', 'B2S2', 'B2S3', 'B2S5']
    # models = ['B2S1_N128','B2S2_N128','B2S3_N128','B2S4_N128','B2S5_N128']

    #models = ['B2S4','A5S4','B05S4']
    #models = ['B2S4']
    # models = ['A5S4']
    #models = ['B05S4']

    # sa = pa.LoadSimSFCloudRadAll(dict(B2S4_N128='/perseus/scratch/gpfs/jk11/GMC/M1E5R20.R.B2.A2.S4.N128.again/'))
    # models = sa.models

    # models = dict(nofb='/perseus/scratch/gpfs/jk11/GMC/M1E5R20.NOFB.B2.A2.S4.N256/')
    # sa = pa.LoadSimSFCloudRadAll(models)
    # models = sa.models

    # For snapshot_2panels
    #models = ['B2S4_N512']
    models = ['B2S4']
    #name = 'A2B2S4_N512'
    name = r'$(\alpha_{\rm vir,0},\,\mu_{\Phi,0})=(2,2)$'
    for mdl in models:
        print(mdl)
        s = sa.set_model(mdl)
        # nums = range(0, s.get_num_max_virial())
        # nums = s.nums[::]
        nums = range(0,1000,1)

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
            fig = s.plt_snapshot_2panel(num, name=name)
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
