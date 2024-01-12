#!/usr/bin/env python

import os
import os.path as osp
import time
import gc
from mpi4py import MPI

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pprint

import pyathena as pa
from ..util.split_container import split_container

from .load_sim_sf_cloud import load_all_alphabeta

if __name__ == '__main__':

    COMM = MPI.COMM_WORLD

    # prefix = 'B2S4'
    # models = dict(
    #     S1='/tigress/jk11/GMC/M1E5R20.R.B2.A1.S4.N256',
    #     S2='/tigress/jk11/GMC/M1E5R20.R.B2.A2.S4.N256',
    #     S3='/tigress/jk11/GMC/M1E5R20.R.B2.A3.S4.N256',
    #     S4='/tigress/jk11/GMC/M1E5R20.R.B2.A4.S4.N256',
    #     S5='/tigress/jk11/GMC/M1E5R20.R.B2.A5.S4.N256')

    # Alternative way
    # prefix = 'B2A2'
    # models = ['B2S1','B2S2','B2S3','B2S4','B2S5']

    # import argparse
    # parser = argparse.ArgumentParser()

    # parser.add_argument('-p', '--prefix', type=str, default='A5B2')
    # args = vars(parser.parse_args())
    # locals().update(args)

    # models_dict = {
    #     'A2S1': ['B05S1','B1S1','B2S1','B4S1','B8S1'],
    #     'A2S2': ['B05S2','B1S2','B2S2','B4S2','B8S2'],
    #     'A2S3': ['B05S3','B1S3','B2S3','B4S3','B8S3'],
    #     'A2S4': ['B05S4','B1S4','B2S4','B4S4','B8S4'],
    #     'A2S5': ['B05S5','B1S5','B2S5','B4S5','B8S5'],
    #     'B05A2': ['B05S1','B05S2','B05S3','B05S4','B05S5'],
    #     'B1A2': ['B1S1','B1S2','B1S3','B1S4','B1S5'],
    #     'B2A2': ['B2S1','B2S2','B2S3','B2S4','B2S5'],
    #     'B4A2': ['B4S1','B4S2','B4S3','B4S4','B4S5'],
    #     'B8A2': ['B8S1','B8S2','B8S3','B8S4','B8S5'],
    #     'B2S1': ['A1S1','A2S1','A3S1','A4S1','A5S1'],
    #     'B2S2': ['A1S2','A2S2','A3S2','A4S2','A5S2'],
    #     'B2S3': ['A1S3','A2S3','A3S3','A4S3','A5S3'],
    #     'B2S4': ['A1S4','A2S4','A3S4','A4S4','A5S4'],
    #     'A5B2': ['A5S1','A5S2','A5S3','A5S4','A5S5'],
    # }

    # models = models_dict[prefix]

    # _, r = load_all_alphabeta()
    # models = dict(r.loc[models]['basedir'])

    # sa = pa.LoadSimSFCloudAll(models)

    models = dict(
        #M1E5R20_R='/tigress/jk11/GMC/M1E5R20.R.B2.A2.S4.N256',
        M1E6R60_R='/scratch/gpfs/jk11/GMC/M1E6R60.R.B2.A2.S4.N256.test/',
        M1E6R60_Rfftp='/scratch/gpfs/jk11/GMC/M1E6R60.R.B2.A2.S4.N256.test_fft',
        M1E6R60_RS='/scratch/gpfs/jk11/GMC/M1E6R60.RS.B2.A2.S4.N256.test.again',
        M1E6R60_RW='/scratch/gpfs/jk11/GMC/M1E6R20.RW.B2.A2.S4.N256.test',
    )

    sa = pa.LoadSimSFCloudAll(models)

    models = ['M1E6R60_R','M1E6R60_Rfftp', 'M1E6R60_RS', 'M1E6R60_RW']
    labels = ['R', 'Rfftp', 'RS', 'RW']
    prefix = 'M1E6R60'

    print(models)

    # num_max = 0
    # for mdl in sa.models:
    #     s = sa.set_model(sa.models[0])
    #     num_max = max(num_max, max(s.nums))

    # nums = range(0,num_max*10,10)

    nums = range(0,2001,2)

    if COMM.rank == 0:
        print('nums', nums)
        nums = split_container(nums, COMM.size)
    else:
        nums = None

    mynums = COMM.scatter(nums, root=0)
    print('[rank, mynums]:', COMM.rank, mynums)

    time0 = time.time()
    for num in mynums:
        print(num, end=' ')
        fig = sa.comp_snapshot(models, num, labels=labels, prefix=prefix, savefig=True)
        plt.close(fig)

    COMM.barrier()
    if COMM.rank == 0:
        print('')
        print('################################################')
        print('# Do tasks')
        print('# Execution time [sec]: {:.1f}'.format(time.time()-time0))
        print('################################################')
        print('')
