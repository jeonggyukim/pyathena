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

from pyathena.util.split_container import split_container
from pyathena.tigress_ncr.rad_load_all import load_sim_ncr_rad_all

if __name__ == '__main__':
    COMM = MPI.COMM_WORLD

    flag_zprof_from_vtk = False
    flag_slice = True

    force_override = True
    # sa, df = load_sim_ncr_rad_all(model_set='lowZ', zprof_summary=False,
    #                               verbose=False)
    # models = sa.models
    # models = ['S150_Om200_Z1r', 'S150_Om200_Z1']

    sa, df = load_sim_ncr_rad_all(model_set='radiation_paper', zprof_summary=False,
                                  verbose=False)
    models = sa.models
    models = ['R8_4pc', 'LGR4_2pc']
    # models = ['LGR4_2pc']

    for mdl in models:
        nums = df.loc[mdl]['nums']
        s = sa.simdict[mdl]

        if COMM.rank == 0:
            print('basedir, nums', s.basedir, nums)
            nums = split_container(nums, COMM.size)
        else:
            nums = None

        mynums = COMM.scatter(nums, root=0)
        print('[rank, mynums]:', COMM.rank, mynums)

        time0 = time.time()
        if flag_zprof_from_vtk:
            for phase_set_name in ['default_rad','warm_eq_LyC_ma']:
                for num in mynums:
                    print(num, end=' ')
                    rr = s.read_zprof_from_vtk(num, phase_set_name=phase_set_name,
                                               force_override=force_override)
                    n = gc.collect()
                    print('Unreachable objects:', n, end=' ')
                    print('Remaining Garbage:', end=' ')
                    pprint.pprint(gc.garbage)

        if flag_slice:
            for num in mynums:
                print(num, end=' ')
                rr = s.read_slice(num, force_override=force_override)

        COMM.barrier()
        if COMM.rank == 0:
            print('')
            print('################################################')
            print('# Do tasks')
            print('# Execution time [sec]: {:.1f}'.format(time.time()-time0))
            print('################################################')
            print('')
