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
from pyathena.util.split_container import split_container
from pyathena.plt_tools.make_movie import make_movie

if __name__ == '__main__':
    COMM = MPI.COMM_WORLD

    basedir_def = '/tigress/changgoo/TIGRESS-NCR/R8s_4pc_NCR'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--basedir', type=str,
                        default=basedir_def,
                        help='Name of the basedir.')
    parser.add_argument('-r', '--redraw',
                    action='store_true', default=False,
                    help='Toggle to redraw files')
    args = vars(parser.parse_args())
    locals().update(args)

    s = pa.LoadSimTIGRESSRT(basedir, verbose=False)
    nums = s.nums
    
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
        # prj = s.read_prj(num, force_override=False)
        # slc = s.read_slc(num, force_override=False)
        savdir = osp.join(s.savdir, 'snapshot')
        savname = osp.join(savdir, '{0:s}_{1:04d}.png'.format(s.basename, num))
        if (not osp.isfile(savname)) or redraw:
            try:
                fig = s.plt_snapshot(num)
                # fig = s.plt_pdf2d_all(num)
            except KeyError:
                fig = s.plt_snapshot(num, force_override=True)
                # fig = s.plt_pdf2d_all(num, force_override=True)
            plt.close(fig)

            n = gc.collect()
            print('Unreachable objects:', n, end=' ')
            print('Remaining Garbage:', end=' ')
            pprint.pprint(gc.garbage)

    COMM.barrier()
    savdir = osp.join(s.savdir, 'movies')
    if not osp.exists(savdir):
        os.makedirs(savdir)
    # Make movies
    if COMM.rank == 0:
        fin = osp.join(s.basedir, 'snapshot/*.png')
        fout = osp.join(s.basedir, 'movies/{0:s}_snapshots.mp4'.format(s.basename))
        make_movie(fin, fout, fps_in=15, fps_out=15)
        from shutil import copyfile
        copyfile(fout, osp.join('/tigress/changgoo/public_html/temporary_movies/TIGRESS-NCR',
                                osp.basename(fout)))
    COMM.barrier()
    if COMM.rank == 0:
        print('')
        print('################################################')
        print('# Do tasks')
        print('# Execution time [sec]: {:.1f}'.format(time.time()-time0))
        print('################################################')
        print('')
