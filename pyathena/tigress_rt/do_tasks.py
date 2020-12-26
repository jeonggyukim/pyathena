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

    try: 
        myid = int(os.environ["SLURM_ARRAY_TASK_ID"])
        narray = 8
    except KeyError:
        myid = 0
        narray = 1

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--basedir', type=str,
                        default='./',
                        help='Name of the basedir.')
    parser.add_argument('-r', '--redraw',
                    action='store_true', default=False,
                    help='Toggle to redraw files')
    args = vars(parser.parse_args())
    locals().update(args)

    s = pa.LoadSimTIGRESSRT(basedir, verbose=False)
    nums = s.nums
    
    print('basedir, nums', s.basedir, nums)
    nums = split_container(nums, narray)

    mynums = nums[myid] 
    print('[rank, mynums]:', myid, mynums)

    for num in mynums:
        print(num, end=' ')

        savdir = osp.join(s.savdir, 'snapshot')
        savname = osp.join(savdir, '{0:s}_{1:04d}.png'.format(s.basename, num))
        if (not osp.isfile(savname)) or redraw:
            try:
                fig = s.plt_snapshot(num)
            except KeyError:
                fig = s.plt_snapshot(num, force_override=True)
            plt.close(fig)

            n = gc.collect()
            print('Unreachable objects:', n, end=' ')
            print('Remaining Garbage:', end=' ')
            pprint.pprint(gc.garbage)

    savdir = osp.join(s.savdir, 'movies')
    if not osp.exists(savdir):
        os.makedirs(savdir)
    # Make movies
    if myid == -1:
        fin = osp.join(s.basedir, 'snapshots/*.png')
        fout = osp.join(s.basedir, 'movies/{0:s}_snapshots.mp4'.format(s.basename))
        make_movie(fin, fout, fps_in=15, fps_out=15)
        from shutil import copyfile
        copyfile(fout, osp.join('/tigress/changgoo/public_html/temporary_movies/TIGRESS-NCR',
                                osp.basename(fout)))
