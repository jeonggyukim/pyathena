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

import pyathena as pa
from pyathena.util.split_container import split_container
from pyathena.plt_tools.make_movie import make_movie

if __name__ == "__main__":

    movie = True
    COMM = MPI.COMM_WORLD

    basedir_def = "/tigress/changgoo/TIGRESS-NCR/R8_8pc_NCR_noSN"

    savdir = None
    savdir_pkl = None

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--basedir", type=str, default=basedir_def, help="Name of the basedir."
    )
    args = vars(parser.parse_args())
    locals().update(args)

    #s = pa.LoadSimTIGRESSNCR(basedir, verbose=False)

    basename = os.path.basename(basedir)
    # Make movies
    if COMM.rank == 0 and movie:
        fin = osp.join(basedir, "volume/time_rotation*.png")
        fout = osp.join(basedir, "{0:s}_time_rotation.mp4".format(basename))
        make_movie(fin, fout, fps_in=15, fps_out=15)
        from shutil import copyfile

        copyfile(
            fout,
            osp.join(
                "/tigress/changgoo/public_html/temporary_movies/SNTI",
                osp.basename(fout),
            ),
        )


        #fin = osp.join(s.basedir, "xray/profile/*.png")
        #fout = osp.join(s.basedir, "movies/{0:s}_xray_profile.mp4".format(s.basename))
        #make_movie(fin, fout, fps_in=15, fps_out=15)
        #from shutil import copyfile
#
#        copyfile(
#            fout,
#            osp.join(
#                "/tigress/changgoo/public_html/temporary_movies/TIGRESS-NCR",
#                osp.basename(fout),
#            ),
#        )
#
#        fin = osp.join(s.basedir, "xray/figure/*.png")
#        fout = osp.join(s.basedir, "movies/{0:s}_xray.mp4".format(s.basename))
#        make_movie(fin, fout, fps_in=15, fps_out=15)
#        from shutil import copyfile
#
#        copyfile(
#            fout,
#            osp.join(
#                "/tigress/changgoo/public_html/temporary_movies/TIGRESS-NCR",
#                osp.basename(fout),
#            ),
#        )
