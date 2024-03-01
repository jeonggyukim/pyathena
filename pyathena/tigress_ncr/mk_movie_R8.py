#!/usr/bin/env python

import os
import os.path as osp
# import gc
# import time
# import matplotlib.pyplot as plt
# import pprint
# import argparse
# import syss

from pyathena.plt_tools.make_movie import make_movie

basedir = "/tigress/changgoo/TIGRESS-R8/combined_slices/"
basedir = "/tigress/changgoo/TIGRESS-NCR/R8_combined_figures/"
basedir = "/tigress/changgoo/TIGRESS-NCR/LGR4_combined_figures/"
if not osp.isdir(osp.join(basedir, "movies")):
    os.mkdir(osp.join(basedir, "movies"))

for field in ["Sigma_gas"]:  # ,'nH','pok','surf']:
    fin = osp.join(basedir, "{0}/{0}.*.png".format(field))
    fout = osp.join(basedir, "movies/{0:s}.mp4".format(field))
    make_movie(fin, fout, fps_in=15, fps_out=15)
    # from shutil import copyfile
