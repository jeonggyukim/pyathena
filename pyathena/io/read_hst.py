"""
Read athena history file using pandas
"""

from __future__ import print_function

import os
import re
import numpy as np
import pandas as pd

def read_hst(filename, force_override=False, verbose=False):
    """ Function to read athena history file and pickle

    Parameters
    ----------
    filename : string
        Name of the file to open, including extension
    force_override : bool
        Flag to force read of hst file even when pickle exists
    verbose : bool
        Print verbose messages

    Returns
    -------
    hst : pandas dataframe
        Each column contains time series data
    """

    fpkl = filename + '.p'
    if not force_override and os.path.exists(fpkl) and \
       os.path.getmtime(fpkl) > os.path.getmtime(filename):
        hst = pd.read_pickle(fpkl)
        if verbose:
            print('[read_hst]: reading from existing pickle.')
    else:
        if verbose:
            print('[read_hst]: pickle does not exist or hst file updated.' + \
                      ' Reading {0:s}'.format(filename))
        vlist = _get_hst_var(filename)

        # C engine is faster but python engine is currently more feature-complete
        hst = pd.read_csv(filename, names=vlist,
                          comment='#', sep=r'\s+', engine='python')
        try:
            hst.to_pickle(fpkl)
        except IOError:
            pass

    return hst


def _get_hst_var(filename):
    """Read variable names from history file

    Parameters
    ----------
    filename : string
        Name of the file to open, including extension

    Returns
    -------
    vlist : list
        List of variables
    """

    with open(filename, 'r') as f:
        # Skip the first line
        # "Athena history dump for level=.. domain=0 volume=..."
        # or
        # "# Athena++ history data"
        h = f.readline()
        h = f.readline()

    vlist = re.split(r"\[\d+]\=|\n", h)
    for i in range(len(vlist)):
        vlist[i] = re.sub(r"\s|\W", "", vlist[i])

    return vlist[1:-1]
