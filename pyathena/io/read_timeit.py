"""
Read athena log file timeit.txt using pandas
"""

from __future__ import print_function

import os
import re
import numpy as np
import pandas as pd

def read_timeit(filename, force_override=False, verbose=False):
    """ Function to read timeit log file and pickle

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

    skiprows = 1

    fpkl = filename + '.p'
    if not force_override and os.path.exists(fpkl) and \
       os.path.getmtime(fpkl) > os.path.getmtime(filename):
        tmit = pd.read_pickle(fpkl)
        if verbose:
            print('[read_timeit]: reading from existing pickle.')
    else:
        if verbose:
            print('[read_timeit]: pickle does not exist or timeit file updated.' + \
                      ' Reading {0:s}'.format(filename))
        vlist = _get_timeit_var(filename)

        # c engine does not support regex separators
        tmit = pd.read_csv(filename, names=vlist, skiprows=skiprows,
                           comment='#', delim_whitespace=True, engine='python')
        try:
            tmit.to_pickle(fpkl)
        except IOError:
            pass

    return tmit


def _get_timeit_var(filename):
    """Read variable names from timeit log file

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
        h = f.readline()

    return h.split()
