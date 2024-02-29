"""
Read athena history file using pandas
"""

from __future__ import print_function

import os
import re
import numpy as np
import pandas as pd

def read_hst(filename, force_override=False, verbose=False, incremental=True):
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
                          comment='#', delim_whitespace=True, engine='python')
        try:
            hst.to_pickle(fpkl)
        except IOError:
            pass

    if incremental:
        hst = correct_restart_hst(hst, verbose=verbose)

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


def correct_restart_hst(h, verbose=True):
    idx = np.where(h.time.diff() < 0)[0]
    n_discont = len(idx)
    if verbose and n_discont > 0:
        print('[read_hst]: found {:d} overlapped time ranges'.format(n_discont))
        print('[read_hst]: cut out overlapped time ranges and make the history ' +\
              'incremental')

    while n_discont > 0:
        i = idx[0]
        t0 = h.iloc[i].time
        h_good1 = h.iloc[np.where(h.iloc[:i-1].time < t0)[0]]
        h_good2 = h.iloc[i:]
        h_good = pd.concat([h_good1, h_good2], ignore_index=True)
        h = h_good
        idx = np.where(h.time.diff()< 0)[0]
        n_discont = len(idx)

    return h
