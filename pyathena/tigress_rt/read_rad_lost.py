import os
import pandas as pd
import numpy as np
import xarray as xr

def read_rad_lost(filename, force_override=False, verbose=False):
    """
    Function to read rad_lost.txt and pickle
    
    Parameters:
       filename : string
           Name of the file to open, including extension
       force_override: bool
           Flag to force read of rad_lost file even when pickle exists

    Returns:
       df, da : tuple
          (pandas dataframe, xarray dataarray)
    """
    
    fpkl = filename + '.p'
    if not force_override and os.path.exists(fpkl) and \
       os.path.getmtime(fpkl) > os.path.getmtime(filename):
        df = pd.read_pickle(fpkl)
        if verbose:
            print('[read_radiators]: reading from existing pickle.')

    if verbose:
        print('[read_radiators]: pickle does not exist or file updated.' + \
                  ' Reading {0:s}'.format(filename))

    df = pd.read_csv(filename, sep=' ', header=None, skiprows=0)

    # drop nan column (due to space at the end of line in output file)
    df = df.drop(labels=[df.columns[-1]], axis=1)
    col = {0:'time',1:'nfreq',2:'nsrc',3:'N_mu'}
    nfreq = df[1][0]
    N_mu = df[3][0]
    for i in range(4, 4 + nfreq):
        col[i] = 'L_tot{0:d}'.format(i-4)

    df = df.rename(columns=col)

    return df
