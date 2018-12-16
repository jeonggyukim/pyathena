"""
Read athena zprof file using pandas and xarray
"""

import os
import glob
import numpy as np
import pandas as pd
import xarray as xr

def read_zprof_all(dirname, problem_id, phase='whole', force_override=False):
    """Read all zprof files and make a DataArray object and write to a NetCDF
    file

    Parameters
    ----------
    dirname : string
        Name of the directory where zprof files are located
    problem_id: string
        Prefix of zprof files
    phase: string
        Name of thermal phase
        ex) whole, phase1, ..., phase5 (cold, intermediate, warm, hot1, hot2)
  
    Returns
    -------
       da: xarray dataarray
    """

    # Find all files with "/dirname/problem_id.xxxx.phase.zprof"    
    fname_base = '{0:s}.????.{1:s}.zprof'.format(problem_id, phase)
    fnames = sorted(glob.glob(os.path.join(dirname, fname_base)))
    
    fnetcdf = '{0:s}.{1:s}.zprof.nc'.format(problem_id, phase)
    fnetcdf = os.path.join(dirname, fnetcdf)

    # check if netcdf file exists and compare last modified times
    mtime_max = np.array([os.path.getmtime(fname) for fname in fnames]).max()
    if not force_override and os.path.exists(fnetcdf) and \
        os.path.getmtime(fnetcdf) > mtime_max:
        da = xr.open_dataarray(fnetcdf)
        return da
    
    # if here, need to create a new dataarray
    time = []
    df_all = []
    for fname in fnames:
        # Read time
        with open(fname, 'r') as f:
            h = f.readline()
            time.append(float(h[h.rfind('t=') + 2:]))

        # read pickle if exists
        df = read_zprof(fname, force_override=False)
        df_all.append(df)

    z = (np.array(df['z'])).astype(float)
    fields = np.array(df.columns)

    # Combine all data
    df_all = np.stack(df_all, axis=0)
    # print df_all.shape
    da = xr.DataArray(df_all.T,
                      coords=dict(fields=fields, z=z, time=time),
                      dims=('fields', 'z', 'time'))

    # Somehow overwriting using mode='w' doesn't work..
    if os.path.exists(fnetcdf):
        os.remove(fnetcdf)

    try:
        da.to_netcdf(fnetcdf, mode='w')
    except IOError:
        pass
    
    return da

def read_zprof(filename, force_override=False, verbose=False):
    """
    Function to read one zprof file and pickle
    
    Parameters
    ----------
    filename : string
        Name of the file to open, including extension
    force_override: bool
        Flag to force read of zprof file even when pickle exists

    Returns
    -------
    zp : pandas dataframe
    """

    skiprows = 2

    fpkl = filename + '.p'
    if not force_override and os.path.exists(fpkl) and \
       os.path.getmtime(fpkl) > os.path.getmtime(filename):
        zp = pd.read_pickle(fpkl)
        if verbose:
            print('[read_zprof]: reading from existing pickle.')
    else:
        if verbose:
            print('[read_zprof]: pickle does not exist or zprof file updated.' + \
                      ' Reading {0:s}'.format(filename))

        with open(filename, 'r') as f:
            # For the moment, skip the first line which contains information about
            # the time at which the file is written
            # "# Athena vertical profile at t=xxx.xx"
            h = f.readline()
            time = float(h[h.rfind('t=') + 2:])
            h = f.readline()
            vlist = h.split(',')
            if vlist[-1].endswith('\n'):
                vlist[-1] = vlist[-1][:-1]    # strip \n

        # c engine does not support regex separators
        zp = pd.read_table(filename, names=vlist, skiprows=skiprows,
                           comment='#', sep=',', engine='python')
        try:
            zp.to_pickle(fpkl)
        except IOError:
            pass
        
    return zp
