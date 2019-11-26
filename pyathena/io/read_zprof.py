"""
Read athena zprof file using pandas and xarray
"""

from __future__ import print_function

import os
import os.path as osp
import glob
import numpy as np
import pandas as pd
import xarray as xr

def read_zprof_all(dirname, problem_id, phase='whole', force_override=False):
    """Function to read all zprof files in directory and make a Dataset object 
    and write to a NetCDF file.

    Note: An xarray DataArray holds a single multi-dimensional variable and its
    coordinates, while a xarray Dataset holds multiple variables that
    potentially share the same coordinates.

    Parameters
    ----------
    dirname : str
        Name of the directory where zprof files are located
    problem_id : str
        Prefix of zprof files
    phase : str
        Name of thermal phase
        ex) whole, phase1, ..., phase5 (cold, intermediate, warm, hot1, hot2)
    force_override : bool
        Flag to force read of hst file even when pickle exists

    Returns
    -------
       ds: xarray dataset

    """

    # Find all files with "/dirname/problem_id.xxxx.phase.zprof"    
    fname_base = '{0:s}.????.{1:s}.zprof'.format(problem_id, phase)
    fnames = sorted(glob.glob(osp.join(dirname, fname_base)))
    
    fnetcdf = '{0:s}.{1:s}.zprof.nc'.format(problem_id, phase)
    fnetcdf = osp.join(dirname, fnetcdf)

    # Check if netcdf file exists and compare last modified times
    mtime_max = np.array([osp.getmtime(fname) for fname in fnames]).max()
    if not force_override and osp.exists(fnetcdf) and \
        osp.getmtime(fnetcdf) > mtime_max:
        da = xr.open_dataset(fnetcdf)
        return da
    
    # If here, need to create a new dataarray
    time = []
    df_all = []
    for i, fname in enumerate(fnames):
        # Read time
        with open(fname, 'r') as f:
            h = f.readline()
            time.append(float(h[h.rfind('t=') + 2:]))

        # read pickle if exists
        df = read_zprof(fname, force_override=False)
        if i == 0: # save z coordinates
            z = (np.array(df['z'])).astype(float)
        df.drop(columns='z', inplace=True)
        df_all.append(df)

        # For test
        # if i > 10:
        #     break
        
    fields = np.array(df.columns)

    # Combine all data
    # Coordinates: time and z
    time = (np.array(time)).astype(float)
    fields = np.array(df.columns)
    df_all = np.stack(df_all, axis=0)
    data_vars = dict()
    for i, f in enumerate(fields):
        data_vars[f] = (('z', 'time'), df_all[...,i].T)

    ds = xr.Dataset(data_vars, coords=dict(z=z, time=time))
    
    # Somehow overwriting using mode='w' doesn't work..
    if osp.exists(fnetcdf):
        os.remove(fnetcdf)

    try:
        ds.to_netcdf(fnetcdf, mode='w')
    except IOError:
        pass
    
    return ds

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
    df : pandas dataframe
    """

    skiprows = 2

    fpkl = filename + '.p'
    if not force_override and osp.exists(fpkl) and \
       osp.getmtime(fpkl) > osp.getmtime(filename):
        df = pd.read_pickle(fpkl)
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
        df = pd.read_csv(filename, names=vlist, skiprows=skiprows,
                         comment='#', sep=',', engine='python')
        try:
            df.to_pickle(fpkl)
        except IOError:
            pass
        
    return df

class ReadZprofBase:

    def read_zprof(self, phase='all', savdir=None, force_override=False):
        """Wrapper function to read all zprof output

        Parameters
        ----------
        phase : str or list of str
            List of thermal phases to read. Possible phases are
            'c': (phase1, cold, T < 180),
            'u': (phase2, unstable, 180 <= T < 5050),
            'w': (phase3, warm, 5050 <= T < 2e4),
            'h1' (phase4, warm-hot, 2e4 < T < 5e5),
            'h2' (phase5, hot-hot, T >= 5e5)
            'whole' (entire temperature range)
            'h' = h1 + h2
            '2p' = c + u + w
            If 'all', read all phases.
        savdir: str
            Name of the directory where pickled data will be saved.
        force_override: bool
            If True, read all (pickled) zprof profiles and dump in netCDF format.

        Returns
        -------
        zp : dict or xarray DataSet
            Dictionary containing xarray DataSet objects for each phase.
        """

        # Mapping from thermal phase name to file suffix 
        dct = dict(c='phase1',
                   u='phase2',
                   w='phase3',
                   h1='phase4',
                   h2='phase5',
                   whole='whole')
        
        if phase == 'all':
            phase = list(dct.keys()) + ['h', '2p']
        else:
            phase = np.atleast_1d(phase)

        zp = dict()
        for ph in phase:
            if ph == 'h':
                zp[ph] = \
                self._read_zprof(phase=dct['h1'], savdir=savdir,
                                 force_override=force_override) + \
                self._read_zprof(phase=dct['h2'], savdir=savdir,
                                 force_override=force_override)
            elif ph == '2p':
                zp[ph] = \
                self._read_zprof(phase=dct['c'], savdir=savdir,
                                 force_override=force_override) + \
                self._read_zprof(phase=dct['u'], savdir=savdir,
                                 force_override=force_override) + \
                self._read_zprof(phase=dct['w'], savdir=savdir,
                                 force_override=force_override)
            else:
                zp[ph] = self._read_zprof(phase=dct[ph], savdir=savdir,
                                          force_override=force_override)

        if len(phase) == 1:
            self.zp = zp[ph]
        else:
            self.zp = zp

        return self.zp
