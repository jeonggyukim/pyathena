"""
Read athena++ hdf5 file
"""

import xarray as xr
import numpy as np
import h5py

from .athena_read import athdf

def read_hdf5(filename, header_only=False, **kwargs):
    """Read Athena hdf5 file and convert it to xarray Dataset

    Parameters
    ----------
    filename : str
        Data filename
    header_only : bool
        Flag to read only attributes, not data.
    **kwargs : dict, optional
        Extra arguments passed to athdf. Refer to athdf documentation for
        a list of all possible arguments.

    Returns
    -------
    ds : xarray.Dataset
        Fluid data

    See Also
    --------
    io.athena_read.athdf
    load_sim.LoadSim.load_hdf5

    Examples
    --------
    >>> from pyathena.io import read_hdf5
    >>> ds = read_hdf5("/path/to/hdf/file")

    >>> from pyathena.load_sim import LoadSim
    >>> s = LoadSim("/path/to/basedir")
    >>> ds = read_hdf5(s.files['hdf5']['prim'][30])
    """
    if header_only:
        with h5py.File(filename, 'r') as f:
            data = {}
            for key in f.attrs:
                data[str(key)] = f.attrs[key]
            return data

    ds = athdf(filename, **kwargs)

    # Convert to xarray object
    possibilities = set(map(lambda x: x.decode('ASCII'), ds['VariableNames']))
    varnames = {var for var in possibilities if var in ds}
    variables = [(['z', 'y', 'x'], ds[varname]) for varname in varnames]
    attr_keys = (set(ds.keys()) - varnames
                 - {'VariableNames','x1f','x2f','x3f','x1v','x2v','x3v'})
    attrs = {attr_key:ds[attr_key] for attr_key in attr_keys}

    # If uniform grid, store cell spacing.
    if ds['MaxLevel'] == 0:
        attrs['dx1'] = np.diff(ds['RootGridX1'])[0] / ds['RootGridSize'][0]
        attrs['dx2'] = np.diff(ds['RootGridX2'])[0] / ds['RootGridSize'][1]
        attrs['dx3'] = np.diff(ds['RootGridX3'])[0] / ds['RootGridSize'][2]
    ds = xr.Dataset(
        data_vars=dict(zip(varnames, variables)),
        coords=dict(x=ds['x1v'], y=ds['x2v'], z=ds['x3v']),
        attrs=attrs
    )
    return ds
