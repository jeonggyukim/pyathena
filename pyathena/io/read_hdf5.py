"""
Read athena++ hdf5 file
"""

import xarray as xr
import numpy as np

from .athena_read import athdf

def read_hdf5(filename, **kwargs):
    """Read Athena hdf5 file and convert it to xarray Dataset

    Parameters
    ----------
    filename : str
        data filename

    Returns
    -------
    xarray.Dataset
    """
    ds = athdf(filename, **kwargs)

    # Convert to xarray object
    varnames = set(map(lambda x: x.decode('ASCII'), ds['VariableNames']))
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
