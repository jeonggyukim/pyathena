"""
Read athena++ hdf5 file
"""

import xarray as xr
import numpy as np

from .athena_read import athdf

def read_hdf5(filename):
    """Read Athena hdf5 file and convert it to xarray Dataset

    Parameters
    ----------
    filename : str
        data filename

    Returns
    -------
    xarray.Dataset
    """
    ds = athdf(filename)

    # Convert to xarray object
    varnames = set(map(lambda x: x.decode('ASCII'), ds['VariableNames']))
    variables = [(['z', 'y', 'x'], ds[varname]) for varname in varnames]
    attr_keys = (set(ds.keys()) - varnames
                 - {'VariableNames', 'x1f', 'x2f', 'x3f', 'x1v', 'x2v', 'x3v'})
    attrs = {attr_key: ds[attr_key] for attr_key in attr_keys}
    for xr_key, ar_key in zip(['dx', 'dy', 'dz'], ['x1f', 'x2f', 'x3f']):
        dx = np.unique(np.diff(ds[ar_key])).squeeze()
        if dx.size == 1:
            dx = dx[()]
        attrs[xr_key] = dx
    ds = xr.Dataset(
        data_vars=dict(zip(varnames, variables)),
        coords=dict(x=ds['x1v'], y=ds['x2v'], z=ds['x3v']),
        attrs=attrs
    )
    return ds
