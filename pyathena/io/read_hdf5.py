"""
Read athena++ hdf5 file
"""

import xarray as xr
import numpy as np
import dask.array as da
import h5py

from .athena_read import athdf


def read_hdf5(filename, header_only=False, chunks=None, raw=False,
              num_ghost=0, **kwargs):
    """Read Athena hdf5 file and convert it to xarray Dataset

    Parameters
    ----------
    filename : str
        Data filename
    header_only : bool
        Flag to read only attributes, not data.
    chunks : (dict or None), default: None
        If provided, used to load the data into dask arrays.
    raw : bool, optional
        If True, return raw data without merging MeshBlocks into a single array.
        Default is False.
    num_ghost : int, optional
        Number of ghost zones to include in the data. Default is 0.
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

    if chunks is not None:
        return read_hdf5_dask(
            filename, (chunks['x'], chunks['y'], chunks['z']),
            num_ghost=num_ghost, raw=raw,
        )
    else:
        if header_only:
            with h5py.File(filename, 'r') as f:
                data = {}
                for key in f.attrs:
                    data[str(key)] = f.attrs[key]
                return data

        ds = athdf(filename, raw=raw, num_ghost=num_ghost, **kwargs)

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

def read_hdf5_dask(filename, chunksize, num_ghost=0, raw=False):
    """Read Athena++ hdf5 file and convert it to dask-xarray Dataset

    In most cases, this function is not needed to be called directly. Instead,
    use `read_hdf5` with `chunks` argument. This function is used internally
    by `read_hdf5` to read data into dask array.

    Parameters
    ----------
    filename : str
        Data filename
    chunksize : tuple of int
        Dask chunk size along (x, y, z) directions.
    num_ghost : int, optional
        Number of ghost zones in the output. Default is 0.
    raw : bool, optional
        If True, do not merge MeshBlocks and return a dask array with the shape
        [nvar, nblocks, z, y, x].
    """
    f = h5py.File(filename, 'r')

    if num_ghost == 0 and np.array(f["x1v"]).min() < f.attrs["RootGridX1"][0]:
        raise RuntimeError('Ghost zones detected but "num_ghost" keyword set to zero.')
    # Read Mesh information
    block_size = f.attrs['MeshBlockSize']
    block_size_noghost = block_size - np.array([num_ghost*2]*3)
    mesh_size = f.attrs['RootGridSize']
    num_blocks = mesh_size // block_size_noghost  # Assuming uniform grid
    varnames = list(map(lambda x: x.decode('ASCII'), f.attrs['VariableNames']))

    if num_blocks.prod() != f.attrs['NumMeshBlocks']:
        raise ValueError("Number of blocks does not match the attribute")

    # Array of logical locations, arranged by Z-ordering
    # (lx1, lx2, lx3)
    # (  0,   0,   0)
    # (  1,   0,   0)
    # (  0,   1,   0)
    # ...
    logical_loc = f['LogicalLocations']

    # Number of MeshBlocks per chunk along each dimension.
    nblock_per_chunk = np.array(chunksize) // block_size_noghost
    chunksize_read = (1, nblock_per_chunk.prod(), *block_size)


    # lazy load from HDF5
    ds = []
    for dsetname in f.attrs['DatasetNames']:
        darr = da.from_array(f[dsetname], chunks=chunksize_read)
        if num_ghost > 0:
            darr = darr[
                :, :, num_ghost:-num_ghost, num_ghost:-num_ghost, num_ghost:-num_ghost
            ]
        if len(darr.shape) != 5:
            # Expected shape: (nvar, nblock, z, y, x)
            raise ValueError("Invalid shape of the dataset")
        ds += [var for var in darr]
    if raw:
        return dict(zip(varnames, ds))

    def _reorder_rechunk(var):
        """
        Loop over the MeshBlocks and place them to correct logical locations
        in 3D Cartesian space. Then, merge them into a single dask array.
        """
        reordered = np.empty(num_blocks[::-1], dtype=object)
        for gid in range(num_blocks.prod()):
            lx1, lx2, lx3 = logical_loc[gid]  # Correct Cartesian coordinates
            reordered[lx3, lx2, lx1] = var[gid, ...]  # Assign the correct block
        # Merge into a single array
        return da.block(reordered.tolist()).rechunk(chunksize)

    # Apply the rechunking function to all variables
    ds = list(map(_reorder_rechunk, ds))

    # Convert to xarray object
    variables = [(['z', 'y', 'x'], d) for d in ds]
    coordnames = ['x1v', 'x2v', 'x3v']

    # Calculate coordinates
    # Borrowed and slightly modified from pyathena.io.athdf
    coords = {}
    for i, (nrbx, xv) in enumerate(zip(num_blocks, coordnames)):
        coords[xv] = np.empty(mesh_size[i])
        for n_block in range(nrbx):
            sample_block = np.where(logical_loc[:, i] == n_block)[0][0]
            index_low = n_block * block_size_noghost[i]
            index_high = index_low + block_size_noghost[i]
            if num_ghost > 0:
                coord_ = f[xv][sample_block, :][num_ghost:-(num_ghost)]
            else:
                coord_ = f[xv][sample_block, :]
            coords[xv][index_low:index_high] = coord_

    # If uniform grid, store cell spacing.
    attrs = dict(f.attrs)
    attrs['dx1'] = np.diff(f.attrs['RootGridX1'])[0] / mesh_size[0]
    attrs['dx2'] = np.diff(f.attrs['RootGridX2'])[0] / mesh_size[1]
    attrs['dx3'] = np.diff(f.attrs['RootGridX3'])[0] / mesh_size[2]

    ds = xr.Dataset(
        data_vars=dict(zip(varnames, variables)),
        coords=dict(x=coords['x1v'], y=coords['x2v'], z=coords['x3v']),
        attrs=attrs
    )

    return ds
