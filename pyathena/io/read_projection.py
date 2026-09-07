"""Read Athena++ full-box raw integrals (code units) into Cartesian maps."""
import h5py
import numpy as np
import xarray as xr


def read_projection(filename, axis='z', quantities=None, header_only=False):
    """Return a Dataset with (z,y), (z,x), or (y,x) maps for LOS x, y, or z.

    Fields are Sigma_gas, mom1/2/3 and, for MHD, rhoB1/2/3. These are
    unnormalized integrals. Divide momentum or rhoB maps by Sigma_gas for
    density-weighted means. Header-only reads access attributes, not datasets.
    """
    slow, fast = {'x': ('z', 'y'), 'y': ('z', 'x'), 'z': ('y', 'x')}[axis]
    with h5py.File(filename, 'r') as handle:
        attrs = dict(handle.attrs)
        attrs['time'] = float(attrs.pop('Time')[0])
        attrs['num'] = int(attrs.pop('OutputNumber')[0])
        attrs['FieldNames'] = [name.decode() for name in attrs['FieldNames']]
        attrs['axis'] = axis
        if header_only:
            return xr.Dataset(attrs=attrs)
        group = handle[axis]
        coords = {dim: group[dim][:] for dim in (slow, fast)}
        locations = group['LogicalLocations'][:]
        fields = attrs['FieldNames'] if quantities is None else quantities
        if isinstance(fields, str):
            fields = [fields]
        data = {}
        for field in fields:
            blocks = group[field][:]
            _, nslow, nfast = blocks.shape
            values = np.empty((len(coords[slow]), len(coords[fast])), dtype=blocks.dtype)
            for block, (loc_fast, loc_slow) in zip(blocks, locations):
                values[loc_slow*nslow:(loc_slow+1)*nslow,
                       loc_fast*nfast:(loc_fast+1)*nfast] = block
            data[field] = ((slow, fast), values)
    return xr.Dataset(data, coords=coords, attrs=attrs)
