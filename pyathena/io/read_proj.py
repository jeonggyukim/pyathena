"""Read AthenaK full-box projections in code units."""
from pathlib import Path
import os

import numpy as np
import xarray as xr


def _read_line(stream):
    line = stream.readline()
    if not line.endswith(b'\n'):
        raise ValueError('Truncated projection text header')
    try:
        return line.decode('ascii').strip()
    except UnicodeDecodeError as error:
        raise ValueError('Projection text header is not ASCII') from error


def _read_value(stream, key, dtype=int):
    name, separator, value = _read_line(stream).partition('=')
    if not separator or name != key:
        raise ValueError(f'Expected projection header entry {key!r}')
    return dtype(value)


def read_proj(filename, axis='z', quantities=None, header_only=False):
    """Read version-1.0 AthenaK projections into an eager xarray Dataset.

    The selected line of sight gives (z, y), (z, x), or (y, x) maps for
    x, y, or z. Stored velocity and magnetic fields are mass-weighted means.
    Values retain their float32/float64 precision and code units. Only the
    requested fields are read; header-only reads return attributes alone.
    """
    dimensions = {'x': ('z', 'y'), 'y': ('z', 'x'), 'z': ('y', 'x')}
    if axis not in dimensions:
        raise ValueError(f'Invalid projection axis {axis!r}; choose x, y, or z')
    path = Path(filename)
    with path.open('rb') as stream:
        if _read_line(stream) != 'Athena projection output version=1.0':
            raise ValueError('Expected Athena projection output version=1.0')
        if _read_value(stream, 'size of preheader') != 11:
            raise ValueError('Expected projection preheader size 11')
        attrs = dict(format_version='1.0', axis=axis, path=str(path))
        for key, name, dtype in (
                ('time', 'time', float), ('cycle', 'cycle', int),
                ('output number', 'num', int),
                ('size of variable', 'size_of_variable', int),
                ('byte order', 'byte_order', str)):
            attrs[name] = _read_value(stream, key, dtype)
        if _read_value(stream, 'number of axes') != 3:
            raise ValueError('Expected three projection axes')
        for key, name, dtype in (
                ('root grid size', 'root_grid_size', int),
                ('domain left edge', 'domain_left_edge', float),
                ('domain right edge', 'domain_right_edge', float)):
            attrs[name] = list(map(dtype, _read_value(stream, key, str).split()))
            if len(attrs[name]) != 3:
                raise ValueError(f'Expected three values for {key}')
        attrs['feature_flags'] = _read_value(stream, 'feature flags')
        nvariable = _read_value(stream, 'number of variables')
        names_line = _read_line(stream)
        if not names_line.startswith('variables:'):
            raise ValueError('Expected projection variable names')
        names = names_line[len('variables:'):].split()
        if len(names) != nvariable or len(set(names)) != nvariable or nvariable < 1:
            raise ValueError('Projection variable names must match count and be unique')
        attrs['variables'] = names
        header_size = _read_value(stream, 'header offset')
        if header_size < 0:
            raise ValueError('Invalid projection parameter-dump size')
        header = stream.read(header_size)
        if len(header) != header_size:
            raise ValueError('Truncated projection parameter dump')
        attrs['header'] = header.decode('ascii')
        counts = dict(zip('xyz', attrs['root_grid_size']))
        if min(counts.values()) < 1:
            raise ValueError('Projection grid sizes must be positive')
        sizes = {view: counts[slow]*counts[fast]
                 for view, (slow, fast) in dimensions.items()}
        itemsize = attrs['size_of_variable']
        if itemsize not in (4, 8) or attrs['byte_order'] not in ('little', 'big'):
            raise ValueError('Unsupported projection precision or byte order')
        dtype = np.dtype(('<' if attrs['byte_order'] == 'little' else '>') + f'f{itemsize}')
        payload_start = stream.tell()
        expected_size = payload_start + itemsize*nvariable*sum(sizes.values())
        if os.fstat(stream.fileno()).st_size != expected_size:
            raise ValueError(f'Incorrect projection file size; expected {expected_size} bytes')
        fields = names if quantities is None else quantities
        if isinstance(fields, str):
            fields = [fields]
        fields = list(fields)
        for field in fields:
            if field not in names:
                raise KeyError(f'Projection field {field!r} is not present')
        if header_only:
            return xr.Dataset(attrs=attrs)
        slow, fast = dimensions[axis]
        axis_offset = sum(sizes[view] for view in 'xyz'[:'xyz'.index(axis)])*nvariable
        data = {}
        for field in fields:
            offset = axis_offset + names.index(field)*sizes[axis]
            stream.seek(payload_start + itemsize*offset)
            values = np.fromfile(stream, dtype=dtype, count=sizes[axis])
            if values.size != sizes[axis]:
                raise ValueError(f'Truncated projection field {field!r}')
            data[field] = ((slow, fast), values.reshape(counts[slow], counts[fast]))
    coords = {}
    for dim in (slow, fast):
        index = 'xyz'.index(dim)
        left, right = attrs['domain_left_edge'][index], attrs['domain_right_edge'][index]
        coords[dim] = left + (np.arange(counts[dim]) + 0.5)*(right-left)/counts[dim]
    return xr.Dataset(data, coords=coords, attrs=attrs)
