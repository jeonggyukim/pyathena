"""Read native AthenaK mesh binary snapshots."""
import re
from pathlib import Path

import numpy as np
import xarray as xr

from . import bin_convert


def read_bin(filename, header_only=False):
    """Read an AthenaK mesh binary snapshot into an eager xarray Dataset.

    Supports uniform Cartesian, full-volume, active-cell output in one shared
    file. All stored fields retain their names and float32/float64 precision;
    cell-center coordinates are float64 and dimensions are (z, y, x), including
    singleton axes. Refinement, slices, ghost zones, coarsened and per-rank
    output are unsupported. Block data coexist with assembled arrays during
    reading, so peak memory exceeds the final dataset size.

    Parameters
    ----------
    filename : str or pathlib.Path
        AthenaK version-1.1 binary mesh file.
    header_only : bool, optional
        Return Time, NumCycles, VariableNames and Header without reading blocks.

    Returns
    -------
    xarray.Dataset
        All stored fields, Cartesian coordinates and mesh/header attributes.
    """
    if header_only:
        with open(filename, 'rb') as stream:
            if stream.readline().strip() != b'Athena binary output version=1.1':
                raise ValueError('Expected Athena binary output version=1.1')
            count = int(stream.readline().split(b'=')[-1])
            metadata = dict(tuple(part.strip() for part in line.decode('ascii').split('=', 1))
                            for line in (stream.readline() for _ in range(count-1)))
            nvariable = int(stream.readline().split(b'=')[-1])
            variables = stream.readline().decode('ascii').split()[1:]
            if len(variables) != nvariable:
                raise ValueError('Binary variable names do not match count')
            header_size = int(stream.readline().split(b'=')[-1])
            header = stream.read(header_size)
            if len(header) != header_size:
                raise ValueError('Truncated binary parameter dump')
            header = [line.split('#')[0].strip()
                      for line in header.decode('ascii').splitlines()]
            return xr.Dataset(attrs=dict(
                Time=float(metadata['time']), NumCycles=int(metadata['cycle']),
                VariableNames=variables, Header=[line for line in header if line]))
    filedata = bin_convert.read_binary(filename)
    _validate_binary_layout(filedata, filename)
    field_names = filedata['var_names']
    field_dtype = filedata['mb_data'][field_names[0]][0].dtype
    data = bin_convert.read_binary_as_athdf(
        filename, filedata=filedata, dtype=field_dtype,
        coordinate_dtype=np.float64)

    attrs = {key: data[key] for key in ('Time', 'NumCycles', 'MaxLevel')}
    attrs.update(
        Coordinates='cartesian', NumMeshBlocks=filedata['n_mbs'],
        MeshBlockSize=[filedata[f'nx{axis}_mb'] for axis in (1, 2, 3)],
        RootGridSize=[filedata[f'Nx{axis}'] for axis in (1, 2, 3)],
        VariableNames=field_names, Header=filedata['header'])
    for axis in (1, 2, 3):
        xmin, xmax = filedata[f'x{axis}min'], filedata[f'x{axis}max']
        attrs[f'RootGridX{axis}'] = [xmin, xmax, 1.0]
        attrs[f'dx{axis}'] = (xmax - xmin) / filedata[f'Nx{axis}']
    return xr.Dataset(
        data_vars={name: (('z', 'y', 'x'), data[name]) for name in field_names},
        coords={name: data[f'x{axis}v']
                for axis, name in enumerate(('x', 'y', 'z'), 1)},
        attrs=attrs)


def _validate_binary_layout(filedata, filename):
    """Reject unsupported layouts before allocating the assembled grid."""
    parameters = {}
    for line in filedata['header']:
        if line.startswith('<'):
            block = parameters.setdefault(line.strip('<>'), {})
        else:
            key, value = line.split('=', 1)
            block[key.strip()] = value.strip()

    # Header options apply only to the stream that produced this file. A run
    # can also have slices/per-rank streams alongside a supported volume stream.
    basename = parameters.get('job', {}).get('basename', '')
    for name, output in parameters.items():
        if not name.startswith('output'):
            continue
        stream_id = output.get('id', output.get('variable', ''))
        pattern = re.escape(f'{basename}.{stream_id}.') + r'[0-9]+\.bin'
        if not re.fullmatch(pattern, Path(filename).name):
            continue
        if output.get('single_file_per_rank', 'false').lower() in ('true', '1'):
            raise ValueError('Per-rank binary output is unsupported; use a shared file')
        if output.get('file_type') == 'cbin':
            raise ValueError('Coarsened binary output is unsupported')
        if (output.get('ghost_zones', 'false').lower() in ('true', '1')
                or any(f'slice_x{axis}' in output for axis in (1, 2, 3))):
            raise ValueError('Binary reader requires full-volume active cells, without slices or ghost zones')

    root_size = np.array([filedata[f'Nx{axis}'] for axis in (1, 2, 3)])
    block_size = np.array([filedata[f'nx{axis}_mb'] for axis in (1, 2, 3)])
    if (np.any(block_size <= 0) or np.any(root_size <= 0)
            or np.any(root_size % block_size)):
        raise ValueError('Binary reader requires a uniform Cartesian block grid')
    if np.any(filedata['mb_logical'][:, 3] != 0):
        raise ValueError('Refined binary meshes are unsupported')

    # The parser subtracts nghost even on inactive axes (native index zero).
    nghost = int(parameters['mesh']['nghost'])
    active_start = np.where(root_size > 1, 0, -nghost)
    active_start[0] = 0  # AthenaK always gives x1 ghost cells.
    expected_extents = np.column_stack((active_start, active_start + block_size - 1))
    if np.any(filedata['mb_index'] != expected_extents.ravel()):
        raise ValueError('Binary reader requires full active-cell extents; slices, ghost zones and coarsening are unsupported')

    blocks_per_axis = root_size // block_size
    locations = filedata['mb_logical'][:, :3]
    if (filedata['n_mbs'] != np.prod(blocks_per_axis)
            or np.any(locations < 0) or np.any(locations >= blocks_per_axis)
            or len(np.unique(locations, axis=0)) != filedata['n_mbs']):
        raise ValueError('Binary reader requires unique complete block coverage in one shared file (no missing, duplicate or per-rank blocks)')

    # Confirm the block bounds agree with the uniform root grid. Location
    # precision may be lower than field precision, so use its stored epsilon.
    bounds = filedata['mb_geometry']
    epsilon = np.finfo(bounds.dtype).eps
    for axis in range(3):
        xmin = filedata[f'x{axis + 1}min']
        xmax = filedata[f'x{axis + 1}max']
        block_width = (xmax - xmin) / blocks_per_axis[axis]
        expected = xmin + (locations[:, axis, None] + np.array([0, 1])) * block_width
        if (not xmax > xmin or not np.allclose(
                bounds[:, 2*axis:2*axis+2], expected, rtol=8*epsilon,
                atol=8*epsilon*max(abs(xmin), abs(xmax)))):
            raise ValueError('Binary reader requires uniform Cartesian block geometry')
