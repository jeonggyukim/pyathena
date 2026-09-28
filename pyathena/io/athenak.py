import re
from pathlib import Path

import xarray as xr

from . import bin_convert
import numpy as np

_RE_HEADER = re.compile(rb'# AthenaK particle data at time=\s*(\S+)\s+nranks=\s*(\S+)\s+cycle=\s*(\S+)\s+variables=(\S+)')
_RE_POINTS = re.compile(rb'POINTS\s+(\d+)\s+float', re.IGNORECASE)
_RE_SCALARS = re.compile(rb'SCALARS\s+(\S+)\s+(\S+)', re.IGNORECASE)

def read_particle_vtk(filename):
    """
    Read legacy VTK particle file

    Parameters
    ----------
    filename : str or Path
        Path to the VTK particle file

    Returns
    -------
    dict
        Dictionary containing particle data
    """
    with open(filename, 'rb') as f:
        while True:
            line = f.readline()
            if not line:
                raise EOFError("No header found")
            match = _RE_HEADER.match(line)
            if match:
                time, nranks, cycle, variables = match.groups()
                time = float(time)
                nranks, cycle = map(int, (nranks, cycle))
                variables = variables.decode('ascii').split(',')
                break


        while True:
            line = f.readline()
            if not line:
                raise EOFError("No POINTS found")
            match = _RE_POINTS.match(line)
            if match:
                nparticles = int(match.group(1))
                break

        # Read positions
        positions = np.fromfile(f, dtype='>f4', count=nparticles*3)
        positions = positions.reshape((nparticles, 3))

        scalars = dict()
        vtk_to_numpy_dtype = {
            "float": ">f4",
            # add more if needed
            # Note, however, that legacy VTK supports only a limited set of types
        }
        while True:
            line = f.readline()
            if not line:
                break # EOF
            match = _RE_SCALARS.match(line)
            if match:
                varname, dtype = match.groups()
                varname = varname.decode('ascii')
                dtype = dtype.decode('ascii')
                f.readline()  # Skip LOOKUP_TABLE line
                # Read gid
                scalars[varname] = np.fromfile(f, dtype=vtk_to_numpy_dtype[dtype],
                                               count=nparticles)
        if not scalars:
            raise EOFError("No SCALARS found in file")

    res = dict(time=time,
               nranks=nranks,
               cycle=cycle,
               variables=variables,
               nparticles=nparticles,
               x=positions[:, 0].astype(np.float32),
               y=positions[:, 1].astype(np.float32),
               z=positions[:, 2].astype(np.float32),
               gid=scalars['gid'].astype(np.int64),
               ptag=scalars['ptag'].astype(np.int64))
    return res


def read_binary(filename):
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

    Returns
    -------
    xarray.Dataset
        All stored fields, Cartesian coordinates and mesh/header attributes.
    """
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
