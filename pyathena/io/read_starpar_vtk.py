"""
Read athena starpar vtk file using pandas, dictionary
"""

from __future__ import print_function

import os
import struct
import pandas as pd
import numpy as np

def _parse_starpar_vtk_line(spl, grid):
    if b"vtk" in spl:
        grid['vtk_version'] = spl[-1]
    elif b"time=" in spl:
        time_index = spl.index(b"time=")
        grid['time'] = float(spl[time_index+1])
    elif b"POINTS" in spl:
        grid['nstars'] = int(spl[1])
    elif b"SCALARS" in spl:
        field = spl[1]
        grid['read_field'] = field
        grid['read_type'] = 'scalar'
        grid['data_type'] = spl[-1]
    elif b"VECTORS" in spl:
        field = spl[1]
        grid['read_field'] = field
        grid['read_type'] = 'vector'
        grid['data_type'] = spl[-1]

def _convert_field_name(name):
    if name == b'star_particle_id':
        return 'id'
    elif name == b'star_particle_mass':
        return 'mass'
    elif name == b'star_particle_age':
        return 'age'
    elif name == b'star_particle_mage':
        return 'mage'
    elif name == b'star_particle_position':
        return 'x'
    elif name == b'star_particle_velocity':
        return 'v'
    elif name == b'star_particle_n_ostar':
        return 'n_ostar'
    elif name == b'star_particle_flag':
        return 'flag'
    elif name == b'star_particle_metal_mass[0]':
        return 'metal_mass[0]'
    elif name == b'star_particle_metal_mass[1]':
        return 'metal_mass[1]'
    elif name == b'star_particle_metal_mass[2]':
        return 'metal_mass[2]'
    elif name == b'star_particle_metal_mass[3]':
        return 'metal_mass[3]'
    elif name == b'star_particle_metal_mass[4]':
        return 'metal_mass[4]'
    elif name == b'star_particle_metal_mass[5]':
        return 'metal_mass[5]'
    elif name == b'star_particle_metal_mass[6]':
        return 'metal_mass[6]'
    elif name == b'star_particle_metal_mass[7]':
        return 'metal_mass[7]'

def read_starpar_vtk(filename, force_override=False, verbose=False):
    """
    Read athena starpar vtk output.
    Returns a dictionary containing mass, position, velocity, age, etc.

    Parameters
    ----------
    filename : string
        Name of the file to open, including extension
    force_override : bool
        Flag to force read of hst file even when pickle exists

    Returns
    -------
    df : dict
        Pandas DataFrame object
    """

    fpkl = filename + '.p'
    if not force_override and os.path.exists(fpkl) and \
       os.path.getmtime(fpkl) > os.path.getmtime(filename):
        hst = pd.read_pickle(fpkl)
        if verbose:
            print('[read_starpar_vtk]: reading from existing pickle.')
    else:
        if verbose:
            print('[read_starpar_vtk]: pickle does not exist or starpar file updated.' + \
                      ' Reading {0:s}'.format(filename))

    # Check for existance of file
    if not os.path.isfile(filename):
        raise IOError('starpar vtk file {0:s} is not found'.format(filename))

    star = {}
    with open(filename, 'rb') as f:
        grid = {}
        line = f.readline()
        # Read time, nstars
        while line:
            spl = line.strip().split()
            _parse_starpar_vtk_line(spl, grid)
            line = f.readline()
            if b"POINT_DATA" in spl:
                break

        time = grid['time']
        nstars = grid['nstars']

        # Read field info
        _field_map = {}
        while line != b'':
            spl = line.strip().split()
            _parse_starpar_vtk_line(spl, grid)
            if b"SCALARS" in spl:
                field = grid['read_field']
                datatype = grid['data_type']
                line = f.readline()  # Read the lookup table line
                _field_map[field] = ('scalar', f.tell(), datatype)
            elif b"VECTORS" in spl:
                field = grid['read_field']
                datatype = grid['data_type']
                _field_map[field] = ('vector', f.tell(), datatype)
            line = f.readline()

        # Read all fields
        for k, v in _field_map.items():
            if v[0] == 'scalar':
                nvar = 1
                shape = [nstars, 1]
            elif v[0]=='vector':
                nvar = 3
                shape = [nstars, 3]
            else:
                raise ValueError('Unknown variable type')

            if v[2] == b'float':
                fmt = '>{}f'.format(nvar*nstars)
            elif v[2] == b'int':
                fmt = '>{}i'.format(nvar*nstars)

            f.seek(v[1])
            size = struct.calcsize(fmt)
            data = np.array(struct.unpack(fmt, f.read(size)))
            name = _convert_field_name(k)
            star[name] = np.transpose(np.reshape(data, shape), (0, 1))
            if nstars > 1:
                star[name] = np.squeeze(star[name])
            elif nstars == 1:
                if v[0] != 'vector':
                    star[name] = star[name][0]

    star['x1'] = star['x'][:,0]
    star['x2'] = star['x'][:,1]
    star['x3'] = star['x'][:,2]
    star['v1'] = star['v'][:,0]
    star['v2'] = star['v'][:,1]
    star['v3'] = star['v'][:,2]
    star.pop('x')
    star.pop('v')

    # Sort id in an ascending order (or age in an descending order)
    if nstars > 1:
        idsrt = star['id'].argsort()
        for k, v in star.items():
            star[k] = v[idsrt]

    # Add time, nstars keys at the end
    try:
        df = pd.DataFrame(star)
    except:
        df = pd.DataFrame(index=star.keys())

    df.time = time
    df.nstars = nstars
    try:
        df.to_pickle(fpkl)
    except IOError:
        pass

    return df
