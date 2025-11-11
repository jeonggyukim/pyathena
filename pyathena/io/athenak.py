import re
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
