"""Write small fake Athena and Athena++ run directories for tests.

The vtk writers reproduce the byte layout of the simulation codes:
Athena-TIGRESS ``src/dump_vtk.c`` (STRUCTURED_POINTS, one file per MPI
rank under ``id<rank>/``) and Athena++ ``src/outputs/vtk.cpp``
(RECTILINEAR_GRID, one file per meshblock). Both write big-endian float32,
loop k (outer), j, i (inner), and put a newline before every data header
after the first.
"""

import numpy as np

BE_F4 = '>f4'


def cell_index_field(nx, offset=0.0):
    """Return a (nz, ny, nx) array whose value at (k, j, i) is
    offset + 100*k + 10*j + i, so each cell can be identified."""
    k, j, i = np.meshgrid(np.arange(nx[2]), np.arange(nx[1]),
                          np.arange(nx[0]), indexing='ij')
    return (offset + 100*k + 10*j + i).astype(np.float32)


def write_vtk_athena(fname, time, le, dx, fields, prim=True):
    """Write one Athena (C) dump_vtk file.

    fields: ordered dict name -> array of shape (nz, ny, nx) for a scalar or
    (nz, ny, nx, 3) for a vector, in the order dump_vtk.c writes them.
    """
    first = next(iter(fields.values()))
    nz, ny, nx = first.shape[:3]
    kind = 'PRIMITIVE' if prim else 'CONSERVED'
    with open(fname, 'wb') as f:
        f.write(b'# vtk DataFile Version 2.0\n')
        f.write(f'{kind} vars at time= {time:e}, level= 0, domain= 0\n'
                .encode())
        f.write(b'BINARY\n')
        f.write(b'DATASET STRUCTURED_POINTS\n')
        f.write(f'DIMENSIONS {nx+1:d} {ny+1:d} {nz+1:d}\n'.encode())
        f.write('ORIGIN {:e} {:e} {:e} \n'.format(*le).encode())
        f.write('SPACING {:e} {:e} {:e} \n'.format(*dx).encode())
        f.write(f'CELL_DATA {nx*ny*nz:d} \n'.encode())
        for n, (name, arr) in enumerate(fields.items()):
            sep = b'' if n == 0 else b'\n'
            if arr.ndim == 3:
                f.write(sep + f'SCALARS {name} float\n'.encode())
                f.write(b'LOOKUP_TABLE default\n')
            else:
                f.write(sep + f'VECTORS {name} float\n'.encode())
            f.write(np.ascontiguousarray(arr, dtype=BE_F4).tobytes())


def write_vtk_athenapp(fname, time, x1f, x2f, x3f, fields, variable='prim'):
    """Write one Athena++ vtk file for a single meshblock.

    x1f, x2f, x3f: cell-face coordinates (length ncells+1 per direction).
    fields: ordered dict name -> (nz, ny, nx) or (nz, ny, nx, 3) array.
    """
    nx, ny, nz = len(x1f) - 1, len(x2f) - 1, len(x3f) - 1
    with open(fname, 'wb') as f:
        f.write(b'# vtk DataFile Version 2.0\n')
        f.write(f'# Athena++ data at time={time:e}  cycle=0  '
                f'variables={variable} \n'.encode())
        f.write(b'BINARY\n')
        f.write(b'DATASET RECTILINEAR_GRID\n')
        f.write(f'DIMENSIONS {nx+1:d} {ny+1:d} {nz+1:d}\n'.encode())
        for n, (label, xf) in enumerate(
                zip('XYZ', (x1f, x2f, x3f))):
            sep = b'' if n == 0 else b'\n'
            f.write(sep + f'{label}_COORDINATES {len(xf):d} float\n'.encode())
            f.write(np.asarray(xf, dtype=BE_F4).tobytes())
        f.write(f'\nCELL_DATA {nx*ny*nz:d}'.encode())
        for name, arr in fields.items():
            kind = 'SCALARS' if arr.ndim == 3 else 'VECTORS'
            f.write(f'\n{kind} {name} float\n'.encode())
            if arr.ndim == 3:
                f.write(b'LOOKUP_TABLE default\n')
            f.write(np.ascontiguousarray(arr, dtype=BE_F4).tobytes())


ATHINPUT_ATHENA = """\
<comment>
problem         = fake run for pyathena tests

<job>
problem_id      = {pid}
maxout          = 2
num_domains     = 1

<output1>
out_fmt         = hst
dt              = 0.1

<output2>
out_fmt         = vtk
out             = prim
dt              = 1.0

<time>
cour_no         = 0.3
nlim            = -1
tlim            = 2.0

<domain1>
level           = 0
Nx1             = {nx[0]}
x1min           = {le[0]}
x1max           = {re[0]}
bc_ix1          = 4
bc_ox1          = 4
Nx2             = {nx[1]}
x2min           = {le[1]}
x2max           = {re[1]}
bc_ix2          = 4
bc_ox2          = 4
Nx3             = {nx[2]}
x3min           = {le[2]}
x3max           = {re[2]}
bc_ix3          = 2
bc_ox3          = 2
NGrid_x1        = {ngrid[0]}
NGrid_x2        = {ngrid[1]}
NGrid_x3        = {ngrid[2]}

<problem>
gamma           = 1.66666667
muH             = 1.4
"""

ATHINPUT_ATHENAPP = """\
<comment>
problem    = fake run for pyathena tests

<job>
problem_id = {pid}

<output1>
file_type  = hst
dt         = 0.1

<output2>
file_type  = vtk
variable   = prim
dt         = 1.0

<time>
cfl_number = 0.3
nlim       = -1
tlim       = 2.0

<mesh>
nx1        = {nx[0]}
x1min      = {le[0]}
x1max      = {re[0]}
ix1_bc     = periodic
ox1_bc     = periodic
nx2        = {nx[1]}
x2min      = {le[1]}
x2max      = {re[1]}
ix2_bc     = periodic
ox2_bc     = periodic
nx3        = {nx[2]}
x3min      = {le[2]}
x3max      = {re[2]}
ix3_bc     = periodic
ox3_bc     = periodic

<meshblock>
nx1        = {nb[0]}
nx2        = {nb[1]}
nx3        = {nb[2]}

<units>
unit_system = ism
mass_cgs = 4.91615563682836e+31
length_cgs = 3.085678e+18
time_cgs = 30856780000000
mean_mass_per_hydrogen = 2.34262e-24
"""

def write_stdout_log(fname, athinput):
    """Write a stdout log holding the parameter dump between PAR_DUMP
    marker lines, as Athena par_dump() and Athena++ ParameterDump() print
    it at startup."""
    marker = '# --------------------- PAR_DUMP -----------------------\n'
    with open(fname, 'w') as f:
        f.write('Simulation started\n')
        f.write(marker + '\n' + athinput + marker)
        f.write('cycle=0 time=0.0 dt=0.01\n')


HST_COLUMNS = ['time', 'dt', 'mass', 'totalE']


def write_hst(fname, times):
    """Write an Athena-style history file with HST_COLUMNS."""
    header = '   '.join(f'[{i+1}]={c}' for i, c in enumerate(HST_COLUMNS))
    with open(fname, 'w') as f:
        f.write('# Athena history dump for level=0 domain=0 volume=1.0\n')
        f.write(f'#   {header}\n')
        for n, t in enumerate(times):
            f.write(f'{t:.6e} {0.01:.6e} {1.0:.6e} {2.0 + n:.6e}\n')


def make_athena_run(basedir, pid='fake', nx=(8, 6, 4), ngrid=(2, 1, 1),
                    le=(-4.0, -3.0, -2.0), re=(4.0, 3.0, 2.0),
                    nums=(0, 1)):
    """Create an Athena (C) run with per-rank vtk files in id<rank>/.

    The domain is split into ngrid[0] grids along x only. For snapshot num
    the global fields are density = cell_index_field(nx, 1000*num),
    velocity[..., c] = density + 0.1*(c+1), pressure = 2*density.
    Returns a dict describing the expected global arrays.
    """
    assert ngrid[1] == ngrid[2] == 1
    nx, le, re = np.array(nx), np.array(le, float), np.array(re, float)
    dx = (re - le)/nx
    (basedir / 'id0').mkdir(parents=True)
    write_stdout_log(basedir / 'out.txt', ATHINPUT_ATHENA.format(
        pid=pid, nx=nx, le=le, re=re, ngrid=ngrid))
    write_hst(basedir / 'id0' / f'{pid}.hst', np.arange(0.0, 2.05, 0.1))

    nxg = nx[0]//ngrid[0]
    expected = dict(nx=nx, le=le, re=re, dx=dx, pid=pid, nums=list(nums),
                    fields={})
    for num in nums:
        d = cell_index_field(nx, 1000.0*num)
        v = np.stack([d + 0.1*(c + 1) for c in range(3)], axis=-1)
        p = 2.0*d
        expected['fields'][num] = dict(density=d, velocity=v, pressure=p)
        for rank in range(ngrid[0]):
            sl = np.s_[:, :, rank*nxg:(rank + 1)*nxg]
            iddir = basedir / f'id{rank}'
            iddir.mkdir(exist_ok=True)
            name = (f'{pid}.{num:04d}.vtk' if rank == 0
                    else f'{pid}-id{rank}.{num:04d}.vtk')
            gle = le + np.array([rank*nxg*dx[0], 0.0, 0.0])
            write_vtk_athena(iddir / name, float(num), gle, dx,
                             dict(density=d[sl], velocity=v[sl],
                                  pressure=p[sl]))
    return expected


ATHDF_NAME_LEN = 21


def write_athdf(fname, time, le, re, nx, nb, fields):
    """Write one Athena++ .athdf file (no mesh refinement) with h5py.

    Mirrors src/outputs/athena_hdf5.cpp: root attributes, per-block
    Levels/LogicalLocations/x?f/x?v datasets, and one 5D dataset
    'prim' of shape (nvar, nblock, nz, ny, nx) in float32.
    fields: ordered dict variable name -> global (nz, ny, nx) array.
    Meshblocks are ordered with x1 fastest, then x2, then x3.
    """
    import h5py

    nx, nb = np.array(nx), np.array(nb)
    le, re = np.array(le, float), np.array(re, float)
    dx = (re - le)/nx
    nblk = nx//nb
    locs = [(i, j, k) for k in range(nblk[2]) for j in range(nblk[1])
            for i in range(nblk[0])]
    names = list(fields)

    def s(lst):
        return np.array(lst, dtype=f'S{ATHDF_NAME_LEN}')

    with h5py.File(fname, 'w') as f:
        f.attrs['NumCycles'] = np.int32(0)
        f.attrs['Time'] = np.float32(time)
        f.attrs['Coordinates'] = np.bytes_(b'cartesian')
        for d in range(3):
            f.attrs[f'RootGridX{d+1}'] = np.array([le[d], re[d], 1.0],
                                                  dtype=np.float32)
        f.attrs['RootGridSize'] = nx.astype(np.int32)
        f.attrs['NumMeshBlocks'] = np.int32(len(locs))
        f.attrs['MeshBlockSize'] = nb.astype(np.int32)
        f.attrs['MaxLevel'] = np.int32(0)
        f.attrs['NumVariables'] = np.array([len(names)], dtype=np.int32)
        f.attrs['DatasetNames'] = s(['prim'])
        f.attrs['VariableNames'] = s(names)

        f['Levels'] = np.zeros(len(locs), dtype=np.int32)
        f['LogicalLocations'] = np.array(locs, dtype=np.int64)
        prim = np.empty((len(names), len(locs), *nb[::-1]), dtype=np.float32)
        for d, key in enumerate(('x1', 'x2', 'x3')):
            xf = np.empty((len(locs), nb[d] + 1), dtype=np.float32)
            for b, loc in enumerate(locs):
                i0 = loc[d]*nb[d]
                xf[b] = le[d] + dx[d]*np.arange(i0, i0 + nb[d] + 1)
            f[f'{key}f'] = xf
            f[f'{key}v'] = 0.5*(xf[:, 1:] + xf[:, :-1])
        for b, (i, j, k) in enumerate(locs):
            sl = np.s_[k*nb[2]:(k + 1)*nb[2], j*nb[1]:(j + 1)*nb[1],
                       i*nb[0]:(i + 1)*nb[0]]
            for n, name in enumerate(names):
                prim[n, b] = fields[name][sl]
        f['prim'] = prim


def make_athenapp_hdf5_run(basedir, pid='fakeh5', nx=(8, 6, 4), nb=(4, 3, 4),
                           le=(-4.0, -3.0, -2.0), re=(4.0, 3.0, 2.0),
                           nums=(0, 1)):
    """Create an Athena++ run with hdf5 (prim) output in basedir/hdf5/.

    Meshblocks split the domain along x and y. Fields: rho as in
    make_athena_run, press = 2*rho, vel1..3 = rho + 0.1, 0.2, 0.3.
    """
    nx, le, re = np.array(nx), np.array(le, float), np.array(re, float)
    basedir.mkdir(parents=True)
    (basedir / 'hdf5').mkdir()
    athinput = ATHINPUT_ATHENAPP.format(pid=pid, nx=nx, le=le, re=re, nb=nb)
    athinput = athinput.replace('file_type  = vtk', 'file_type  = hdf5')
    write_stdout_log(basedir / 'out.txt', athinput)
    write_hst(basedir / f'{pid}.hst', np.arange(0.0, 2.05, 0.1))

    expected = dict(nx=nx, le=le, re=re, dx=(re - le)/nx, pid=pid,
                    nums=list(nums), fields={})
    for num in nums:
        d = cell_index_field(nx, 1000.0*num)
        fields = dict(rho=d, press=2.0*d)
        for c in range(3):
            fields[f'vel{c+1}'] = d + np.float32(0.1*(c + 1))
        expected['fields'][num] = fields
        write_athdf(basedir / 'hdf5' / f'{pid}.out2.{num:05d}.athdf',
                    float(num), le, re, nx, nb, fields)
    return expected


def make_athenapp_run(basedir, pid='fakepp', nx=(8, 6, 4), nb=(4, 6, 4),
                      le=(-4.0, -3.0, -2.0), re=(4.0, 3.0, 2.0),
                      nums=(0, 1)):
    """Create an Athena++ run with one vtk file per meshblock.

    Meshblocks split the domain along x only. Fields as in make_athena_run,
    named rho, vel, press as Athena++ writes them.
    """
    assert nb[1] == nx[1] and nb[2] == nx[2]
    nx, le, re = np.array(nx), np.array(le, float), np.array(re, float)
    dx = (re - le)/nx
    basedir.mkdir(parents=True)
    write_stdout_log(basedir / 'out.txt', ATHINPUT_ATHENAPP.format(
        pid=pid, nx=nx, le=le, re=re, nb=nb))
    write_hst(basedir / f'{pid}.hst', np.arange(0.0, 2.05, 0.1))

    nblock = nx[0]//nb[0]
    expected = dict(nx=nx, le=le, re=re, dx=dx, pid=pid, nums=list(nums),
                    nblock=nblock, fields={})
    for num in nums:
        d = cell_index_field(nx, 1000.0*num)
        v = np.stack([d + 0.1*(c + 1) for c in range(3)], axis=-1)
        p = 2.0*d
        expected['fields'][num] = dict(rho=d, vel=v, press=p)
        for b in range(nblock):
            sl = np.s_[:, :, b*nb[0]:(b + 1)*nb[0]]
            x1f = le[0] + dx[0]*np.arange(b*nb[0], (b + 1)*nb[0] + 1)
            x2f = le[1] + dx[1]*np.arange(nx[1] + 1)
            x3f = le[2] + dx[2]*np.arange(nx[2] + 1)
            write_vtk_athenapp(basedir / f'{pid}.block{b}.out2.{num:05d}.vtk',
                               float(num), x1f, x2f, x3f,
                               dict(rho=d[sl], vel=v[sl], press=p[sl]))
    return expected
