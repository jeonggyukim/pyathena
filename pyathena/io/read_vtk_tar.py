"""
Read athena tarred vtk file
"""


import os
import os.path as osp
import glob, struct
import numpy as np
import xarray as xr
import astropy.constants as ac
import astropy.units as au
import tarfile
import dask
import dask.array as da

from .read_vtk import AthenaDataSet,_parse_filename,_vtk_parse_line

from ..util.units import Units

def read_vtk_tar(filename, id0_only=False):
    """Convenience wrapper function to read Athena vtk output file
    using AthenaDataSet class.

    Parameters
    ----------
    filename : string
        Name of the file to open, including extension
    id0_only : bool
        Flag to enforce to read vtk file in id0 directory only.
        Default value is False.

    Returns
    -------
    ds : AthenaDataSet
    """

    return AthenaDataSetTar(filename, id0_only=id0_only)

class AthenaDataSetTar(AthenaDataSet):
    def __init__(self, filename, id0_only=False, units=Units(), dfi=None):
        """Class to read athena vtk file.

        Parameters
        ----------
        filename : string
            Name of the file to open, including extension
        id0_only : bool
            Flag to enforce to read vtk file in id0 directory only.
            Default value is False.
        units : Units
            pyathena Units object (used for reading derived fields)
        dfi : dict
            Dictionary containing derived fields info
        """

        if not osp.exists(filename):
            raise IOError(('File does not exist: {0:s}'.format(filename)))

        dirname, problem_id, num, suffix, ext, mpi_mode, nonzero_id = \
            _parse_filename(filename)

        if id0_only:
            mpi_mode = False

        self.dirname = dirname
        self.problem_id = problem_id
        self.num = int(num)
        self.suffix = suffix
        self.ext = ext
        self.mpi_mode = mpi_mode
        self.fnames = [filename]
        self.u = units
        self.dfi = dfi
        if dfi is not None:
            self.derived_field_list = list(dfi.keys())
        else:
            self.derived_field_list = None

        # open tar file
        if ext == 'tar':
            self.tarfile = tarfile.open(filename)
            self.fnames = self.tarfile.getnames()[1:]
            self.mpi_mode = len(self.fnames)>1
        else:
            raise IOError(('[read_vtk_tar] Expected tarred file but provided:'
                           ' {0:s}'.format(filename)))
        self.grid = self._set_grid()
        self.domain = self._set_domain()
        self.set_region()

        # Need separte field_map for different grids
        if self.domain['all_grid_equal']:
            self._field_map = _set_field_map(self.grid[0],self.tarfile)
            for g in self.grid:
                g['field_map'] = self._field_map
        else:
            for g in self.grid:
                g['field_map'] = _set_field_map(g,self.tarfile)
            self._field_map = self.grid[0]['field_map']

        self.field_list = list(self._field_map.keys())

    def _set_grid(self):
        grid = []
        members = self.tarfile.getmembers()
        # Record filename and data_offset
        for i, tarinfo in enumerate(members):
            if tarinfo.isdir(): continue
            file = self.tarfile.extractfile(tarinfo)
            g = dict()
            g['data'] = dict()
            g['filename'] = tarinfo.name[5:]
            g['tarinfo'] = tarinfo
            g['read_field'] = None
            g['read_type'] = None

            while g['read_field'] is None:
                g['data_offset'] = file.tell()
                line = file.readline()
                _vtk_parse_line(line, g)
            file.close()
            g['Nx'] -= 1
            g['Nx'][g['Nx'] == 0] = 1
            g['dx'][g['Nx'] == 1] = 1.0

            # Right edge
            g['re'] = g['le'] + g['Nx']*g['dx']
            grid.append(g)
        ranklist=[]
        for g in grid:
            fname = g['filename']
            if '-id' in fname: rank = int(fname.split('-id')[1].split('.')[0])
            else: rank = 0
            ranklist.append(rank)
        return list(np.array(grid)[np.argsort(ranklist)])

    def get_field_dask(self, field_list=None, le=None, re=None, chunksize=(64,64,64)):
        """Lazy Read 3d fields data using Dask

        Parameters
        ----------
        le : sequence of floats
           Left edge. Default value is the domain left edge.
        re : sequence of floats
           Right edge. Default value is the domain right edge.
        chunksize : tuple of int, optional
            Dask chunk size along (x, y, z) directions. Default is (512, 512, 512).
        Returns
        -------
        dat : xarray dataset
            An xarray dataset containing fields.
        """

        self.set_region(le=le, re=re)

        if field_list is None:
            field_list = self.field_list
        dall = self._get_array_dask(field_list,chunksize)

        # save as xarray dataset
        # Cell center positions
        coords = dict()
        for axis, le, re, dx in zip(('x', 'y', 'z'), \
                self.region['gle'], self.region['gre'], self.domain['dx']):
            # May not result in correct number of elements due to truncation error
            # x[axis] = np.arange(le + 0.5*dx, re + 0.5*dx, dx)
            coords[axis] = np.arange(le + 0.5*dx, re + 0.25*dx, dx)

        dat = dict()
        for k, v in dall.items():
            if len(v.shape) > self.domain['ndim']:
                for i in range(v.shape[0]):
                    dat[k + str(i+1)] = (('z','y','x'), v[i,...])
            else:
                dat[k] = (('z','y','x'), v)

        attrs = dict()
        for k, v in self.domain.items():
            attrs[k] = v
            attrs['num'] = self.num
        dset = xr.Dataset(dat, coords=coords, attrs=attrs)

        return dset

    def _get_array_dask(self,field_list,chunksize):
        # Read Mesh information
        block_size = self.grid[0]["Nx"]
        mesh_size = self.domain["Nx"]
        num_blocks = mesh_size // block_size  # Assuming uniform grid

        if num_blocks.prod() != self.domain["ngrid"]:
            raise ValueError("Number of blocks does not match the attribute")

        dall = dict()
        for field in field_list:
            fm = self.grid[0]['field_map'][field]
            vector = fm["nvar"] > 1
            if vector:
                reordered = np.empty((fm["nvar"],*num_blocks[::-1]), dtype=object)
            else:
                reordered = np.empty(num_blocks[::-1], dtype=object)

            for gid in self.region["gidx"]:
                grid = self.grid[gid]
                fp = self.tarfile.extractfile(grid['tarinfo'])
                fp.seek(fm['offset'])
                fp.readline() # skip header
                if fm['read_table']:
                    fp.readline()
                arr = readarr(fp, block_size, fm)
                if vector:
                    arr_shape = (*block_size, fm["nvar"])
                else:
                    arr_shape = block_size
                x = da.from_delayed(arr, shape = arr_shape, dtype=fm["dtype"])
                lx1,lx2,lx3 = np.array((grid["le"]-self.domain["le"])/
                                       (self.domain["dx"]*grid["Nx"]),
                                       dtype=int)
                if vector:
                    for i in range(fm["nvar"]):
                        reordered[i, lx3, lx2, lx1] = x[...,i]
                else:
                    reordered[lx3, lx2, lx1] = x
            if vector:
                data = da.block(reordered.tolist()).rechunk((1,*chunksize))
            else:
                data = da.block(reordered.tolist()).rechunk(chunksize)
            dall[field] = data
        return dall

@dask.delayed
def readarr(fp,shape,field_map):
    """read binary output lazily"""
    dsize = field_map["dsize"]
    dtype = field_map["dtype"]
    arr = (np.frombuffer(buffer=fp.read(dsize),dtype=dtype)).newbyteorder()
    if field_map['nvar'] == 1:
        shape = np.flipud(shape)
    else:
        shape = (*np.flipud(shape), field_map['nvar'])
    arr.shape = shape

    return arr

def _set_field_map(grid,tf):
    fp = tf.extractfile(grid['tarinfo'])

    fp.seek(0, 2)
    eof = fp.tell()
    offset = grid['data_offset']
    fp.seek(offset)

    field_map = dict()
    if 'Nx' in grid:
        Nx = grid['Nx']

    while offset < eof:
        line = fp.readline()
        sp = line.strip().split()
        field = sp[1].decode('utf-8')
        field_map[field] = dict()
        field_map[field]['read_table'] = False
        if b"SCALARS" in line:
            tmp = fp.readline()
            field_map[field]['read_table'] = True
            field_map[field]['nvar'] = 1
        elif b"VECTORS" in line:
            field_map[field]['nvar'] = 3
        else:
            raise TypeError(sp[0] + ' is unknown type.')

        field_map[field]['offset'] = offset
        field_map[field]['ndata'] = field_map[field]['nvar']*grid['ncells']
        if field == 'face_centered_B1':
            field_map[field]['ndata'] = (Nx[0]+1)*Nx[1]*Nx[2]
        elif field == 'face_centered_B2':
            field_map[field]['ndata'] = Nx[0]*(Nx[1]+1)*Nx[2]
        elif field == 'face_centered_B3':
            field_map[field]['ndata'] = Nx[0]*Nx[1]*(Nx[2]+1)

        if sp[2] == b'int':
            dtype = 'i'
        elif sp[2] == b'float':
            dtype = 'f'
        elif sp[2] == b'double':
            dtype = 'd'

        field_map[field]['dtype'] = dtype
        field_map[field]['dsize'] = field_map[field]['ndata']*struct.calcsize(dtype)
        fp.seek(field_map[field]['dsize'], 1)
        offset = fp.tell()
        tmp = fp.readline()
        if len(tmp) > 1:
            fp.seek(offset)
        else:
            offset = fp.tell()

    return field_map
