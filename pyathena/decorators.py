import os
import os.path as osp
import functools
import xarray as xr
from inspect import getcallargs

def check_netcdf_zprof_vtk(_read_zprof_vtk):

    @functools.wraps(_read_zprof_vtk)
    def wrapper(cls, *args, **kwargs):
        # Convert positional args to keyword args
        call_args = getcallargs(_read_zprof_vtk, cls, *args, **kwargs)
        call_args.pop('self')
        kwargs = call_args

        try:
            prefix = kwargs['prefix']
        except KeyError:
            prefix = 'zprof_vtk'

        if 'savdir' in kwargs:
            savdir = kwargs['savdir']
            if savdir is None:
                savdir = osp.join(cls.savdir, prefix)
        else:
            savdir = osp.join(cls.savdir, prefix)

        if 'force_override' in kwargs:
            force_override = kwargs['force_override']
        else:
            force_override = False

        # Create savdir if it doesn't exist
        if not osp.exists(savdir):
            os.makedirs(savdir)
            force_override = True

        if 'num' in kwargs:
            fnetcdf = '{0:s}.{1:s}.{2:s}.{3:04d}.nc'.\
                format(cls.problem_id, cls.basename,
                       kwargs['prefix'], kwargs['num'])
        else:
            fnetcdf = '{0:s}.{1:s}.{2:s}.all.nc'.\
                format(cls.problem_id, cls.basename, kwargs['prefix'])

        fnetcdf = osp.join(savdir, fnetcdf)

        if not force_override and osp.exists(fnetcdf):
            cls.logger.info('[read_zprof_vtk]: Read' + \
                            ' zprof_vtk from existing NetCDF dump.')
            ds = xr.open_dataset(fnetcdf)
            return ds
        else:
            cls.logger.info('[read_zprof_vtk]: Construct zprof from vtk')
            ds = _read_zprof_vtk(cls, **kwargs)

            # Somehow overwriting with mode='w' in to_netcdf doesn't work..
            # Delete file first
            if osp.exists(fnetcdf):
                os.remove(fnetcdf)

            try:
                ds.to_netcdf(fnetcdf, mode='w')
            except (IOError, PermissionError) as e:
                cls.logger.warning('[read_zprof-vtk]: Could not netcdf to {0:s}.'\
                                   .format(fnetcdf))

            return ds

    return wrapper
