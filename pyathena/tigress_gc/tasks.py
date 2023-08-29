"""Module containing functions that are not generally reusable"""

# python modules
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd

# pyathena modules
from pyathena.tigress_gc import tools


def save_ring_averages(s, Rmax, mf_crit=0.9, overwrite=False):
    """Calculates ring masked averages and save to file

    Parameters
    ----------
    s : pyathena.LoadSim instance
    """
    fname = Path(s.basedir, 'ring_averages', 'gc_ring_average_warmcold.nc')
    fname.parent.mkdir(exist_ok=True)
    if fname.exists() and not overwrite:
        print('File {} already exists; skipping...'.format(fname))
        return

    time, ravgs = [], []
    for num in s.nums[100:]:
        print('processing model {} num {}'.format(s.basename, num))
        ravg = tools.calculate_ring_averages(s, num, Rmax, mf_crit, warmcold=True)
        time.append(ravg.time)
        ravgs.append(ravg)
    ravgs = xr.concat(ravgs, dim=pd.Index(time, name='t'), combine_attrs='drop_conflicts')
    ravgs.to_netcdf(fname)


def save_azimuthal_averages(s, overwrite=False):
    """Calculates azimuthal averages and save to file

    Parameters
    ----------
    s : pyathena.LoadSim instance
    """
    for num in s.nums:
        fname = Path(s.basedir, 'azimuthal_averages_warmcold',
                     'gc_azimuthal_average.{:04}.nc'.format(num))
        fname.parent.mkdir(exist_ok=True)
        if fname.exists() and not overwrite:
            print('File {} already exists; skipping...'.format(fname))
            continue
        print('processing model {} num {}'.format(s.basename, num))
        rprf = tools.calculate_azimuthal_averages(s, num, warmcold=True)
        rprf.to_netcdf(fname)


def prfm_quantities(s, num, overwrite=False):
    """Calculate Q(x, y) of the warm-cold gas.

    Parameters
    ----------
    s : pyathena.LoadSimTIGRESSGC
        LoadSim instance.
    num : int
        Snapshot number
    overwrite : bool, optional
        Flag to overwrite
    """
    fname = Path(s.basedir, 'prfm_quantities',
                 'prfm.{:04}.nc'.format(num))
    fname.parent.mkdir(exist_ok=True)
    if fname.exists() and not overwrite:
        print('File {} already exists; skipping...'.format(fname))
        return

    msg = '[prfm_quantities] processing model {} num {}'
    print(msg.format(s.basename, num))

    ds = s.load_vtk(num, id0=False)
    dat = ds.get_field(['density',
                        'velocity',
                        'pressure',
                        'gravitational_potential'])
    tools.add_derived_fields(s, dat, ['temperature',
                                      'turbulent_pressure',
                                      'weight_self',
                                      'weight_ext',
                                      'fwarm',
                                      'surface_density',
                                      'sfr40'])
    ptot = (((dat.pressure + dat.turbulent_pressure)*dat.fwarm).sel(z=slice(-s.dz, s.dz)).sum(dim='z')
            / (dat.fwarm.sel(z=slice(-s.dz, s.dz)).sum(dim='z')))
    ptop = (((dat.pressure + dat.turbulent_pressure)*dat.fwarm).isel(z=-1)
            / dat.fwarm.isel(z=-1))
    wtot = dat.weight_self + dat.weight_ext
    prfm = xr.Dataset(dict(ptot=ptot*s.u.pok,
                           ptop=ptop*s.u.pok,
                           wtot=wtot*s.u.pok,
                           sigma=dat.surface_density*s.u.Msun,
                           sigma_sfr=dat.sfr40))
    prfm.to_netcdf(fname)


def save_time_averaged_snapshot(s, ts, te, overwrite=False):
    """Generate time averaged snapshot between [ts, te]

    Parameters
    ----------
    s : pyathena.LoadSim instance
    ts : start time
    te : end time
    """

    fname = Path(s.basedir, 'time_averages', 'prims.nc')
    fname.parent.mkdir(exist_ok=True)
    if fname.exists() and not overwrite:
        print('File {} already exists; skipping...'.format(fname))
        return

    ns = tools.find_snapshot_number(s, ts)
    ne = tools.find_snapshot_number(s, te)
    nums = np.arange(ns, ne+1)

    fields = ['density', 'velocity', 'pressure']

    # load a first vtk
    ds = s.load_vtk(nums[0], id0=False)
    dat = ds.get_field(fields)

    # loop through vtks
    for num in nums[1:]:
        ds = s.load_vtk(num, id0=False)
        tmp = ds.get_field(fields)
        dat += tmp
    dat /= len(nums)
    dat.attrs.update({'ts':s.load_vtk(ns).domain['time'],
                      'te':s.load_vtk(ne).domain['time']})
    if fname.exists():
        fname.unlink()
    dat.to_netcdf(fname)
