"""Module containing functions that are not generally reusable"""

# python modules
import pyathena as pa
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import pickle
from grid_dendro import dendrogram

# pyathena modules
from pyathena.tigress_gc import tools


def run_grid(s, num, overwrite=False):
    """Run GRID-dendro

    Parameters
    ----------
    s : LoadSimTIGRESSGC
        Simulation metadata.
    num : int
        Snapshot number.
    """
    # Check if file exists
    ofname = Path(s.savdir, 'GRID',
                  'dendrogram.{:04d}.p'.format(num))
    ofname.parent.mkdir(exist_ok=True)
    if ofname.exists() and not overwrite:
        print('[run_grid] file already exists. Skipping...')
        return

    # Load data and construct dendrogram
    print('[run_grid] processing model {} num {}'.format(s.basename, num))
    fpath = Path(s.basedir, 'postproc_gravity', f'gc.{num:04d}.vtk')
    ds = pa.io.read_vtk.read_vtk(fpath)
    phi = ds.get_field('Phi').Phi.transpose('z', 'y', 'x').to_numpy()
    gd = dendrogram.Dendrogram(phi, verbose=False)
    gd.construct()
    gd.prune()

    # Write to file
    with open(ofname, 'wb') as handle:
        pickle.dump(gd, handle, protocol=pickle.HIGHEST_PROTOCOL)


def linewidth_size_grid_dendro(s, num, overwrite=False):
    """Calculate linewidth-size relation for GRID-dendro nodes

    Parameters
    ----------
    s : LoadSimTIGRESSGC
        Simulation metadata.
    num : int
        Snapshot number.
    """
    # Check if file exists
    ofname = Path(s.savdir, 'linewidth_size',
                  'grid_dendro.{:04d}.p'.format(num))
    ofname.parent.mkdir(exist_ok=True)
    if ofname.exists() and not overwrite:
        print('[linewidth_size_grid_dendro] file already exists. Skipping...')
        return

    # Load data and construct dendrogram
    print('[linewidth_size_grid_dendro] processing model {} num {}'.format(s.basename, num))
    gd = s.load_dendro(num)
    ds = s.load_vtk(num)
    dat = ds.get_field(['density', 'pressure', 'velocity'])
    tools.add_derived_fields(s, dat, 'temperature')

    # Calculate velocity dispersion and size of the GRID-dendro nodes.
    reff, sigma_z, sigma_x, sigma_y = [], [], [], []
    for nd in gd.nodes:
        temp = gd.filter_data(dat.temperature, nd, drop=True)
        warmcold = temp < 2e4
        rho = gd.filter_data(dat.density, nd, drop=True)[warmcold]
        vx = gd.filter_data(dat.velocity1, nd, drop=True)[warmcold]
        vy = gd.filter_data(dat.velocity2, nd, drop=True)[warmcold]
        vz = gd.filter_data(dat.velocity3, nd, drop=True)[warmcold]
        reff.append((3*gd.len(nd)*s.dx**3 / (4*np.pi))**(1./3.))
        sigma_x.append(np.sqrt(np.average(vx**2, weights=rho) - np.average(vx, weights=rho)**2))
        sigma_y.append(np.sqrt(np.average(vy**2, weights=rho) - np.average(vy, weights=rho)**2))
        sigma_z.append(np.sqrt(np.average(vz**2, weights=rho) - np.average(vz, weights=rho)**2))
    reff = np.array(reff)
    sigma_x = np.array(sigma_x)
    sigma_y = np.array(sigma_y)
    sigma_z = np.array(sigma_z)

    res = dict(num=num,
               time=ds.domain['time']*s.u.Myr,
               radius=reff,
               veldisp_x=sigma_x,
               veldisp_y=sigma_y,
               veldisp_z=sigma_z)

    # Write to file
    with open(ofname, 'wb') as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)


def calculate_linewidth_size(s, num, seed, overwrite=False, dat=None):
    # Check if file exists
    ofname = Path(s.savdir, 'linewidth_size',
                  'linewidth_size.{:04d}.{}.nc'.format(num, seed))
    ofname.parent.mkdir(exist_ok=True)
    if ofname.exists() and not overwrite:
        print('[linewidth_size] file already exists. Skipping...')
        return

    msg = '[linewidth_size] processing model {} num {} seed {}'
    print(msg.format(s.basename, num, seed))

    if dat is None:
        ds = s.load_vtk(num)
        dat = ds.get_field(['density', 'pressure', 'velocity'])
        tools.add_derived_fields(s, dat, ['temperature', 'eta'])

    rng = np.random.default_rng(seed)
    if len(np.unique(s.domain['Nx'])) > 1:
        raise ValueError("Cubic domain is assumed, but the domain is not cubic")
    Nx = s.domain['Nx'][0]  # Assume cubic domain
    i, j, k = rng.integers(low=0, high=Nx-1, size=(3))
    origin = (dat.x.isel(x=i).data[()],
              dat.y.isel(y=j).data[()],
              dat.z.isel(z=k).data[()])

    dat.coords['r'] = np.sqrt((dat.z - origin['z'])**2 + (dat.y - origin['y'])**2 + (dat.x - origin['x'])**2)

    rmax = s.Lbox/2

    nbin = int(np.ceil(rmax/s.dx))
    ledge = 0.5*s.dx
    redge = (nbin + 0.5)*s.dx

    rprf = {}
    for cum_flag, suffix in zip([True, False], ['', '_sh']):
        for k in ['vel1', 'vel2', 'vel3']:
            rprf[k+suffix] = transform.fast_groupby_bins(dat[k], 'r', ledge, redge, nbin, cumulative=cum_flag)
            rprf[f'{k}_sq'+suffix] = transform.fast_groupby_bins(dat[k]**2, 'r', ledge, redge, nbin, cumulative=cum_flag)
            rprf[f'dat{k}'+suffix] = np.sqrt(rprf[f'{k}_sq'+suffix] - rprf[k+suffix]**2)
        rprf['rho'+suffix] = transform.fast_groupby_bins(dat['dens'], 'r', ledge, redge, nbin, cumulative=cum_flag)
    rprf = xr.Dataset(rprf)

    # write to file
    if ofname.exists():
        ofname.unlink()
    rprf.to_netcdf(ofname)


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
    ptop_hot = (((dat.pressure + dat.turbulent_pressure)*(~dat.fwarm)).isel(z=-1)
            / (~dat.fwarm).isel(z=-1))
    wtot = dat.weight_self + dat.weight_ext
    prfm = xr.Dataset(dict(ptot=ptot*s.u.pok,
                           ptop=ptop*s.u.pok,
                           ptop_hot=ptop_hot*s.u.pok,
                           wtot=wtot*s.u.pok,
                           sigma=dat.surface_density*s.u.Msun,
                           sigma_sfr=dat.sfr40))
    fname.unlink(missing_ok=True)
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
