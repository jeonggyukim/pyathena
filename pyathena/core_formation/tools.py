import numpy as np
import xarray as xr
from ..util.transform import to_spherical, groupby_bins


def calculate_radial_profiles(ds, origin, rmax):
    """Calculates radial density velocity profiles at the selected position

    Args:
        ds: xarray.Dataset instance containing conserved variables.
        origin: tuple-like (x0, y0, z0) representing the origin of the spherical coords.
        rmax: maximum radius to bin.

    Returns:
        rprof: xarray.Dataset instance containing angular-averaged radial profiles of
               density, velocities, and velocity dispersions.
    """
    # Convert density and velocities to spherical coord.
    vel = {}
    for dim, axis in zip(['x','y','z'], [1,2,3]):
        vel_ = ds['mom{}'.format(axis)]/ds.dens
        vel[dim] = vel_ - vel_.sel(x=origin[0], y=origin[1], z=origin[2])
    ds_sph = {}
    r, (ds_sph['vel1'], ds_sph['vel2'], ds_sph['vel3']) = to_spherical(vel.values(), origin)
    ds_sph['rho'] = ds.dens.assign_coords(dict(r=r))

    # Radial binning
    edges = np.insert(np.arange(ds.dx/2, rmax, ds.dx), 0, 0)
    rprf = {}
    for key, value in ds_sph.items():
        rprf[key] = groupby_bins(value, 'r', edges)
    for key in ('vel1', 'vel2', 'vel3'):
        rprf[key+'_std'] = np.sqrt(groupby_bins(ds_sph[key]**2, 'r', edges))
    rprf = xr.Dataset(rprf)
    return rprf


def get_coords_iso(ds, iso):
    """Get coordinates at the generating point (parent) of the iso

    Args:
        ds: xarray.Dataset instance containing conserved variables.
        iso: FISO flattened index of an iso

    Returns:
        x, y, z
    """
    k, j, i = np.unravel_index(iso, ds.phi.shape, order='C')
    x = ds.x.isel(x=i).values[()]
    y = ds.y.isel(y=j).values[()]
    z = ds.z.isel(z=k).values[()]
    return x, y, z
