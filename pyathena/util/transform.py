import numpy as np
import xarray as xr
from scipy.spatial.transform import Rotation
import fast_histogram as fh

def euler_rotation(vec, angles):
    """Rotate coordinate axes to transform the components of vector field `vec`

    Parameters
    ----------
    vec : tuple-like
        (vx, vy, vz) representing Cartesian vector components
    angles : array
        Euler angles [alpha, beta, gamma] in radian

    Returns
    -------
    tuple
        Cartesian components of the rotated vector.
    """
    angles = np.array(angles)
    r = Rotation.from_euler('zyx', -angles, degrees=False)
    rotmat = r.as_matrix()
    vxp = rotmat[0, 0]*vec[0] + rotmat[0, 1]*vec[1] + rotmat[0, 2]*vec[2]
    vyp = rotmat[1, 0]*vec[0] + rotmat[1, 1]*vec[1] + rotmat[1, 2]*vec[2]
    vzp = rotmat[2, 0]*vec[0] + rotmat[2, 1]*vec[1] + rotmat[2, 2]*vec[2]
    return (vxp, vyp, vzp)


def to_spherical(vec, origin, newz=None):
    """Transform vector components from Cartesian to spherical coordinates

    Assumes vec is a tuple of xarray.DataArray.

    Parameters
    ----------
    vec : tuple-like
        (vx, vy, vz) representing Cartesian vector components
    origin : tuple-like
        (x0, y0, z0) representing the origin of the spherical coords.
    newz : array, optional
        Cartesian components of the z-axis vector for the spherical
        coordinates. If not given, it is assumed to be (0, 0, 1).

    Returns
    -------
    r : array
        Binned radius
    vec_sph : tuple-like
        (v_r, v_th, v_ph) representing the three components of
        velocities in spherical coords.
    """
    vx, vy, vz = vec
    x0, y0, z0 = origin
    x, y, z = vx.x - x0, vx.y - y0, vx.z - z0

    if newz is not None:
        if ((np.array(newz)**2).sum() == 0):
            raise ValueError("new z axis vector should not be a null vector")

        # Rotate to align z axis
        zhat = np.array([0, 0, 1])
        alpha = np.arctan2(newz[1], newz[0])
        beta = np.arccos(np.dot(newz, zhat) / np.sqrt(np.dot(newz, newz)))
        vx, vy, vz = euler_rotation((vx, vy, vz), [alpha, beta, 0])
        x, y, z = euler_rotation((x, y, z), [alpha, beta, 0])

    # Calculate spherical coordinates
    R = np.sqrt(x**2 + y**2)
    r = np.sqrt(R**2 + z**2)
    th = np.arctan2(R, z)
    ph = np.arctan2(y, x)

    # Move branch cut [-pi, pi] -> [0, 2pi]
    ph = ph.where(ph >= 0, other=ph + 2*np.pi)
    sin_th, cos_th = R/r, z/r
    sin_ph, cos_ph = y/R, x/R

    # Avoid singularity
    sin_th = sin_th.where(r != 0, other=0)
    cos_th = cos_th.where(r != 0, other=0)
    sin_ph = sin_ph.where(R != 0, other=0)
    cos_ph = cos_ph.where(R != 0, other=0)

    # Transform Cartesian (vx, vy, vz) ->  spherical (v_r, v_th, v_phi)
    v_r = (vx*sin_th*cos_ph + vy*sin_th*sin_ph + vz*cos_th).rename('v_r')
    v_th = (vx*cos_th*cos_ph + vy*cos_th*sin_ph - vz*sin_th).rename('v_theta')
    v_ph = (-vx*sin_ph + vy*cos_ph).rename('v_phi')

    # assign spherical coordinates
    v_r.coords['r'] = r
    v_th.coords['r'] = r
    v_ph.coords['r'] = r
    v_r.coords['th'] = th
    v_th.coords['th'] = th
    v_ph.coords['th'] = th
    v_r.coords['ph'] = ph
    v_th.coords['ph'] = ph
    v_ph.coords['ph'] = ph
    vec_sph = (v_r, v_th, v_ph)
    return r, vec_sph


def to_cylindrical(vec, origin):
    """Transform vector components from Cartesian to cylindrical coords.

    Assumes vec is a tuple of xarray.DataArray.

    Parameters
    ----------
    vec : tuple-like
        (vx, vy, vz) representing Cartesian vector components
    origin : tuple-like
        (x0, y0, z0) representing the origin of the cylindrical coords.

    Returns
    -------
    R : array
        Binned radius
    vec_cyl : tuple-like
        (v_R, v_ph, v_z) representing the three components of
        velocities in cylindrical coords.
    """
    vx, vy, vz = vec
    x0, y0, z0 = origin
    x, y, z = vx.x, vx.y, vx.z

    # Calculate cylindrical coordinates
    R = np.sqrt((x-x0)**2 + (y-y0)**2)
    ph = np.arctan2(y-y0, x-x0)
    # Move branch cut [-pi, pi] -> [0, 2pi]
    ph = ph.where(ph >= 0, other=ph + 2*np.pi)
    sin_ph, cos_ph = (y-y0)/R, (x-x0)/R
    # Avoid singularity
    if x0 in x and y0 in y:
        sin_ph.loc[dict(x=x0, y=y0)] = 0
        cos_ph.loc[dict(x=x0, y=y0)] = 0

    # Transform Cartesian (vx, vy, vz) ->  cylindrical (v_R, v_phi, v_z)
    v_R = (vx*cos_ph + vy*sin_ph).rename('v_R')
    v_ph = (-vx*sin_ph + vy*cos_ph).rename('v_phi')
    v_z = vz

    # assign spherical coordinates
    v_R.coords['R'] = R
    v_ph.coords['R'] = R
    v_z.coords['R'] = R
    v_R.coords['ph'] = ph
    v_ph.coords['ph'] = ph
    v_z.coords['ph'] = ph
    v_R.coords['z'] = z - z0
    v_ph.coords['z'] = z - z0
    v_z.coords['z'] = z - z0
    vec_cyl = (v_R, v_ph, v_z)
    return R, vec_cyl


def groupby_bins(dat, coord, edges, cumulative=False):
    """Alternative to xr.groupby_bins, which is very slow

    Parameters
    ----------
    dat : xarray.DataArray
        input dataArray
    coord : str
        coordinate name along which data is binned
    edges : array-like
        bin edges
    cumulative : bool
        if True, perform cumulative binning, e.g.,
          v_r_binned[i] = v_r( edge[0] <= r < edge[i+1] ).mean()
        to calculate average velocity dispersion within radius r

    Returns
    ------
    res: xarray.DataArray
        binned array
    """
    dat = dat.transpose(*sorted(list(dat.dims), reverse=True))
    fc = dat[coord].data.flatten()  # flattened coordinates
    fd = dat.data.flatten()  # flattened data
    bin_sum = np.histogram(fc, edges, weights=fd)[0]
    bin_cnt = np.histogram(fc, edges)[0]
    if cumulative:
        bin_sum = np.cumsum(bin_sum)
        bin_cnt = np.cumsum(bin_cnt)
    res = bin_sum / bin_cnt
    # set new coordinates at the bin center
    centers = 0.5*(edges[1:] + edges[:-1])
    res = xr.DataArray(data=res, coords={coord: centers}, name=dat.name)
    return res


def fast_groupby_bins(dat, coord, ledge, redge, nbin, cumulative=False, skipna=True):
    """High performance version of groupby_bins using fast_histogram.

    Although groupby_bins using np.histogram is significantly faster than
    xr.groupby_bins, it is still too slow. Assuming equally spaced bins,
    fast_histogram achieves order of magnitude higher performance.
    This function implements groupby_bins based on fast_histogram.

    Parameters
    ----------
    dat : xarray.DataArray
        Input dataArray.
    coord : str
        Coordinate name along which data is binned.
    ledge : float
        Leftmost bin edge.
    redge : float
        Rightmost bin edge.
    nbin : int
        Number of bins (= number of edges - 1)
    cumulative : bool
        If True, perform cumulative binning, e.g.,
          v_r_binned[i] = v_r( edge[0] <= r < edge[i+1] ).mean()
        to calculate average velocity dispersion within radius r

    Returns
    ------
    res: xarray.DataArray
        binned array
    """
    dat = dat.transpose(*sorted(list(dat.dims), reverse=True))
    fc = dat[coord].data.flatten()  # flattened coordinates
    fd = dat.data.flatten()  # flattened data
    if skipna:
        mask = ~np.isnan(fd)
        fc = fc[mask]
        fd = fd[mask]
    bin_sum = fh.histogram1d(fc, nbin, (ledge, redge), weights=fd)
    bin_cnt = fh.histogram1d(fc, nbin, (ledge, redge))
    if cumulative:
        bin_sum = np.cumsum(bin_sum)
        bin_cnt = np.cumsum(bin_cnt)
    res = bin_sum / bin_cnt
    # set new coordinates at the bin center
    edges = np.linspace(ledge, redge, nbin + 1)
    centers = 0.5*(edges[1:] + edges[:-1])
    res = xr.DataArray(data=res, coords={coord: centers}, name=dat.name)
    return res
