import numpy as np
import xarray as xr
import dask.array as da
from scipy.spatial.transform import Rotation
import fast_histogram as fh
import warnings

def euler_rotation(vec, angles):
    """Rotate coordinate axes to transform the components of vector field `vec`

    This assumes "passive" rotation where the vector itself is not rotated geometrically,
    but the coordinate axes are rotated. The euler angles take the "intrinsic" convention
    where the subsequent rotations are applied about the rotated axes.

    Parameters
    ----------
    vec : tuple, list, or numpy.ndarray of xarray.DataArray or numpy.ndarray
        (vx, vy, vz) representing Cartesian vector field components.
    angles : tuple, list, or numpy.ndarray
        Euler angles [alpha, beta, gamma] in radian.

    Returns
    -------
    tuple of xarray.DataArray or numpy.ndarray matching the input type
        Cartesian components of the rotated vector.
    """
    alpha, beta, gamma = angles
    ca = np.cos(alpha)
    cb = np.cos(beta)
    cr = np.cos(gamma)
    sa = np.sin(alpha)
    sb = np.sin(beta)
    sr = np.sin(gamma)
    rotmat = [[ ca*cr - sa*cb*sr,  sa*cr + ca*cb*sr, sb*sr],
              [-ca*sr - sa*cb*cr, -sa*sr + ca*cb*cr, sb*cr],
              [ sa*sb           , -ca*sb           ,    cb]]
    # The order is important here. To utilize xarray broadcasting rule,
    # we compute [2, 1, 0] order to preserve (z, y, x) layout for Athena data
    vxp = rotmat[0][2]*vec[2] + rotmat[0][1]*vec[1] + rotmat[0][0]*vec[0]
    vyp = rotmat[1][2]*vec[2] + rotmat[1][1]*vec[1] + rotmat[1][0]*vec[0]
    vzp = rotmat[2][2]*vec[2] + rotmat[2][1]*vec[1] + rotmat[2][0]*vec[0]
    return (vxp, vyp, vzp)


def to_spherical(vec, origin, newz=None):
    """Transform vector components from Cartesian to spherical coordinates.

    Supports numpy.ndarray and xarray.DataArray (both dask and numpy-backed).

    Parameters
    ----------
    vec : tuple or list of xarray.DataArray
        (vx, vy, vz) representing Cartesian vector components.
    origin : tuple or list
        (x0, y0, z0) representing the origin of the spherical coords.
    newz : tuple or list, optional
        Cartesian components of the z-axis vector for the spherical
        coordinates. If not given, it is assumed to be (0, 0, 1).

    Returns
    -------
    r : xarray.DataArray
        Binned radius
    vec_sph : tuple
        (v_r, v_th, v_ph) representing the three components of
        vector in spherical coords.
    """
    vx, vy, vz = vec
    x0, y0, z0 = origin
    if not vx.chunksizes == vy.chunksizes == vz.chunksizes:
        raise ValueError("All input arrays should have the same chunk sizes")
    else:
        # Note that if vx, vy, vz are numpy-backed xarray.DataArray, then
        # vx.chunksizes is an empty dictionary. In this case, _chunk_like
        # simply returns the input arrays.
        chunks = vx.chunksizes
    # Workaround for xarray being unable to chunk coordinates
    # see https://github.com/pydata/xarray/issues/6204
    # The workaround is provided by _chunk_like helper function introduced in
    # xclim. See https://github.com/Ouranosinc/xclim/pull/1542
    x, y, z = _chunk_like(vx.x, vx.y, vx.z, chunks=chunks)
    x, y, z = x - x0, y - y0, z - z0

    if newz is not None:
        # Let's avoid eager computation at the cost of safety.
#        if ((np.array(newz)**2).sum() == 0):
#            raise ValueError("new z axis vector should not be a null vector")

        # Obtain Euler angles to rotate z-axis to newz
        newz_mag = np.sqrt(newz[0]**2 + newz[1]**2 + newz[2]**2)
        alpha = (np.arctan2(newz[1], newz[0]) + np.pi/2) % (2*np.pi)
        beta = np.arccos(newz[2] / newz_mag)
        vx, vy, vz = euler_rotation((vx, vy, vz), [alpha, beta, 0])
        x, y, z = euler_rotation((x, y, z), [alpha, beta, 0])

    # Calculate spherical coordinates
    R = np.sqrt(y**2 + x**2)
    r = np.sqrt(z**2 + R**2)
    th = np.arctan2(R, z)
    ph = np.arctan2(y, x)

    # Move branch cut [-pi, pi] -> [0, 2pi]
    ph = ph.where(ph >= 0, other=ph + 2*np.pi)
    sin_th, cos_th = R/r, z/r
    sin_ph, cos_ph = y/R, x/R

    # Break degeneracy by choosing arbitrary theta and phi at coordinate singularities
    # \theta = 0 at r = 0, and \phi = 0 at R = 0.
    sin_th = sin_th.where(r != 0, other=0)
    cos_th = cos_th.where(r != 0, other=1)
    sin_ph = sin_ph.where(R != 0, other=0)
    cos_ph = cos_ph.where(R != 0, other=1)

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

    TODO
    ----
    - Add support for dask arrays
    - Make structure identical to to_spherical

    Parameters
    ----------
    vec : tuple or list of xarray.DataArray
        (vx, vy, vz) representing Cartesian vector components.
    origin : tuple or list
        (x0, y0, z0) representing the origin of the spherical coords.

    Returns
    -------
    R : xarray.DataArray
        Binned radius
    vec_cyl : tuple
        (v_R, v_ph, v_z) representing the three components of
        velocities in cylindrical coords.
    """
    vx, vy, vz = vec
    x0, y0, z0 = origin
    if not vx.chunksizes == vy.chunksizes == vz.chunksizes:
        raise ValueError("All input arrays should have the same chunk sizes")
    else:
        # Note that if vx, vy, vz are numpy-backed xarray.DataArray, then
        # vx.chunksizes is an empty dictionary. In this case, _chunk_like
        # simply returns the input arrays.
        chunks = vx.chunksizes
    # Workaround for xarray being unable to chunk coordinates
    # see https://github.com/pydata/xarray/issues/6204
    # The workaround is provided by _chunk_like helper function introduced in
    # xclim. See https://github.com/Ouranosinc/xclim/pull/1542
    x, y, z = _chunk_like(vx.x, vx.y, vx.z, chunks=chunks)
    x, y, z = x - x0, y - y0, z - z0

    # Calculate cylindrical coordinates
    R = np.sqrt(y**2 + x**2)
    ph = np.arctan2(y, x)

    # Move branch cut [-pi, pi] -> [0, 2pi]
    ph = ph.where(ph >= 0, other=ph + 2*np.pi)
    sin_ph, cos_ph = y/R, x/R

    # Break degeneracy by choosing arbitrary theta and phi at coordinate singularities
    # \phi = 0 at R = 0.
    sin_ph = sin_ph.where(R != 0, other=0)
    cos_ph = cos_ph.where(R != 0, other=1)

    # Transform Cartesian (vx, vy, vz) ->  cylindrical (v_R, v_phi, v_z)
    v_R = (vx*cos_ph + vy*sin_ph).rename('v_R')
    v_ph = (-vx*sin_ph + vy*cos_ph).rename('v_phi')
    v_z = vz

    # assign cylindrical coordinates
    v_R.coords['R'] = R
    v_ph.coords['R'] = R
    v_z.coords['R'] = R
    v_R.coords['ph'] = ph
    v_ph.coords['ph'] = ph
    v_z.coords['ph'] = ph
    v_R.coords['z'] = vx.z  # Note that we do not want to set z coordinate z - z0.
    v_ph.coords['z'] = vx.z # Use original z coordinates.
    v_z.coords['z'] = vx.z
    vec_cyl = (v_R, v_ph, v_z)
    return R, vec_cyl


def groupby_bins(dat, coord, bins, range=None, cumulative=False):
    """Alternative to xr.groupby_bins, which is very slow

    Parameters
    ----------
    dat : xarray.DataArray
        input dataArray
    coord : str
        coordinate name along which data is binned
    bins : int or sequence of scalars
        If bins is an int, it defines the number of equal-width bins in the given range.
        If bins is a sequence, it defines the bin edges, including the rightmost edge,
        allowing for non-uniform bin widths.
    range : (float, float), optional
        The lower and upper range of the bins.
    cumulative : bool
        if True, perform cumulative binning, e.g.,
          v_r_binned[i] = v_r( edge[0] <= r < edge[i+1] ).mean()
        to calculate average velocity dispersion within radius r

    Returns
    ------
    res: xarray.DataArray
        binned array
    """
    if isinstance(bins, (int, np.integer)):
        if range is None:
            raise ValueError("range should be provided when bins is an int")
        # if bins is an int, then it defines the number of equal-spaced bins
        # where the leftmost and rightmost bin edges are determined by the
        # "range" parameter.
        edges = np.linspace(range[0], range[1], bins + 1)
    else:
        edges = bins
    dat = dat.transpose(*sorted(list(dat.dims), reverse=True))
    fc = dat[coord].data  # coordinates
    fd = dat.data  # data
    bin_sum = np.histogram(fc, bins=bins, range=range, weights=fd)[0]
    bin_cnt = np.histogram(fc, bins=bins, range=range)[0]
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

    .. warning::
        Feb 13, 2025: In fact, np.histogram already optimizes for the uniform bin size.
        To use the optimized version, one just needs to pass integer "bins" and provide
        "range". Still, fast_histogram is slightly faster, but the difference is not
        significant. with 1024^3 data, the speedup is only x1.25. Moreover, fast_histogram
        is not compatible with dask arrays.

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
    warnings.warn(
        "fast_groupby_bins will be deprecated. Use groupby_bins instead.",
        DeprecationWarning,
        stacklevel=2
    )
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


def _chunk_like(*inputs: xr.DataArray | xr.Dataset, chunks: dict[str, int] | None):
    """Helper function that (re-)chunks inputs according to a single chunking dictionary.
    Will also ensure passed inputs are not IndexVariable types, so that they can be chunked.

    Copy-pasted from Ouranosinc/xclim
    See https://github.com/Ouranosinc/xclim/pull/1542
    """
    if not chunks:
        return tuple(inputs)

    outputs = []
    for da in inputs:
        if isinstance(da, xr.DataArray) and isinstance(
            da.variable, xr.core.variable.IndexVariable
        ):
            da = xr.DataArray(da, dims=da.dims, coords=da.coords, name=da.name)
        if not isinstance(da, (xr.DataArray, xr.Dataset)):
            outputs.append(da)
        else:
            outputs.append(
                da.chunk(**{d: c for d, c in chunks.items() if d in da.dims})
            )
    return tuple(outputs)
