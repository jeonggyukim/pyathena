import numpy as np
import xarray as xr
from pyathena.util import transform
from fiso import boundary


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
    r, (ds_sph['vel1'], ds_sph['vel2'], ds_sph['vel3']) = transform.to_spherical(vel.values(), origin)
    ds_sph['rho'] = ds.dens.assign_coords(dict(r=r))

    # Radial binning
    edges = np.insert(np.arange(ds.dx/2, rmax, ds.dx), 0, 0)
    rprf = {}
    for key, value in ds_sph.items():
        rprf[key] = transform.groupby_bins(value, 'r', edges)
    for key in ('vel1', 'vel2', 'vel3'):
        rprf[key+'_std'] = np.sqrt(transform.groupby_bins(ds_sph[key]**2, 'r', edges))
    rprf = xr.Dataset(rprf)
    return rprf


def apply_fiso_mask(dat, iso_dict=None, isos=None, indices=None, fill_value=np.nan):
    """Mask DataArray using FISO dictionary or the flattened indexes.

    Args:
        dat: xarray.DataArray instance to be filtered (rho, phi, etc.)
        iso_dict: FISO object dictionary, optional.
        isos: int or sequence of ints representing the selected isos, optional.
        indices: FISO flattend indices, optional. If given, overrides iso_dict and isos.
        fill_value: value to fill outside of the filtered region, optional.
                    Default value is np.nan.

    Returns:
        out: Filtered DataArray
    """
    if iso_dict is None and isos is None and indices is None:
        # nothing to do
        return dat
    elif iso_dict is not None and indices is None:
        indices = []
        if isos is None:
            # select all cells
            for v in iso_dict.values():
                indices += list(v)
        elif isinstance(isos, int):
            indices += iso_dict[isos]
        else:
            for iso in isos:
                indices += iso_dict[iso]
    dat1d = dat.data.flatten()
    out = np.full(len(dat1d), fill_value)
    out[indices] = dat1d[indices]
    out = out.reshape(dat.shape)
    out = xr.DataArray(data=out, coords=dat.coords, dims=dat.dims)
    return out


def get_coords_minimum(dat):
    """returns coordinates at the minimum of dat

    Args:
        dat : xarray.DataArray instance (usually potential)
    Returns:
        x0, y0, z0
    """
    center = dat.argmin(...)
    x0, y0, z0 = [dat.isel(center).coords[dim].data[()]
                  for dim in ['x', 'y', 'z']]
    return x0, y0, z0


def calculate_cum_energies(ds, iso_dict, iso, mode='HBR', boundary_flag='periodic'):
    """Calculate cumulative energies for all levels

    Args:
        ds: xarray.Dataset instance containing primitive variables
        iso_dict: FISO object dictionary
        iso: ID of the iso
        mode: Definition of the boundness, optional
              Available options: ( 'HBR' | 'HBR+1' | 'HBR-1' | 'virial' )
              Default value is 'HBR'.
        boundary_flag: ( periodic | outflow ), optional

    Returns:
        energies : dict containing integrated energies and effective radius at each level
    """
    # Create 1-D flattened primitive variables of this iso
    indices = np.array(iso_dict[iso])
    ds = ds.transpose('z', 'y', 'x')
    dat1d = dict(indices=indices,
                 x=np.broadcast_to(ds.x.data, ds.dens.shape
                                   ).flatten()[indices],
                 y=np.broadcast_to(ds.y.data, ds.dens.shape
                                   ).transpose(1, 2, 0).flatten()[indices],
                 z=np.broadcast_to(ds.z.data, ds.dens.shape
                                   ).transpose(2, 0, 1).flatten()[indices],
                 rho=ds.dens.data.flatten()[indices],
                 vx=ds.vel1.data.flatten()[indices],
                 vy=ds.vel2.data.flatten()[indices],
                 vz=ds.vel3.data.flatten()[indices],
                 prs=ds.prs.data.flatten()[indices],
                 phi=ds.phi.data.flatten()[indices])
    # Assume uniform grid
    dV = ds.dx*ds.dy*ds.dz
    gm1 = (5./3. - 1)
    Ncells = len(indices)

    # Sort variables in ascending order of potential
    indices_sorted = dat1d['phi'].argsort()
    dat1d = {key: value[indices_sorted] for key, value in dat1d.items()}
    indices = dat1d['indices']

    # Gravitational potential at the HBP boundary
    phi0 = dat1d['phi'][-1]

    # Calculate the center of momentum frame
    # note: dV factor is omitted
    M = (dat1d['rho']).cumsum()
    vx0 = (dat1d['rho']*dat1d['vx']).cumsum() / M
    vy0 = (dat1d['rho']*dat1d['vy']).cumsum() / M
    vz0 = (dat1d['rho']*dat1d['vz']).cumsum() / M
    # Potential minimum
    phi_hpr = apply_fiso_mask(ds.phi, indices=indices)
    x0, y0, z0 = get_coords_minimum(phi_hpr)

    # Kinetic energy
    # \int 0.5 \rho | v - v_com |^2 dV
    # = \int 0.5 \rho |v|^2 dV - (\int 0.5\rho dV) |v_com|^2
    # Note that v_com depends on the limit of the volume integral.
    Ekin = (0.5*dat1d['rho']*(dat1d['vx']**2
                              + dat1d['vy']**2
                              + dat1d['vz']**2)*dV).cumsum()
    Ekin -= (0.5*dat1d['rho']*dV).cumsum()*(vx0**2 + vy0**2 + vz0**2)

    # Thermal energy
    Eth = (dat1d['prs']/gm1*dV).cumsum()

    # Gravitational energy
    if mode == 'HBR' or mode == 'HBR+1' or mode == 'HBR-1':
        Egrav = (dat1d['rho']*(dat1d['phi'] - phi0)*dV).cumsum()
    elif mode == 'virial':
        dat1d['gx'] = -ds.phi.differentiate('x').data.flatten()[indices]
        dat1d['gy'] = -ds.phi.differentiate('y').data.flatten()[indices]
        dat1d['gz'] = -ds.phi.differentiate('z').data.flatten()[indices]
        Egrav = (dat1d['rho']*((dat1d['x'] - x0)*dat1d['gx']
                               + (dat1d['y'] - y0)*dat1d['gy']
                               + (dat1d['z'] - z0)*dat1d['gz'])*dV).cumsum()
    else:
        raise ValueError("Unknown mode; select (HBR | HBR+1 | HBR-1 | virial)")

    # Surface terms
    if mode == 'HBR':
        Ekin0 = Eth0 = np.zeros(Ncells)
    elif mode == 'HBR+1' or mode == 'HBR-1':
        pcn = boundary.precompute_neighbor(ds.phi.shape, boundary_flag)
        edge = get_edge_cells(indices, pcn)
        edg1d = dict(rho=ds.dens.data.flatten()[edge],
                     vx=ds.vel1.data.flatten()[edge],
                     vy=ds.vel2.data.flatten()[edge],
                     vz=ds.vel3.data.flatten()[edge],
                     prs=ds.prs.data.flatten()[edge])
        # COM velocity of edge cells
        M = (edg1d['rho']).sum()
        vx0 = (edg1d['rho']*edg1d['vx']).sum() / M
        vy0 = (edg1d['rho']*edg1d['vy']).sum() / M
        vz0 = (edg1d['rho']*edg1d['vz']).sum() / M
        # Mean surface energies
        Ekin0 = (0.5*edg1d['rho']*((edg1d['vx'] - vx0)**2
                                   + (edg1d['vy'] - vy0)**2
                                   + (edg1d['vz'] - vz0)**2)).mean()
        Eth0 = (edg1d['prs']/gm1).mean()
        # Integrated surface energy to compare with volume energies.
        # Note that the excess energy is given by \int (E - E_0) dV
        Ekin0 = (Ekin0*np.ones(Ncells)*dV).cumsum()
        Eth0 = (Eth0*np.ones(Ncells)*dV).cumsum()
    elif mode == 'virial':
        divPx = ((ds.prs*(ds.x - x0)).differentiate('x')
                 + (ds.prs*(ds.y - y0)).differentiate('y')
                 + (ds.prs*(ds.z - z0)).differentiate('z')
                 ).data.flatten()[indices]
        Eth0 = ((1./3.)*divPx/gm1*dV).cumsum()
        # Kinetic energy surface term
        v0 = np.array([vx0, vy0, vz0])
        # A1
        rho_rdotv = ds.dens*((ds.x - x0)*ds.vel1
                              + (ds.y - y0)*ds.vel2
                              + (ds.z - z0)*ds.vel3)
        A1 = ((rho_rdotv*ds.vel1).differentiate('x')
              + (rho_rdotv*ds.vel2).differentiate('y')
              + (rho_rdotv*ds.vel3).differentiate('z'))
        A1 = (A1.data.flatten()[indices]*dV).cumsum()
        # A2
        grad_rho_r = np.empty((3, 3), dtype=xr.DataArray)
        for i, crd_i in enumerate(['x', 'y', 'z']):
            for j, (crd_j, pos0_j) in enumerate(zip(['x', 'y', 'z'], [x0, y0, z0])):
                grad_rho_r[i,j] = (ds.dens*(ds[crd_j] - pos0_j)
                                   ).differentiate(crd_i)
        A2 = np.empty((3, 3, Ncells))
        for i, crd_i in enumerate(['x', 'y', 'z']):
            for j, (crd_j, pos0_j) in enumerate(zip(['x', 'y', 'z'], [x0, y0, z0])):
                A2[i, j, :] = (grad_rho_r[i, j].data.flatten()[indices]*dV).cumsum()
        A2 = np.einsum('i..., ij..., j...', v0, A2, v0)
        # A3
        grad_rho_rdotv = np.empty(3, dtype=xr.DataArray)
        for i, crd_i in enumerate(['x', 'y', 'z']):
            grad_rho_rdotv[i] = rho_rdotv.differentiate(crd_i)
        A3 = np.empty((3, Ncells))
        for i, crd_i in enumerate(['x', 'y', 'z']):
            A3[i, :] = (grad_rho_rdotv[i].data.flatten()[indices]*dV).cumsum()
        A3 = np.einsum('i...,i...', v0, A3)
        # A4
        div_rhorv = np.empty(3, dtype=xr.DataArray)
        for i, (crd_i, pos0_i) in enumerate(zip(['x', 'y', 'z'],
                                                [x0, y0, z0])):
            div_rhorv[i] = ((ds.dens*(ds[crd_i] - pos0_i)*ds.vel1).differentiate('x')
                          + (ds.dens*(ds[crd_i] - pos0_i)*ds.vel2).differentiate('y')
                          + (ds.dens*(ds[crd_i] - pos0_i)*ds.vel3).differentiate('z'))
        A4 = np.empty((3, Ncells))
        for i, crd_i in enumerate(['x', 'y', 'z']):
            A4[i, :] = (div_rhorv[i].data.flatten()[indices]*dV).cumsum()
        A4 = np.einsum('i...,i...', v0, A4)
        Ekin0 = 0.5*(A1 + A2 - A3 - A4)

    Reff = ((np.ones(Ncells)*dV).cumsum() / (4.*np.pi/3.))**(1./3.)
    if mode == 'HBR':
        Etot = Ekin + Eth + Egrav
    elif mode == 'HBR+1':
        Etot = (Ekin - Ekin0) + (Eth - Eth0) + Egrav
    elif mode == 'HBR-1':
        Etot = (Ekin + Ekin0) + (Eth + Eth0) + Egrav
    elif mode == 'virial':
        Etot = 2*(Ekin - Ekin0) + 3*gm1*(Eth - Eth0) + Egrav

    energies = dict(Reff=Reff, Ekin=Ekin, Eth=Eth, Ekin0=Ekin0, Eth0=Eth0,
                    Egrav=Egrav, Etot=Etot)
    return energies


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
