import numpy as np
import xarray as xr
from scipy.special import erfinv
from pathlib import Path
from pyathena.util import transform
from pyathena.core_formation import load_sim_core_formation
from grid_dendro import boundary
from grid_dendro import dendrogram


class LognormalPDF:
    """
    Lognormal probability distribution function

    b is the order unity coefficient that depends on the ratio of the
    compressive and solenoidal modes in the turbulence
    (see Federrath 2010, fig. 8; zeta=0.5 corresponds to the natural
    mixture, at which b~0.4)

    """
    def __init__(self, Mach, b=0.4, weight='mass'):
        self.mu = 0.5*np.log(1 + b**2*Mach**2)
        self.var = 2*self.mu
        self.sigma = np.sqrt(self.var)
        if weight=='mass':
            pass
        elif weight=='volume':
            self.mu *= -1
        else:
            ValueError("weight must be either mass or volume")

    def fx(self,x):
        """The mass fraction between x and x+dx, where x = ln(rho/rho_0)"""
        f = (1/np.sqrt(2*np.pi*self.var))*np.exp(-(x - self.mu)**2/(2*self.var))
        return f

    def get_contrast(self, frac):
        """
        Returns rho/rho_0 below which frac (0 to 1) of the total mass
        is contained.
        """
        x = self.mu + np.sqrt(2)*self.sigma*erfinv(2*frac - 1)
        return np.exp(x)


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


def find_tcoll_core(s, pid):
    """Find the GRID-dendro ID of the t_coll core of particle pid"""
    # load GRID-dendro leaves at t = t_coll
    num = s.nums_tcoll[pid]
    leaves = s.load_leaves(num)

    # find closeast leaf node to this particle
    dx, dy, dz = s.domain['dx']
    dst_inc = min(dx, dy, dz)
    search_dst = dst_inc
    particle_speed = np.sqrt(s.vpx0[pid]**2 + s.vpy0[pid]**2 + s.vpz0[pid]**2)
    search_dst_max = max(10*max(dx, dy, dz), 2*s.dt_output['hdf5']*particle_speed)
    tcoll_core = set()
    while len(tcoll_core) == 0:
        for leaf in leaves:
            kji = np.unravel_index(leaf, s.domain['Nx'][::-1], order='C')
            ijk = np.array(kji)[::-1]
            pos_node = s.domain['le'] + ijk*s.domain['dx']
            pos_particle = np.array((s.xp0[pid], s.yp0[pid], s.zp0[pid]))
            dst = get_periodic_distance(pos_node, pos_particle, s.Lbox)
            if dst <= search_dst:
                tcoll_core.add(leaf)
        search_dst += dst_inc
        if search_dst > search_dst_max:
            msg = f"pid = {pid}: Cannot find a t_coll core within distance {search_dst_max}"
            raise ValueError(msg)
    return tcoll_core.pop()


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


def get_coords_node(ds, node):
    """Get coordinates of the generating point of this node

    Args:
        ds: xarray.Dataset instance containing conserved variables.
        node: int representing the selected node.

    Returns:
        coordinates: tuple representing physical coordinates (x, y, z)
    """
    k, j, i = np.unravel_index(node, ds.phi.shape, order='C')
    x = ds.x.isel(x=i).values[()]
    y = ds.y.isel(y=j).values[()]
    z = ds.z.isel(z=k).values[()]
    coordinates = (x, y, z)
    return coordinates


def calculate_cum_energies(ds, nodes, node, mode='HBR', boundary_flag='periodic'):
    """Calculate cumulative energies for all levels

    Args:
        ds: xarray.Dataset instance containing primitive variables
        nodes: grid_dendro nodes dictionary, optional.
        node: int representing the selected node
        mode: Definition of the boundness, optional
              Available options: ( 'HBR' | 'HBR+1' | 'HBR-1' | 'virial' )
              Default value is 'HBR'.
        boundary_flag: ( periodic | outflow ), optional

    Returns:
        energies : dict containing integrated energies and effective radius at each level
    """
    # Create 1-D flattened primitive variables of this node
    cells = np.array(nodes[node])
    ds = ds.transpose('z', 'y', 'x')
    dat1d = dict(cells=cells,
                 x=np.broadcast_to(ds.x.data, ds.dens.shape
                                   ).flatten()[cells],
                 y=np.broadcast_to(ds.y.data, ds.dens.shape
                                   ).transpose(1, 2, 0).flatten()[cells],
                 z=np.broadcast_to(ds.z.data, ds.dens.shape
                                   ).transpose(2, 0, 1).flatten()[cells],
                 rho=ds.dens.data.flatten()[cells],
                 vx=ds.vel1.data.flatten()[cells],
                 vy=ds.vel2.data.flatten()[cells],
                 vz=ds.vel3.data.flatten()[cells],
                 prs=ds.prs.data.flatten()[cells],
                 phi=ds.phi.data.flatten()[cells])
    # Assume uniform grid
    dV = ds.dx*ds.dy*ds.dz
    gm1 = (5./3. - 1)
    Ncells = len(cells)

    # Sort variables in ascending order of potential
    cells_sorted = dat1d['phi'].argsort()
    dat1d = {key: value[cells_sorted] for key, value in dat1d.items()}
    cells = dat1d['cells']

    # Gravitational potential at the HBP boundary
    phi0 = dat1d['phi'][-1]

    # Calculate the center of momentum frame
    # note: dV factor is omitted
    M = (dat1d['rho']).cumsum()
    vx0 = (dat1d['rho']*dat1d['vx']).cumsum() / M
    vy0 = (dat1d['rho']*dat1d['vy']).cumsum() / M
    vz0 = (dat1d['rho']*dat1d['vz']).cumsum() / M
    # Potential minimum
    phi_hpr = dendrogram.filter_by_node(ds.phi, cells=cells)
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
        dat1d['gx'] = -ds.phi.differentiate('x').data.flatten()[cells]
        dat1d['gy'] = -ds.phi.differentiate('y').data.flatten()[cells]
        dat1d['gz'] = -ds.phi.differentiate('z').data.flatten()[cells]
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
        edge = get_edge_cells(cells, pcn)
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
                 ).data.flatten()[cells]
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
        A1 = (A1.data.flatten()[cells]*dV).cumsum()
        # A2
        grad_rho_r = np.empty((3, 3), dtype=xr.DataArray)
        for i, crd_i in enumerate(['x', 'y', 'z']):
            for j, (crd_j, pos0_j) in enumerate(zip(['x', 'y', 'z'], [x0, y0, z0])):
                grad_rho_r[i,j] = (ds.dens*(ds[crd_j] - pos0_j)
                                   ).differentiate(crd_i)
        A2 = np.empty((3, 3, Ncells))
        for i, crd_i in enumerate(['x', 'y', 'z']):
            for j, (crd_j, pos0_j) in enumerate(zip(['x', 'y', 'z'], [x0, y0, z0])):
                A2[i, j, :] = (grad_rho_r[i, j].data.flatten()[cells]*dV).cumsum()
        A2 = np.einsum('i..., ij..., j...', v0, A2, v0)
        # A3
        grad_rho_rdotv = np.empty(3, dtype=xr.DataArray)
        for i, crd_i in enumerate(['x', 'y', 'z']):
            grad_rho_rdotv[i] = rho_rdotv.differentiate(crd_i)
        A3 = np.empty((3, Ncells))
        for i, crd_i in enumerate(['x', 'y', 'z']):
            A3[i, :] = (grad_rho_rdotv[i].data.flatten()[cells]*dV).cumsum()
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
            A4[i, :] = (div_rhorv[i].data.flatten()[cells]*dV).cumsum()
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


def get_resolution_requirement(Mach, Lbox, mfrac=None, rho_amb=None, N_LP=10):
    if mfrac is None and rho_amb is None:
        raise ValueError("Specify either mfrac or rho_amb")
    s = load_sim_core_formation.LoadSimCoreFormation(Mach)
    lmb_sonic = get_sonic(Mach, Lbox)
    if rho_amb is None:
        rho_amb = s.get_contrast(mfrac)
    rhoc_BE, R_BE, M_BE = s.get_critical_TES(rho_amb, np.inf, p=0.5)
    rhoc_TES, R_TES, M_TES = s.get_critical_TES(rho_amb, lmb_sonic, p=0.5)
    R_LP_BE = s.get_RLP(M_BE)
    R_LP_TES = s.get_RLP(M_TES)
    dx_req = R_LP_BE/N_LP
    Ncells_req = np.ceil(Lbox/dx_req).astype(int)

    print(f"Mach number = {Mach}")
    print("sonic length = {}".format(lmb_sonic))
    print("Ambient density={:.3f}".format(rho_amb))
    print("Bonner-Ebert mass = {:.3f}".format(M_BE))
    print("Bonner-Ebert radius = {:.3f}".format(R_BE))
    print("Bonner-Ebert central density = {:.3f}".format(rhoc_BE))
    print("Critical TES mass = {:.3f}".format(M_TES))
    print("Critical TES radius = {:.3f}".format(R_TES))
    print("Critical TES central density = {:.3f}".format(rhoc_TES))
    print("Equivalent LP radius for Bonner-Ebert sphere = {:.3f}".format(R_LP_BE))
    print("Equivalent LP radius for TES = {:.3f}".format(R_LP_TES))
    print("Required resolution dx to resolve BE sphere = {}".format(dx_req))
    print("Required resolution Ncells to resolve BE sphere= {}".format(Ncells_req))

def get_sonic(Mach_outer, l_outer, p=0.5):
    """returns sonic scale assuming linewidth-size relation v ~ R^p
    """
    if Mach_outer==0:
        return np.inf
    lambda_s = l_outer*Mach_outer**(-1/p)
    return lambda_s

def recenter_dataset(ds, center):
    shape = np.array(list(ds.dims.values()), dtype=int)
    hNz, hNy, hNx = shape >> 1
    xc, yc, zc = center
    ishift = hNx - np.where(ds.x.data==xc)[0][0]
    jshift = hNy - np.where(ds.y.data==yc)[0][0]
    kshift = hNz - np.where(ds.z.data==zc)[0][0]
    xc_new = ds.x.isel(x=hNx).data[()]
    yc_new = ds.y.isel(y=hNy).data[()]
    zc_new = ds.z.isel(z=hNz).data[()]
    return ds.roll(x=ishift, y=jshift, z=kshift), (xc_new, yc_new, zc_new)

def get_rhocrit_KM05(lmb_sonic):
    """Equation (17) of Krumholz & McKee (2005)

    Args:
        lmb_sonic: sonic length devided by Jeans length at mean density.
    Returns:
        rho_crit: critical density devided by mean density.
    """
    phi_x = 1.12
    rho_crit = (phi_x/lmb_sonic)**2
    return rho_crit

def roundup(a, decimal):
    return np.ceil(a*10**decimal) / 10**decimal

def rounddown(a, decimal):
    return np.floor(a*10**decimal) / 10**decimal

def get_periodic_distance(pos1, pos2, Lbox):
    hLbox = 0.5*Lbox
    rds2 = 0
    for x1, x2 in zip(pos1, pos2):
        dst = np.abs(x1-x2)
        dst = Lbox - dst if dst > hLbox else dst
        rds2 += dst**2
    dst = np.sqrt(rds2)
    return dst
