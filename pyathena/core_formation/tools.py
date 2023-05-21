import numpy as np
import xarray as xr
from scipy.special import erfinv
from scipy import odr
from pyathena.util import transform
from pyathena.core_formation import load_sim_core_formation
from pyathena.core_formation import tes


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
        if weight == 'mass':
            pass
        elif weight == 'volume':
            self.mu *= -1
        else:
            ValueError("weight must be either mass or volume")

    def fx(self, x):
        """The mass fraction between x and x+dx, where x = ln(rho/rho_0)"""
        f = (1 / np.sqrt(2*np.pi*self.var))*np.exp(-(x - self.mu)**2
                                                   / (2*self.var))
        return f

    def get_contrast(self, frac):
        """
        Returns rho/rho_0 below which frac (0 to 1) of the total mass
        is contained.
        """
        x = self.mu + np.sqrt(2)*self.sigma*erfinv(2*frac - 1)
        return np.exp(x)


def calculate_critical_tes(s, rprf, use_vel='disp', fixed_slope=False):
    """Calculates critical tes given the radial profile.

    Given the radial profile, find the critical tes at the same central
    density. return the ambient density, radius, power law index, and the sonic
    scale.

    Parameters
    ----------
    use_vel : str, optional
        If 'total', use <v_r^2> to find sonic radius.
        If 'disp', use <dv_r^2> = <v_r^2> - <v_r>^2 to find sonic radius.
    fixed_slope : bool, optional
        If true, fix the slope of velocity-size relation to 0.5.

    Returns
    -------
    res : dict
        center_density, edge_density, critical_radius, pindex, sonic_radius
    """
    if use_vel == 'disp':
        # select the subsonic portion for fitting
        vsq = rprf['vel1_sq_mw'] - rprf['vel1_mw']**2
    elif use_vel == 'total':
        vsq = rprf['vel1_sq_mw']
    else:
        ValueError("Unknown option for use_vel")

    # select the subsonic portion for fitting
    idx = np.where(vsq.data > 1)[0][0]
    Rmax = rprf.r.isel(r=idx).data[()]
    r = rprf.r.sel(r=slice(0, Rmax)).data[1:]
    vr = np.sqrt(vsq.sel(r=slice(0, Rmax)).data[1:])
    rhoc = rprf.rho.isel(r=0).data[()]
    LJ_c = 1.0 / np.sqrt(rhoc)

    if len(r) < 1:
        # Sonic radius is zero. Cannot find critical tes.
        p = np.nan
        rs = np.nan
        rhoe = np.nan
        rcrit = np.nan
    else:
        # fit the velocity dispersion to get power law index and sonic radius
        if fixed_slope:
            def f(B, x):
                return 0.5*x + B
            beta0 = [1,]
        else:
            def f(B, x):
                return B[0]*x + B[1]
            beta0 = [0.5, 1]

        linear = odr.Model(f)
        mydata = odr.Data(np.log(r), np.log(vr))
        myodr = odr.ODR(mydata, linear, beta0=beta0)
        myoutput = myodr.run()
        if fixed_slope:
            p = 0.5
            intercept = myoutput.beta[0]
        else:
            p, intercept = myoutput.beta
        rs = np.exp(-intercept/(p))

        # Find critical TES at the central density
        xi_s = rs / LJ_c
        ts = tes.TESc(p=p, xi_s=xi_s)
        try:
            xi_crit = ts.get_crit()
            u, du = ts.solve(xi_crit)
            rhoe = rhoc*np.exp(u[-1])
            rcrit = xi_crit*LJ_c
        except ValueError:
            rcrit = np.nan
            rhoe = np.nan

    res = dict(center_density=rhoc, edge_density=rhoe, critical_radius=rcrit,
               pindex=p, sonic_radius=rs)
    return res


def calculate_radial_profiles(s, ds, origin, rmax):
    """Calculates radial profiles of various properties at selected position

    Parameters
    ----------
    s : LoadSimCoreFormation
        Object containing simulation metadata
    ds : xarray.Dataset
        Object containing simulation data
    origin : tuple-like
        Coordinate origin (x0, y0, z0)
    rmax : float
        Maximum radius of radial bins.

    Returns
    -------
    rprof : xarray.Dataset
        Angle-averaged radial profiles

    Notes
    -----
    vel1, vel2, vel3: density-weighted mean velocities (v_r, v_theta, v_phi).
    vel1_sq, vel2_sq, vel3_sq: density-weighted mean squared velocities.
    ggas1, ggas2, ggas3: density-weighted mean gravity due to gas.
    gstar1, gstar2, gstar3: density-weighted mean gravity due to stars.
    """
    # Convert density and velocities to spherical coord.
    ds['phistar'] = ds['phi'] - ds['phigas']
    vel, ggas, gstar, grad_pthm, grad_ptrb = {}, {}, {}, {}, {}
    for dim, axis in zip(['x', 'y', 'z'], [1, 2, 3]):
        # Recenter velocity and calculate gravitational acceleration
        vel_ = ds['mom{}'.format(axis)]/ds.dens
        vel[dim] = vel_ - vel_.sel(x=origin[0], y=origin[1], z=origin[2])
        ggas[dim] = -ds['phigas'].differentiate(dim)
        gstar[dim] = -ds['phistar'].differentiate(dim)
    ds_sph = {}
    r, (ds_sph['vel1'], ds_sph['vel2'], ds_sph['vel3'])\
        = transform.to_spherical(vel.values(), origin)
    _, (ds_sph['ggas1'], ds_sph['ggas2'], ds_sph['ggas3'])\
        = transform.to_spherical(ggas.values(), origin)
    _, (ds_sph['gstar1'], ds_sph['gstar2'], ds_sph['gstar3'])\
        = transform.to_spherical(gstar.values(), origin)
    ds_sph['rho'] = ds.dens.assign_coords(dict(r=r))
    div_v = vel['x'].differentiate('x') + vel['y'].differentiate('y')\
        + vel['z'].differentiate('z')
    ds_sph['div_v'] = div_v.assign_coords(dict(r=r))

    # Calculate pressure gradient forces and transform to spherical coord.
    pthm = ds.dens*s.cs**2
    ptrb = ds.dens*ds_sph['vel1']**2
    for dim in ['x', 'y', 'z']:
        grad_pthm[dim] = pthm.differentiate(dim)
        grad_ptrb[dim] = ptrb.differentiate(dim)
    _, (ds_sph['grad_pthm1'], ds_sph['grad_pthm2'], ds_sph['grad_pthm3'])\
        = transform.to_spherical(grad_pthm.values(), origin)
    _, (ds_sph['grad_ptrb1'], ds_sph['grad_ptrb2'], ds_sph['grad_ptrb3'])\
        = transform.to_spherical(grad_ptrb.values(), origin)

    # Radial binning
    edges = np.insert(np.arange(ds.dx1/2, rmax, ds.dx1), 0, 0)
    rprf = {}

    for k in ['grad_pthm1', 'grad_ptrb1', 'rho', 'div_v']:
        rprf[k] = transform.groupby_bins(ds_sph[k], 'r', edges)
    # We can use weighted groupby_bins, but let's do it like this to reuse
    # rprf['rho'] for performance
    for k in ['ggas1', 'gstar1', 'vel1', 'vel2', 'vel3']:
        rprf[k] = transform.groupby_bins(ds_sph[k], 'r', edges)
        rprf[k+'_mw'] = transform.groupby_bins(ds_sph['rho']*ds_sph[k],
                                               'r', edges) / rprf['rho']
    for k in ['vel1', 'vel2', 'vel3']:
        rprf[k+'_sq'] = transform.groupby_bins(ds_sph[k]**2, 'r', edges)
        rprf[k+'_sq_mw'] = transform.groupby_bins(ds_sph['rho']*ds_sph[k]**2,
                                                  'r', edges) / rprf['rho']

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
    search_dst_max = max(10*max(dx, dy, dz),
                         2*s.dt_output['hdf5']*particle_speed)
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
            msg = "pid = {}: Cannot find a t_coll core within distance {}"
            msg = msg.format(pid, search_dst_max)
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
    print("Equivalent LP radius for Bonner-Ebert sphere = "
          "{:.3f}".format(R_LP_BE))
    print("Equivalent LP radius for TES = {:.3f}".format(R_LP_TES))
    print("Required resolution dx to resolve BE sphere = {}".format(dx_req))
    print("Required resolution Ncells to resolve BE sphere = "
          "{}".format(Ncells_req))


def get_sonic(Mach_outer, l_outer, p=0.5):
    """returns sonic scale assuming linewidth-size relation v ~ R^p
    """
    if Mach_outer == 0:
        return np.inf
    lambda_s = l_outer*Mach_outer**(-1/p)
    return lambda_s


def recenter_dataset(ds, center):
    shape = np.array(list(ds.dims.values()), dtype=int)
    hNz, hNy, hNx = shape >> 1
    xc, yc, zc = center
    ishift = hNx - np.where(ds.x.data == xc)[0][0]
    jshift = hNy - np.where(ds.y.data == yc)[0][0]
    kshift = hNz - np.where(ds.z.data == zc)[0][0]
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
