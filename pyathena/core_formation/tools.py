import numpy as np
import xarray as xr
from scipy.special import erfinv
from scipy import odr
from pyathena.util import transform
from pyathena.core_formation import load_sim_core_formation
from pyathena.core_formation import tes
from pyathena.core_formation.exceptions import NoNearbyCoreError


class LognormalPDF:
    """Lognormal probability distribution function"""

    def __init__(self, Mach, b=0.4, weight='mass'):
        """Constructor of the LognormalPDF class

        Parameter
        ---------
        Mach : float
            Sonic Mach number
        b : float, optional
            Parameter in the density dispersion-Mach number relation.
            Default to 0.4, corresponding to natural mode mixture.
            See Fig. 8 of Federrath et al. 2010.
        weight : string, optional
            Weighting of the PDF. Default to mass-weighting.
        """
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
        """The mass fraction between x and x+dx

        Parameter
        ---------
        x : float
            Logarithmic density contrast, ln(rho/rho_0).
        """
        f = (1 / np.sqrt(2*np.pi*self.var))*np.exp(-(x - self.mu)**2
                                                   / (2*self.var))
        return f

    def get_contrast(self, frac):
        """Calculates density contrast for given mass coverage

        Returns rho/rho_0 below which frac (0 to 1) of the total mass
        is contained.

        Parameter
        ---------
        frac : float
            Mass fraction.
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
    s : LoadSimCoreFormation
        Object containing simulation metadata.
    rprf : xarray Dataset
        Object containing radial profiles.
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
        vsq = rprf['dvel1_sq_mw']
    elif use_vel == 'total':
        vsq = rprf['vel1_sq_mw']
    else:
        ValueError("Unknown option for use_vel")

    # Select the region for fitting the velocity-size relation.
    Mach = np.sqrt(vsq.data) / s.cs
    Mach_threshold = 1.5
    idx = np.where(Mach < Mach_threshold)[0][-1]
    Rmax = rprf.r.isel(r=idx).data[()]
    r = rprf.r.sel(r=slice(0, Rmax)).data[1:]
    vr = np.sqrt(vsq.sel(r=slice(0, Rmax)).data[1:])
    rhoc = rprf.rho.isel(r=0).data[()]
    LJ_c = 1.0 / np.sqrt(rhoc)
    MJ_c = 1.0 / np.sqrt(rhoc)

    if len(r) < 1:
        # Sonic radius is zero. Cannot find critical tes.
        p = np.nan
        rs = np.nan
        rhoe = np.nan
        rcrit = np.nan
        mcrit = np.nan
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
        mydata = odr.Data(np.log(r), np.log(vr/s.cs))
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
            mcrit = ts.get_mass(xi_crit)*MJ_c
        except ValueError:
            rcrit = np.nan
            mcrit = np.nan
            rhoe = np.nan

    res = dict(center_density=rhoc, edge_density=rhoe, critical_radius=rcrit,
               critical_mass=mcrit, pindex=p, sonic_radius=rs)
    return res


def calculate_radial_profiles(s, ds, origin, rmax):
    """Calculates radial profiles of various properties at selected position

    Parameters
    ----------
    s : LoadSimCoreFormation
        Object containing simulation metadata.
    ds : xarray.Dataset
        Object containing simulation data.
    origin : tuple-like
        Coordinate origin (x0, y0, z0).
    rmax : float
        Maximum radius of radial bins.

    Returns
    -------
    rprof : xarray.Dataset
        Angle-averaged radial profiles.

    Notes
    -----
    vel1, vel2, vel3: density-weighted mean velocities (v_r, v_theta, v_phi).
    vel1_sq, vel2_sq, vel3_sq: density-weighted mean squared velocities.
    ggas1, ggas2, ggas3: density-weighted mean gravity due to gas.
    gstar1, gstar2, gstar3: density-weighted mean gravity due to stars.
    """
    # Convert density and velocities to spherical coord.
    ds['phistar'] = ds['phi'] - ds['phigas']
    vel, ggas, gstar = {}, {}, {}
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
    div_v = (vel['x'].differentiate('x')
             + vel['y'].differentiate('y')
             + vel['z'].differentiate('z'))
    ds_sph['div_v'] = div_v.assign_coords(dict(r=r))

    # Radial binning
    edges = np.insert(np.arange(ds.dx1/2, rmax, ds.dx1), 0, 0)
    rprf = {}

    for k in ['rho', 'div_v']:
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


def find_rtidal_envelop(s, pid, tol=1.1):
    """Finds upper envelop of tidal radius evolution

    Parameters
    ----------
    s : LoadSimCoreFormation
        Object containing simulation metadata.
    pid : int
        Unique particle ID.
    tol : float, optional
        Tolerance in the temporal discontinuity.

    Returns
    -------
    rtidal : array
        Envelop tidal radius
    """
    cores = s.cores[pid].sort_values('num', ascending=False)
    rr = cores.iloc[0].radius
    rtidal = []
    for num, core in cores.iterrows():
        rl = core.radius
        nid = core.nid
        while rl < rr/tol:
            gd = s.load_dendrogram(num)
            nid = gd.parent[nid]
            vol = len(gd.get_all_descendant_cells(nid))*s.dV
            rparent = (3*vol/(4*np.pi))**(1./3.)
            if rparent < rr*tol:
                rl = rparent
            else:
                break
        rtidal.append(rl)
        rr = rl
    rtidal = np.array(rtidal)[::-1]
    return rtidal


def get_accelerations(rprf):
    """Calculate RHS of the Lagrangian EOM (force per unit mass)

    Parameters
    ----------
    rprf : xarray.Dataset
        Radial profiles

    Returns
    -------
    acc : xarray.Dataset
        Accelerations appearing in Lagrangian EOM

    """
    if 'num' in rprf.indexes:
        # Temporary patch to the xarray bug;
        # When there are multiple indexes associated with the same
        # dimension 't', calculation among arrays are disabled.
        # So drop 'num' index.
        rprf = rprf.drop_indexes('num')
    pthm = rprf.rho
    ptrb = rprf.rho*rprf.dvel1_sq_mw
    acc = dict(adv=rprf.vel1_mw*rprf.vel1_mw.differentiate('r'),
               thm=-pthm.differentiate('r') / rprf.rho,
               trb=-ptrb.differentiate('r') / rprf.rho,
               cen=(rprf.vel2_mw**2 + rprf.vel3_mw**2) / rprf.r,
               grv=rprf.ggas1_mw + rprf.gstar1_mw,
               ani=((rprf.dvel2_sq_mw + rprf.dvel3_sq_mw - 2*rprf.dvel1_sq_mw)
                    / rprf.r))
    acc = xr.Dataset(acc)
    acc['dvdt_lagrange'] = acc.thm + acc.trb + acc.grv + acc.cen + acc.ani
    acc['dvdt_euler'] = acc.dvdt_lagrange - acc.adv
    return acc


def find_tcoll_core(s, pid):
    """Find the GRID-dendro ID of the t_coll core of particle pid"""
    # load dendrogram at t = t_coll
    num = s.tcoll_cores.loc[pid].num
    gd = s.load_dendrogram(num)

    # find closeast leaf node to this particle
    dx, dy, dz = s.domain['dx']
    dst_inc = min(dx, dy, dz)
    search_dst = dst_inc
    vel = s.tcoll_cores.loc[pid][['v1', 'v2', 'v3']].to_numpy()
    particle_speed = np.sqrt((vel**2).sum())
    search_dst_max = max(20*max(dx, dy, dz),
                         2*s.dt_output['hdf5']*particle_speed)
    tcoll_core = set()
    while len(tcoll_core) == 0:
        for leaf in gd.leaves:
            kji = np.unravel_index(leaf, s.domain['Nx'][::-1], order='C')
            ijk = np.array(kji)[::-1]
            pos_node = s.domain['le'] + ijk*s.domain['dx']
            pos_particle = s.tcoll_cores.loc[pid][['x1', 'x2', 'x3']]
            pos_particle = pos_particle.to_numpy()
            dst = get_periodic_distance(pos_node, pos_particle, s.Lbox)
            if dst <= search_dst:
                tcoll_core.add(leaf)
        search_dst += dst_inc
        if search_dst > search_dst_max:
            msg = "pid = {}: Cannot find a t_coll core within distance {}"
            msg = msg.format(pid, search_dst_max)
            raise NoNearbyCoreError(msg)
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


def get_coords_node(s, nd):
    """Get coordinates of the generating point of this node

    Parameters
    ----------
    s : LoadSimCoreFormation
        Simulation metadata.
    nd : int
        GRID-dendro node ID.

    Returns
    -------
    coordinates: tuple representing physical coordinates (x, y, z)
    """
    k, j, i = np.unravel_index(nd, s.domain['Nx'].T, order='C')
    coordinates = (s.domain['le']
                   + np.array([i+0.5, j+0.5, k+0.5])*s.domain['dx'])
    return coordinates


def get_resolution_requirement(Mach, Lbox, mfrac=None, rho_amb=None,
                               ncells_min=10):
    """Print resolution requirements

    Parameters
    ----------
    Mach : float
        Mach number
    Lbox : float
        Box size
    mfrac : float, optional
        Cumulative mass fraction in a mass-weighted density pdf.
    rho_amb : float, optional
        Ambient density. If given, override mfrac.
    ncells_min : int, optional
        Minimum number of cells to resolve critical TES.
    """
    if mfrac is None and rho_amb is None:
        raise ValueError("Specify either mfrac or rho_amb")
    s = load_sim_core_formation.LoadSimCoreFormation(Mach)
    lmb_sonic = get_sonic(Mach, Lbox)
    if rho_amb is None:
        rho_amb = s.get_contrast(mfrac)
    rhoc_BE, R_BE, M_BE = tes.get_critical_tes(rhoe=rho_amb, lmb_sonic=np.inf)
    rhoc_TES, R_TES, M_TES = tes.get_critical_tes(rhoe=rho_amb,
                                                  lmb_sonic=lmb_sonic)
    R_LP_BE = s.get_RLP(M_BE)
    R_LP_TES = s.get_RLP(M_TES)
    dx_req_LP = R_LP_BE/ncells_min
    dx_req_BE = R_BE/ncells_min
    ncells_req_LP = np.ceil(Lbox/dx_req_LP).astype(int)
    ncells_req_BE = np.ceil(Lbox/dx_req_BE).astype(int)

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
    print("Required resolution dx to resolve LP core = {}".format(dx_req_LP))
    print("Required resolution Ncells to resolve LP core = {}".format(
        ncells_req_LP))
    print("Required resolution dx to resolve BE sphere = {}".format(dx_req_BE))
    print("Required resolution Ncells to resolve BE sphere = {}".format(
        ncells_req_BE))


def get_sonic(Mach_outer, l_outer, p=0.5):
    """returns sonic scale assuming linewidth-size relation v ~ R^p
    """
    if Mach_outer == 0:
        return np.inf
    lambda_s = l_outer*Mach_outer**(-1/p)
    return lambda_s


def recenter_dataset(ds, center):
    """Recenter whole dataset or dataarray.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Dataset to be recentered.
    center : tuple
        New (x, y, z) coordinates of the center.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Recentered dataset.
    tuple
        Position of the new center. This must be the grid coordinates
        closest, but not exactly the same, to (0, 0, 0).
    """
    if isinstance(ds, xr.Dataset):
        shape = np.array(list(ds.dims.values()), dtype=int)
    elif isinstance(ds, xr.DataArray):
        shape = np.array(ds.shape, dtype=int)
    else:
        TypeError("Data type {} is not supported".format(type(ds)))
    hNz, hNy, hNx = shape >> 1
    xc, yc, zc = center
    dx = ds.x.data[1] - ds.x.data[0]
    dy = ds.y.data[1] - ds.y.data[0]
    dz = ds.z.data[1] - ds.z.data[0]
    ishift = hNx - np.where(np.isclose(ds.x.data, xc, atol=1e-1*dx))[0][0]
    jshift = hNy - np.where(np.isclose(ds.y.data, yc, atol=1e-1*dy))[0][0]
    kshift = hNz - np.where(np.isclose(ds.z.data, zc, atol=1e-1*dz))[0][0]
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


def test_resolved_core(s, pid, ncells_min=10):
    """Test if the given core is sufficiently resolved.

    Parameters
    ----------
    s : LoadSimCoreFormation
        Object containing simulation metadata
    pid : int
        Unique ID of the particle.
    ncells_min : int
        Minimum grid distance between a core and a particle.

    Returns
    -------
    bool
        True if a core is isolated, false otherwise.
    """
    if s.cores[pid].iloc[-1].radius / s.dx > ncells_min:
        return True
    else:
        return False


def test_isolated_core(s, pid):
    """Test if the given core is isolated.

    Criterion for an isolated core is that the core must not contain
    any particle at the time of collapse.

    Parameters
    ----------
    s : LoadSimCoreFormation
        Object containing simulation metadata.
    pid : int
        Unique ID of the particle.

    Returns
    -------
    bool
        True if a core is isolated, false otherwise.
    """
    num_tcoll = s.tcoll_cores.loc[pid].num
    pds = s.load_partab(num_tcoll)
    gd = s.load_dendrogram(num_tcoll)

    nid = s.cores[pid].loc[num_tcoll].nid

    # Get all cells in this node.
    cells = set(gd.get_all_descendant_cells(nid))

    # Test whether there are any existing particle.
    position_indices = np.floor((pds[['x1', 'x2', 'x3']] - s.domain['le'])
                                / s.domain['dx']).astype('int')
    flatidx = (position_indices['x3']*s.domain['Nx'][2]*s.domain['Nx'][1]
               + position_indices['x2']*s.domain['Nx'][1]
               + position_indices['x1'])
    return not np.any([i in cells for i in flatidx])


def get_critical_core_props(s, pid, e1=0.7, e2=0.4):
    """Calculate core properties at the time when it becomes unstable.

    Parameters
    ----------
    s : LoadSimCoreFormation
        Object containing simulation metadata.
    pid : int
        Particle id.
    e1 : float, optional
        Multiplier to t_ff for left bin edge.
    e2 : float, optional
        Multiplier to t_ff for right bin edge.

    Returns
    -------
    cprops : pandas.Series
        Core properties at the time when it becomes unstable.

    Notes
    -----
    It is difficult to pinpoint the time when a given core becomes unstable.
    Instead, it can be approximately inferred by rewinding ~ t_ff from t_coll.
    Empirically, it is found that the required amount of rewinding is
    ~ 0.6 t_ff. The optional parameters e1 and e2 defines the bin edge
    such that the averaging is performed between
      t_coll - e1*t_ff < t < t_coll - e2*t_ff.
    """
    cores = s.cores[pid]
    tcoll = s.tcoll_cores.loc[pid].time
    num_tcoll = s.tcoll_cores.loc[pid].num
    # Mean free-fall time at t = t_coll
    tff = np.sqrt(3*np.pi/(32*cores.mean_density.loc[num_tcoll]))
    t1 = tcoll - e1*tff
    t2 = tcoll - e2*tff
    mask = (cores.time > t1) & (cores.time < t2)
    cores = cores[mask]
    rprf = s.rprofs[pid].sel(num=cores.index)
    rhoe = []
    for num, core in cores.iterrows():
        rhoe.append(rprf.rho.sel(num=num).interp(r=core.radius))
    rhoe = np.array(rhoe).mean()
    cprops = cores.mean()
    cprops['t1'] = t1
    cprops['t2'] = t2
    cprops['rhoe'] = rhoe
    return cprops


def get_periodic_distance(pos1, pos2, Lbox):
    hLbox = 0.5*Lbox
    rds2 = 0
    for x1, x2 in zip(pos1, pos2):
        dst = np.abs(x1-x2)
        dst = Lbox - dst if dst > hLbox else dst
        rds2 += dst**2
    dst = np.sqrt(rds2)
    return dst


def get_node_distance(s, nd1, nd2):
    """Calculate periodic distance between two nodes

    Parameters
    ----------
    s : LoadSimCoreFormation
        Simulation metadata.
    nd1 : int
        GRID-dendro node ID
    nd2 : int
        GRID-dendro node ID
    """
    pos1 = get_coords_node(s, nd1)
    pos2 = get_coords_node(s, nd2)
    # TODO generalize this
    dst = get_periodic_distance(pos1, pos2, s.Lbox)
    return dst
