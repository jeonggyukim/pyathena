import numpy as np
import xarray as xr
import pandas as pd
from scipy.special import erfinv
from scipy import odr
from pyathena.util import transform
from pyathena.core_formation import load_sim_core_formation
from pyathena.core_formation import tes
from pyathena.core_formation import config


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


def find_tcoll_core(s, pid):
    """Find the GRID-dendro ID of the t_coll core of particle pid"""
    # load dendrogram at t = t_coll
    num = s.tcoll_cores.loc[pid].num
    gd = s.load_dendro(num)

    # find closeast leaf node to this particle
    pos_particle = s.tcoll_cores.loc[pid][['x1', 'x2', 'x3']]
    pos_particle = pos_particle.to_numpy()

    distance = []
    for leaf in gd.leaves:
        pos_node = get_coords_node(s, leaf)
        distance.append(get_periodic_distance(pos_node, pos_particle, s.Lbox))
    tcoll_core = gd.leaves[np.argmin(distance)]
    return tcoll_core


def track_cores(s, pid, tol=1.1, sub_frac=0.2):
    """
    Parameters
    ----------
    s : LoadSimCoreFormation
    pid : int
    tol : float, optional
    sub_frac : float, optional

    Returns
    -------
    cores : pandas.DataFrame
    """
    # start from t = t_coll and track backward
    nums = np.arange(s.tcoll_cores.loc[pid].num, config.GRID_NUM_START-1, -1)
    num = nums[0]
    msg = '[track_cores] processing model {} pid {} num {}'
    print(msg.format(s.basename, pid, num))
    gd = s.load_dendro(num)
    ds = s.load_hdf5(num, header_only=True)

    # Calculate effective radius of this leaf
    lid = find_tcoll_core(s, pid)
    rlf = reff_sph(gd.len(lid)*s.dV)

    # Do the tidal correction to neglect attached substructures.
    eid, _ = disregard_substructures(s, gd, lid, tol=sub_frac)
    renv = reff_sph(gd.len(eid)*s.dV)

    # Calculate tidal radius
    rtidal = calculate_tidal_radius(s, gd, eid, lid)

    nums_track = [num,]
    time = [ds['Time'],]
    leaf_id = [lid,]
    leaf_radius = [rlf,]
    envelop_id = [eid,]
    envelop_radius = [renv,]
    tidal_radius = [rtidal,]

    for num in nums[1:]:
        msg = '[track_cores] processing model {} pid {} num {}'
        print(msg.format(s.basename, pid, num))
        gd = s.load_dendro(num)
        ds = s.load_hdf5(num, header_only=True)

        # find closeast leaf to the previous preimage
        dst = [get_node_distance(s, leaf, leaf_id[-1]) for leaf in gd.leaves]
        lid = gd.leaves[np.argmin(dst)]
        rlf = reff_sph(gd.len(lid)*s.dV)

        # linear extrapolation to predict the envelop radius
        if len(envelop_radius) == 1:
            dr = 0
        elif leaf_radius[-1] > envelop_radius[-2]:
            # If the tracked leaf is larger than the future envelop, this
            # indicates fragmentation. In this case, extrapolation from
            # (smaller) envelop to (larger) leaf is supressed.
            dr = 0
        else:
            dr = envelop_radius[-1] - envelop_radius[-2]
        renv_predicted = envelop_radius[-1] + dr

        # Do the tidal correction to neglect attached substructures.
        eid, _ = disregard_substructures(s, gd, lid, tol=sub_frac)
        renv = reff_sph(gd.len(eid)*s.dV)
        if renv > renv_predicted*tol:
            # preimage is too large; the tidal correction may have been too generous.
            # Undo the tidal correction
            eid = lid
            renv = rlf

        # If preimage is too small, go to envelop
        while renv < renv_predicted/tol:
            if eid == gd.trunk:
                break
            parent = gd.parent[eid]
            rparent = reff_sph(gd.len(parent)*s.dV)
            if rparent < renv_predicted*tol:
                # Try going up in the hierarchy.
                # If parent joins continuously, accept it.
                eid = parent
                renv = rparent
            else:
                # Stop going up in the hierarchy.
                break

        # Correct tidal radius for the last time
        eid_try, _ = disregard_substructures(s, gd, eid, tol=sub_frac)
        renv_try = reff_sph(gd.len(eid_try)*s.dV)
        if renv_try < renv_predicted*tol:
            eid = eid_try
            renv = renv_try

        # Reset the leaf such that it is has the minimum potential inside the
        # envelop.
        if eid not in gd.leaves:
            lid = gd.find_minimum(eid)
            rlf = reff_sph(gd.len(lid)*s.dV)

        # Calculate tidal radius
        rtidal = calculate_tidal_radius(s, gd, eid, lid)

        fdst = get_node_distance(s, lid, leaf_id[-1]) / max(rtidal, tidal_radius[-1])
        if fdst > 1:
            break

        nums_track.append(num)
        time.append(ds['Time'])
        leaf_id.append(lid)
        leaf_radius.append(rlf)
        envelop_id.append(eid)
        envelop_radius.append(renv)
        tidal_radius.append(rtidal)

    # SMOON: Using dtype=object is to prevent automatic upcasting from int to float
    # when indexing a single row. Maybe there is a better approach.
    cores = pd.DataFrame(dict(time=time,
                              leaf_id=leaf_id,
                              leaf_radius=leaf_radius,
                              envelop_id=envelop_id,
                              envelop_radius=envelop_radius,
                              tidal_radius=tidal_radius),
                         index=nums_track, dtype=object).sort_index()
    return cores


def disregard_substructures(s, gd, node, tol):
    """Go up the dendrogram hierarchy by neglecting substructures.

    Parameters
    ----------
    s : LoadSimCoreFormation
        Object containing simulation metadata.
    gd : grid_dendro.Dendrogram
        Dendrogram object.
    node : int
        Input node to be corrected.
    tol : float
        Fraction of the effective radius below which a sibling is
        considered as subtructure.

    Returns
    -------
    nd : int
        ID of the corrected node.
    reff : float
        Effective radius of the corrected node.
    """
    nd = node
    while True:
        reff = reff_sph(gd.len(nd)*s.dV)
        if nd == gd.trunk:
            return nd, reff
        parent = gd.parent[nd]

        sib = gd.sibling(nd)
        reff_sibling = reff_sph(gd.len(sib)*s.dV)

        if reff_sibling < tol*reff:
            nd = parent
        else:
            break
    return nd, reff


def calculate_tidal_radius(s, gd, node, leaf=None):
    """Calculate tidal radius of this node

    Tidal radius is defined as the distance to the closest node, excluding
    itself and its descendants.

    Parameters
    ----------
    s : LoadSimCoreFormation
        Object containing simulation metadata.
    gd : grid_dendro.Dendrogram
        Dendrogram object.
    node : int
        ID of the grid-dendro node.

    Returns
    -------
    rtidal : float
        Tidal radius.
    """
    if node == gd.trunk:
        # If this node is a trunk, tidal radius is the half the box size,
        # assuming periodic boundary condition.
        return 0.5*s.Lbox
    if leaf is None:
        leaf = gd.find_minimum(node)
    nodes = set(gd.nodes.keys()) - set(gd.descendants[node]) - {node}
    dst = [get_node_distance(s, nd, leaf) for nd in nodes]
    rtidal = np.min(dst)
    return rtidal


def calculate_critical_tes(s, rprf, core, mode='tot'):
    """Calculates critical tes given the radial profile.

    Given the radial profile, find the critical tes at the same central
    density. return the ambient density, radius, power law index, and the sonic
    scale.

    Parameters
    ----------
    s : LoadSimCoreFormation
        Object containing simulation metadata.
    rprf : xarray.Dataset
        Object containing radial profiles.
    core : pandas.Series
        Object containing core informations

    Returns
    -------
    res : dict
        center_density, edge_density, critical_radius, pindex, sonic_radius
    """
    # Select data for sonic radius fit
    r = rprf.r.sel(r=slice(0, core.tidal_radius)).data[1:]
    vr = np.sqrt(rprf.dvel1_sq_mw.sel(r=slice(0, core.tidal_radius)).data[1:])

    # Set scale length and mass based on the center and edge densities
    rhoc = rprf.rho.isel(r=0).data[()]
    LJ_c = MJ_c = np.sqrt(s.rho0/rhoc)

    rhoe = rprf.rho.interp(r=core.tidal_radius).data[()]
    if mode=='tot':
        ptrb = (rprf.rho*rprf.dvel1_sq_mw).interp(r=core.tidal_radius).data[()]
        pthm = (rprf.rho*s.cs**2).interp(r=core.tidal_radius).data[()]
        LJ_e = MJ_e = np.sqrt(s.cs**2*s.rho0/(pthm + ptrb))
    else:
        LJ_e = MJ_e = np.sqrt(s.rho0/rhoe)

    mtidal = (4*np.pi*rprf.r**2*rprf.rho).sel(r=slice(0, core.tidal_radius)
                                              ).integrate('r').data[()]
    mean_density = mtidal / (4*np.pi*core.tidal_radius**3/3)

    if len(r) < 1:
        # Sonic radius is zero. Cannot find critical tes.
        p = np.nan
        rs = np.nan
        dcrit = np.nan
        rcrit = np.nan
        mcrit = np.nan
        dcrit_e = np.nan
        rcrit_e = np.nan
        mcrit_e = np.nan
    else:
        def f(B, x):
            return B[0]*x + B[1]
        beta0 = [0.5, 1]

        linear = odr.Model(f)
        mydata = odr.Data(np.log(r), np.log(vr/s.cs))
        myodr = odr.ODR(mydata, linear, beta0=beta0)
        myoutput = myodr.run()
        p, intercept = myoutput.beta
        rs = np.exp(-intercept/(p))

        # Find critical TES at the central density
        xi_s = rs / LJ_c
        tsc = tes.TESc(p=p, xi_s=xi_s)
        try:
            xi_crit = tsc.get_rcrit(mode=mode)
            u, du = tsc.solve(xi_crit)
            dcrit = np.exp(-u)
            rcrit = xi_crit*LJ_c
            mcrit = tsc.get_mass(xi_crit)*MJ_c
        except ValueError:
            rcrit = np.nan
            mcrit = np.nan
            dcrit = np.nan

        # Find critical TES at the edge density
        xi_s = rs / LJ_e
        try:
            tse = tes.TESe(p=p, xi_s=xi_s, mode=mode)
            uc, rc, mc = tse.get_crit()
            dcrit_e = np.exp(uc)
            rcrit_e = rc*LJ_e
            mcrit_e = mc*MJ_e
        except ValueError:
            rcrit_e = np.nan
            mcrit_e = np.nan
            dcrit_e = np.nan

    res = dict(tidal_mass=mtidal, center_density=rhoc, edge_density=rhoe,
               mean_density=mean_density, sonic_radius=rs, pindex=p,
               critical_contrast=dcrit, critical_radius=rcrit,
               critical_mass=mcrit, critical_contrast_e=dcrit_e,
               critical_radius_e=rcrit_e, critical_mass_e=mcrit_e)
    return res


def calculate_radial_profile(s, ds, origin, rmax, lvec=None):
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
    lvec : array, optional
        Angular momentum vector to align the polar axis.

    Returns
    -------
    rprof : xarray.Dataset
        Angle-averaged radial profiles.

    Notes
    -----
    vel1, vel2, vel3: density-weighted mean velocities (v_r, v_theta, v_phi).
    vel1_sq, vel2_sq, vel3_sq: density-weighted mean squared velocities.
    gacc1: density-weighted mean gravitational acceleration.
    """
    # Sometimes, tidal radius is so small that the angular momentum vector
    # Cannot be computed. In this case, fall back to default behavior.
    # (to_spherical will assume z axis as the polar axis).
    if (np.array(lvec)**2).sum() == 0:
        lvec = None

    # Slice data
    nmax = np.floor(rmax/s.dx) + 1
    edges = np.insert(np.arange(s.dx/2, (nmax + 1)*s.dx, s.dx), 0, 0)
    ds = ds.sel(x=slice(origin[0] - edges[-1], origin[0] + edges[-1]),
                y=slice(origin[1] - edges[-1], origin[1] + edges[-1]),
                z=slice(origin[2] - edges[-1], origin[2] + edges[-1]))

    # Convert density and velocities to spherical coord.
    vel, gacc = {}, {}
    for dim, axis in zip(['x', 'y', 'z'], [1, 2, 3]):
        # Recenter velocity and calculate gravitational acceleration
        vel_ = ds['mom{}'.format(axis)]/ds.dens
        vel[dim] = vel_ - vel_.sel(x=origin[0], y=origin[1], z=origin[2])
        gacc[dim] = -ds.phi.differentiate(dim)

    ds_sph = {}
    r, (ds_sph['vel1'], ds_sph['vel2'], ds_sph['vel3'])\
        = transform.to_spherical(vel.values(), origin, lvec)
    _, (ds_sph['gacc1'], ds_sph['gacc2'], ds_sph['gacc3'])\
        = transform.to_spherical(gacc.values(), origin, lvec)
    ds_sph['rho'] = ds.dens.assign_coords(dict(r=r))
    ds_sph['phi'] = ds.phi.assign_coords(dict(r=r))

    # Radial binning
    rprf = {}
    for k in ['rho']:
        rprf[k] = transform.groupby_bins(ds_sph[k], 'r', edges)
    # We can use weighted groupby_bins, but let's do it like this to reuse
    # rprf['rho'] for performance
    for k in ['gacc1', 'vel1', 'vel2', 'vel3', 'phi']:
        rprf[k+'_mw'] = transform.groupby_bins(ds_sph['rho']*ds_sph[k],
                                               'r', edges) / rprf['rho']
    for k in ['vel1', 'vel2', 'vel3']:
        rprf[k+'_sq_mw'] = transform.groupby_bins(ds_sph['rho']*ds_sph[k]**2,
                                                  'r', edges) / rprf['rho']

    rprf = xr.Dataset(rprf)
    return rprf


def calculate_cumulative_energies(s, rprf, core):
    """Calculate cumulative energies based on radial profiles

    Use the mass-weighted mean gravitational potential at the tidal radius
    as the reference point. Mass-weighted mean is appropriate if we want
    the d(egrv)/dr = 0 as R -> Rtidal.

    Parameters
    ----------
    s : LoadSimCoreFormation
        Object containing simulation metadata.
    rprf : xarray.Dataset
        Object containing radial profiles.
    core : pandas.Series
        Object containing core informations.

    Returns
    -------
    rprf : xarray.Dataset
        Object containing radial profiles, augmented by energy fields
    """
    # TODO(SMOON) change the argument core to rmax and
    # substitute tidal_radius below to rmax.
    # Also, return the bound radius.
    # from scipy.interpolate import interp1d
    # etot_f = interp1d(rprf.r, rprf.etot)
    # rcore = brentq(etot_f, rprf.r[1], core.tidal_radius)

    # Thermal energy
    gm1 = (5/3 - 1)
    ethm = (4*np.pi*rprf.r**2*s.cs**2*rprf.rho/gm1).cumulative_integrate('r')

    # Kinetic energy
    vsq = rprf.vel1_sq_mw + rprf.vel2_sq_mw + rprf.vel3_sq_mw
    vcomsq = rprf.vel1_mw**2 + rprf.vel2_mw**2 + rprf.vel3_mw**2
    ekin = ((4*np.pi*rprf.r**2*0.5*rprf.rho*vsq).cumulative_integrate('r')
            - vcomsq*(4*np.pi*rprf.r**2*0.5*rprf.rho).cumulative_integrate('r'))

    # Gravitational energy
    phi0 = rprf.phi_mw.interp(r=core.tidal_radius)
    egrv = ((4*np.pi*rprf.r**2*rprf.rho*rprf.phi_mw).cumulative_integrate('r')
            - phi0*(4*np.pi*rprf.r**2*rprf.rho).cumulative_integrate('r'))

    rprf['ethm'] = ethm
    rprf['ekin'] = ekin
    rprf['egrv'] = egrv
    rprf['etot'] = ethm + ekin + egrv

    return rprf


def calculate_infall_rate(rprofs, cores):
    time, vr, mdot = [], [], []
    for num, rtidal in cores.tidal_radius.items():
        rprf = rprofs.sel(num=num).interp(r=rtidal)
        time.append(rprf.t.data[()])
        vr.append(-rprf.vel1_mw.data[()])
        mdot.append((-4*np.pi*rprf.r**2*rprf.rho*rprf.vel1_mw).data[()])
    if 'num' in rprofs.indexes:
        rprofs = rprofs.drop_indexes('num')
    rprofs['infall_speed'] = xr.DataArray(vr, coords=dict(t=time))
    rprofs['infall_rate'] = xr.DataArray(mdot, coords=dict(t=time))
    if 'num' not in rprofs.indexes:
        rprofs = rprofs.set_xindex('num')
    return rprofs


def calculate_accelerations(rprf):
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
        rprf = rprf.drop_indexes('num')
    pthm = rprf.rho
    ptrb = rprf.rho*rprf.dvel1_sq_mw
    acc = dict(adv=rprf.vel1_mw*rprf.vel1_mw.differentiate('r'),
               thm=-pthm.differentiate('r') / rprf.rho,
               trb=-ptrb.differentiate('r') / rprf.rho,
               cen=(rprf.vel2_mw**2 + rprf.vel3_mw**2) / rprf.r,
               grv=rprf.gacc1_mw,
               ani=((rprf.dvel2_sq_mw + rprf.dvel3_sq_mw - 2*rprf.dvel1_sq_mw)
                    / rprf.r))
    acc = xr.Dataset(acc)
    acc['dvdt_lagrange'] = acc.thm + acc.trb + acc.grv + acc.cen + acc.ani
    acc['dvdt_euler'] = acc.dvdt_lagrange - acc.adv
    return acc


def critical_time(s, pid):
    cores = s.cores[pid]
    rprofs = s.rprofs[pid]
    menc = []
    for num, core in cores.iterrows():
        # TODO(SMOON) critical radius become nan when pindex < 0
        if np.isnan(core.critical_radius):
            menc.append(np.nan)
        else:
            menc.append(rprofs.sel(num=num).menc.interp(r=core.critical_radius).data[()])
    cond = ((cores.tidal_radius > cores.critical_radius)
            & (menc > cores.critical_mass))
    cores_past_critical = cores.index[cond]
    if len(cores_past_critical) == 0:
        return np.nan
    else:
        return cores_past_critical[0]


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
    R_LP_BE = lpradius(M_BE, s.cs, s.gconst)
    R_LP_TES = lpradius(M_TES, s.cs, s.gconst)
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


def test_resolved_core(s, cores, ncells_min, f):
    """Test if the given core is sufficiently resolved.

    Need to be resolved at the time of collapse and at the time of instability,
    where former and latter are estimated to be t_coll and t_coll - f*tff.
    The additional requirement that the core must be resolved also at t_coll
    is to filter out such cores that form from unresolved fragmentation.

    Parameters
    ----------
    s : LoadSimCoreFormation
        Object containing simulation metadata
    cores : pandas.DataFrame
        Object containing core informations.
    ncells_min : int
        Minimum grid distance between a core and a particle.
    f : float
        Fuzzy factor to estimate critical time: tcoll - f*tff

    Returns
    -------
    bool
        True if a core is resolved, false otherwise.
    """
    cores = cores.sort_index()
    num = cores.tnorm.sub(-f).abs().astype('float64').idxmin()
    ncells = cores.loc[num].tidal_radius / s.dx
    ncells = min(ncells, cores.iloc[-1].tidal_radius / s.dx)
    if ncells >= ncells_min:
        return True
    else:
        return False


def test_isolated_core(s, cores):
    """Test if the given core is isolated.

    Criterion for an isolated core is that the core must not contain
    any particle at the time of collapse.

    Parameters
    ----------
    s : LoadSimCoreFormation
        Object containing simulation metadata.
    cores : pandas.DataFrame
        Object containing core informations.

    Returns
    -------
    bool
        True if a core is isolated, false otherwise.
    """
    num_tcoll = cores.index[-1]
    pds = s.load_partab(num_tcoll)
    gd = s.load_dendro(num_tcoll)

    nd = cores.loc[num_tcoll].envelop_id

    # Get all cells in this node.
    cells = set(gd.get_all_descendant_cells(nd))

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
    tff = tfreefall(cores.mean_density.loc[num_tcoll], s.gconst)
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


def lpdensity(r, cs, gconst):
    """Larson-Penston density profile

    Parameter
    ---------
    r : float
        Radius.
    cs : float
        Isothermal sound speed.
    gconst : float
        Gravitational constant.

    Returns
    -------
    float
        Asymptotic Larson-Penston density
    """

    return 8.86*cs**2/(4*np.pi*gconst*r**2)


def lpradius(m, cs, gconst):
    """Equivalent Larson-Penston radius containing mass m

    Parameter
    ---------
    m : float
        Mass.
    cs : float
        Isothermal sound speed.
    gconst : float
        Gravitational constant.

    Returns
    -------
    float
        Equivalent radius
    """
    return gconst*m/8.86/cs**2


def tfreefall(dens, gconst):
    """Free fall time at a given density.

    Parameter
    ---------
    dens : float
        Density.
    gconst : float
        Gravitational constant.

    Returns
    -------
    float
        Gravitational free-fall time
    """
    return np.sqrt(3*np.pi/(32*gconst*dens))


def reff_sph(vol):
    """Effective radius of a volume

    Reff = (3*vol/(4 pi))**(1/3)

    Parameter
    ---------
    vol : float
        Volume

    Returns
    -------
    float
        Effective spherical radius
    """
    fac = 0.6203504908994000865973817
    return fac*vol**(1/3)
