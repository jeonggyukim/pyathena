import numpy as np
import xarray as xr
import pandas as pd
from scipy.special import erfcinv
from scipy import odr
from scipy.optimize import brentq
from scipy.integrate import quad
from scipy.interpolate import interp1d
from pathlib import Path
from pyathena.util import transform
from pyathena.core_formation import load_sim_core_formation
from pyathena.core_formation.tes import TESc, get_critical_tes
from pyathena.core_formation import config
from turb_sphere import utils


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
        x = self.mu + np.sqrt(2)*self.sigma*erfcinv(2 - 2*frac)
        return np.exp(x)


def find_tcoll_core(s, pid, ncells_min=27):
    """Find the GRID-dendro ID of the t_coll core of particle pid

    Parameters
    ----------
    s : LoadSim
        LoadSimCoreFormation instance.
    pid : int
        Particle id.

    Returns
    -------
    lid : int or None
        ID of the leaf corresponding to t_coll core. If unresolved, return None.
    """
    # load dendrogram at t = t_coll
    num = s.tcoll_cores.loc[pid].num
    gd = s.load_dendro(num, pruned=False)

    # find closeast leaf node to this particle
    pos_particle = s.tcoll_cores.loc[pid][['x1', 'x2', 'x3']]
    pos_particle = pos_particle.to_numpy()
    dst = [get_periodic_distance(get_coords_node(s, lid), pos_particle, s.Lbox)
           for lid in gd.leaves]
    lid = gd.leaves[np.argmin(dst)]

    if gd.len(lid) < ncells_min:
        # t_coll core is unresolved
        return None
    else:
        # return the grid-dendro ID of the t_coll core
        return lid


# TODO Can we predict the new sink position using the mean velocity inside the core?
# But that would require loading the hdf5 snapshot, making the core tracking more expensive.
# TODO Stopping condition due to leaf distance is just arbitrary, because in principle if the
# leaf disappears by a merger, it would keep tracking. We need more physically motivated stopping
# condition
def track_cores(s, pid):
    """Perform reverse core tracking

    Parameters
    ----------
    s : LoadSimCoreFormation
    pid : int

    Returns
    -------
    cores : pandas.DataFrame

    See also
    --------
    track_protostellar_cores : Forward core tracking after t_coll into
                               the protostellar stage.
    """
    # start from t = t_coll and track backward
    nums = np.arange(s.tcoll_cores.loc[pid].num, config.GRID_NUM_START-1, -1)
    num = nums[0]
    msg = f'[track_cores] processing model {s.basename} pid {pid} num {num}'
    print(msg)

    lid = find_tcoll_core(s, pid)
    ds = s.load_hdf5(num, header_only=True)

    if lid is None:
        msg = (
            f'[track_cores] t_coll core for pid {pid} is unresolved. '
            ' do not perform core tracking for this core.'
        )
        print(msg)
        nums_track = [num,]
        time = [ds['Time'],]
        leaf_id = [np.nan,]
        leaf_radius = [np.nan,]
        tidal_radius = [np.nan,]
        tcoll_resolved = False
    else:
        tcoll_resolved = True
        gd = s.load_dendro(num)

        # Calculate effective radius of this leaf
        rleaf = reff_sph(gd.len(lid)*s.dV)

        # Do the tidal correction to neglect attached substructures.

        # Calculate tidal radius
        rtidal = calculate_tidal_radius(s, gd, lid, lid)

        nums_track = [num,]
        time = [ds['Time'],]
        leaf_id = [lid,]
        leaf_radius = [rleaf,]
        tidal_radius = [rtidal,]

        for num in nums[1:]:
            msg = '[track_cores] processing model {} pid {} num {}'
            print(msg.format(s.basename, pid, num))
            gd = s.load_dendro(num)
            ds = s.load_hdf5(num, header_only=True)
            pds = s.load_partab(num)

            # find closeast leaf to the previous preimage
            dst = [get_node_distance(s, leaf, leaf_id[-1]) for leaf in gd.leaves]
            lid = gd.leaves[np.argmin(dst)]
            rleaf = reff_sph(gd.len(lid)*s.dV)

            # If there is sink particle in the leaf, stop tracking.
            idx = np.floor((pds[['x1', 'x2', 'x3']] - s.domain['le']) / s.dx).astype('int')
            idx = idx[['x3', 'x2', 'x1']]
            idx = idx.values
            idx = idx[:, 0]*s.domain['Nx'][1]*s.domain['Nx'][0] + idx[:,1]*s.domain['Nx'][0] + idx[:, 2]
            flag = 0
            for idx_ in idx:
                if idx_ in gd.get_all_descendant_cells(lid):
                    flag += 1
            if flag > 0:
                break

            # Calculate tidal radius
            rtidal = calculate_tidal_radius(s, gd, lid, lid)

            # If the center moved more than the tidal radius, stop tracking.
            if get_node_distance(s, lid, leaf_id[-1]) > tidal_radius[-1]:
                break

            nums_track.append(num)
            time.append(ds['Time'])
            leaf_id.append(lid)
            leaf_radius.append(rleaf)
            tidal_radius.append(rtidal)

    # SMOON: Using dtype=object is to prevent automatic upcasting from int to float
    # when indexing a single row. Maybe there is a better approach.
    cores = pd.DataFrame(dict(time=time,
                              leaf_id=leaf_id,
                              leaf_radius=leaf_radius,
                              tidal_radius=tidal_radius),
                         index=nums_track, dtype=object).sort_index()

    # Set attributes
    cores.attrs['pid'] = pid
    cores.attrs['numcoll'] = cores.index[-1]
    cores.attrs['tcoll_resolved'] = tcoll_resolved

    return cores


def track_protostellar_cores(s, pid):
    """Perform forward core tracking

    Parameters
    ----------
    s : LoadSimCoreFormation
    pid : int

    Returns
    -------
    cores : pandas.DataFrame

    See also
    --------
    track_cores : Reverse core tracking from t_coll back into
                  the prestellar stage.
    """
    # Load prestellar core list
    # Do not load from self.cores, which might already contain the derived core properties.
    # We do not want to write derived properties into cores.par{}.p.
    fname = Path(s.savdir, 'cores', 'cores.par{}.p'.format(pid))
    cores = pd.read_pickle(fname).sort_index()
    ncoll = cores.attrs['numcoll']

    # Select prestellar part
    cores = cores.loc[:ncoll]

    # nums after t_coll
    nums = [num for num in s.nums if num > ncoll]

    nums_track = []
    time = []
    leaf_id = []
    leaf_radius = []
    tidal_radius  = []

    for num in nums:
        msg = '[track_protostellar_cores] processing model {} pid {} num {}'
        print(msg.format(s.basename, pid, num))
        gd = s.load_dendro(num)
        ds = s.load_hdf5(num, header_only=True)
        pds = s.load_partab(num)

        if pid not in pds.index:
            # This sink particle has merged to other sink. Stop tracking
            break

        # Find closet leaf to the sink particle
        sink_pos = pds.loc[pid][['x1', 'x2', 'x3']].to_numpy()
        dst = [get_periodic_distance(get_coords_node(s, lid), sink_pos, s.Lbox)
               for lid in gd.leaves]
        lid = gd.leaves[np.argmin(dst)]
        rleaf = reff_sph(gd.len(lid)*s.dV)

        # Calculate tidal radius
        rtidal = calculate_tidal_radius(s, gd, lid, lid)

        nums_track.append(num)
        time.append(ds['Time'])
        leaf_id.append(lid)
        leaf_radius.append(rleaf)
        tidal_radius.append(rtidal)

    tmp = pd.DataFrame(dict(time=time,
                            leaf_id=leaf_id,
                            leaf_radius=leaf_radius,
                            tidal_radius=tidal_radius),
                       index=nums_track, dtype=object).sort_index()
    tmp.attrs = cores.attrs

    cores = pd.concat([cores, tmp])

    return cores


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
    mtidal = (4*np.pi*rprf.r**2*rprf.rho).sel(r=slice(0, core.tidal_radius)
                                              ).integrate('r').data[()]
    mean_tidal_density = mtidal / (4*np.pi*core.tidal_radius**3/3)

    if len(r) < 1:
        # Sonic radius is zero. Cannot find critical tes.
        p = rs = dcrit = rcrit = mcrit = np.nan
    else:
        def f(B, x):
            return B[0]*x + B[1]
        beta0 = [0.5, 1]

        linear = odr.Model(f)
        mydata = odr.Data(np.log(r), np.log(vr/s.cs))
        myodr = odr.ODR(mydata, linear, beta0=beta0)
        myoutput = myodr.run()
        p, intercept = myoutput.beta

        if p <= 0:
            rs = dcrit = rcrit = mcrit = np.nan
        else:
            # sonic radius
            rs = np.exp(-intercept/(p))
    
            # Find critical TES at the central density
            xi_s = rs / LJ_c
            tsc = TESc(p=p, rs=xi_s)
            try:
                xi_crit = tsc.get_rcrit(mode=mode)
                u, du = tsc.solve(xi_crit)
                dcrit = np.exp(-u)
                rcrit = xi_crit*LJ_c
                mcrit = tsc.menc(xi_crit)*MJ_c
            except ValueError:
                dcrit = rcrit = mcrit = np.nan

    res = dict(tidal_mass=mtidal, center_density=rhoc,
               mean_tidal_density=mean_tidal_density, sonic_radius=rs, pindex=p,
               critical_contrast=dcrit, critical_radius=rcrit,
               critical_mass=mcrit)
    return res


def calculate_critical_radius(s, core, rprf):
    tsc = TESc(p=core.pindex)
    rs_floor = tsc.sonic_radius_floor()
    rs_ceil = tsc._rs_ceil

    rs = core.sonic_radius
    for rmax, menc in zip(rprf.r.data[1:], rprf.menc.data[1:]):
        # Find a TES that has the critical radius = rmax
        def fx(xi_s):
            tsc = TESc(p=core.pindex, rs=xi_s)
            return xi_s*rmax/tsc.get_rcrit() - rs
        # Sometime, when p and rs is extremely small, ValueError occurs.
        try:
            xi_s = brentq(fx, rs_floor, rs_ceil)
        except ValueError:
            return np.nan
        tsc = TESc(p=core.pindex, rs=xi_s)
        conv = rs/xi_s
        mcrit = conv*tsc.menc(tsc.get_rcrit())
        if menc > mcrit:
            return rmax
    return np.inf


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
    vel1, vel2, vel3 : Mass-weighted mean velocities (v_r, v_theta, v_phi).
    vel1_sq_mw, vel2_sq_mw, vel3_sq_mw : Mass-weighted variance of the velocities.
    gacc1_mw : Mass-weighted mean gravitational acceleration.
    phi_mw : Mass-weighted mean gravitational potential.
    """
    # Sometimes, tidal radius is so small that the angular momentum vector
    # Cannot be computed. In this case, fall back to default behavior.
    # (to_spherical will assume z axis as the polar axis).
    if lvec is not None and (np.array(lvec)**2).sum() == 0:
        lvec = None

    # Slice data
    nbin = int(np.ceil(rmax/s.dx))
    ledge = 0.5*s.dx
    redge = (nbin + 0.5)*s.dx
    ds = ds.sel(x=slice(origin[0] - redge, origin[0] + redge),
                y=slice(origin[1] - redge, origin[1] + redge),
                z=slice(origin[2] - redge, origin[2] + redge))

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

    # Perform radial binnings
    rprofs = {}

    # Volume-weighted averages
    for k in ['rho']:
        rprf_c = ds_sph[k].sel(x=origin[0], y=origin[1], z=origin[2]).drop(['x', 'y', 'z'])
        rprf = transform.fast_groupby_bins(ds_sph[k], 'r', ledge, redge, nbin)
        rprofs[k] = xr.concat([rprf_c, rprf], 'r')

    # Mass-weighted averages
    for k in ['gacc1', 'vel1', 'vel2', 'vel3', 'phi']:
        rprf_c = ds_sph[k].sel(x=origin[0], y=origin[1], z=origin[2]).drop(['x', 'y', 'z'])
        rprf = transform.fast_groupby_bins(ds_sph['rho']*ds_sph[k], 'r', ledge, redge, nbin) / rprofs['rho']
        rprofs[k+'_mw'] = xr.concat([rprf_c, rprf], 'r')

    # RMS averages
    for k in ['vel1', 'vel2', 'vel3']:
        rprf_c = ds_sph[k].sel(x=origin[0], y=origin[1], z=origin[2]).drop(['x', 'y', 'z'])**2
        rprf = transform.fast_groupby_bins(ds_sph['rho']*ds_sph[k]**2, 'r', ledge, redge, nbin) / rprofs['rho']
        rprofs[k+'_sq_mw'] = xr.concat([rprf_c, rprf], 'r')

    rprofs = xr.Dataset(rprofs)

    # Drop theta and phi coordinates
    for k in ['th', 'ph']:
        if k in rprofs:
            rprofs = rprofs.drop_vars(k)

    return rprofs


def calculate_prj_radial_profile(s, num, origin):
    """Calculate projected radial profile of column density and velocities

    Parameters
    ----------
    s : LoadSim
        LoadSimCoreFormation instance.
    core : pandas.Series
        Object containing core informations

    Returns
    -------
    rprf : dict
        Dictionary containing projected radial profiles along x,y,z directions
    """

    # Read the central position of the core and recenter the snapshot
    xc, yc, zc = origin

    prj = s.read_prj(num)

    xycoordnames = dict(z=['x', 'y'],
                        x=['y', 'z'],
                        y=['z', 'x'])

    xycenters = dict(z=[xc, yc],
                     x=[yc, zc],
                     y=[zc, xc])

    # Calculate surface density radial profiles
    rprofs = {}
    ledge = 0.5*s.dx
    nbin = s.domain['Nx'][0]//2 - 1
    redge = (nbin + 0.5)*s.dx
    for i, ax in enumerate(['x', 'y', 'z']):
        x1, x2 = xycoordnames[ax]
        x1c, x2c = xycenters[ax]

        # Volume-weighted averages
        for qty in prj[ax].keys():
            ds = prj[ax][qty].copy(deep=True)
            ds, new_center = recenter_dataset(ds, {x1:x1c, x2:x2c})
            ds.coords['R'] = np.sqrt((ds.coords[x1]- new_center[x1])**2
                                     + (ds.coords[x2] - new_center[x2])**2)
            rprf_c = xr.DataArray(ds.sel({x1:new_center[x1], x2:new_center[x2]}).data[()],
                                  dims='R', coords={'R':[0,]})
            rprf = transform.fast_groupby_bins(ds, 'R', ledge, redge, nbin)
            rprf = xr.concat([rprf_c, rprf], dim='R')
            rprofs[f'{ax}_{qty}'] = rprf

        # Mass-weighted averages
        for qty in prj[ax].keys():
            ds = prj[ax][qty].copy(deep=True)
            ds, new_center = recenter_dataset(ds, {x1:x1c, x2:x2c})
            ds.coords['R'] = np.sqrt((ds.coords[x1]- new_center[x1])**2
                                     + (ds.coords[x2] - new_center[x2])**2)
            w = prj[ax]['Sigma_gas'].copy(deep=True)
            w, _ = recenter_dataset(w, {x1:x1c, x2:x2c})
            w.coords['R'] = np.sqrt((w.coords[x1]- new_center[x1])**2
                                    + (w.coords[x2] - new_center[x2])**2)
            rprf_c = xr.DataArray(ds.sel({x1:new_center[x1], x2:new_center[x2]}).data[()],
                                  dims='R', coords={'R':[0,]})
            num = transform.fast_groupby_bins(w*ds, 'R', ledge, redge, nbin)
            den = transform.fast_groupby_bins(w, 'R', ledge, redge, nbin)
            rprf = num/den
            rprf = xr.concat([rprf_c, rprf], dim='R')
            rprofs[f'{ax}_{qty}_mw'] = rprf

    rprofs = xr.Dataset(rprofs)

    return rprofs


def calculate_lagrangian_props(s, cores, rprofs):
    # Slice cores that have corresponding radial profiles
    common_indices = sorted(set(cores.index) & set(rprofs.num.data))
    cores = cores.loc[common_indices]
    ncrit = cores.attrs['numcrit']
    ncoll = cores.attrs['numcoll']
    rcore = cores.attrs['rcore']
    mcore = cores.attrs['mcore']

    if np.isnan(ncrit):
        radius = menc = rhoe = rhoavg = np.nan
        vinfall = sigma_mw = np.nan
        Fthm = Ftrb = Fcen = Fani = Fgrv = np.nan
    else:
        radius, menc, rhoe, rhoavg = [], [], [], []
        vinfall, sigma_mw = [], []
        Fthm, Ftrb, Fcen, Fani, Fgrv = [], [], [], [], []
        for num, core in cores.iterrows():
            rprof = rprofs.sel(num=num)

            # Find radius which encloses mcore.
            if rprof.menc.isel(r=-1) < mcore:
                # In this case, no radius up to maximum tidal radius encloses
                # mcore. This means we are safe to set rcore = Rtidal.
                r_M = np.inf
            else:
                r_M = brentq(lambda x: rprof.menc.interp(r=x) - mcore,
                                           rprof.r.isel(r=0), rprof.r.isel(r=-1))
            radius.append(r_M)
            # enclosed mass within the critical radius
            if np.isnan(core.critical_radius):
                menc.append(np.nan)
            else:
                menc.append(rprof.menc.interp(r=core.critical_radius).data[()])


            # Mass-weighted infall speed
            rprf = rprof.sel(r=slice(0, r_M))
            vin = rprf.vel1_mw.weighted(rprf.r**2*rprf.rho).mean().data[()]
            vinfall.append(vin)

            # Mass-weighted velocity dispersion
            rprf = rprof.sel(r=slice(0, r_M))
            sigmw = np.sqrt(rprf.dvel1_sq_mw.weighted(rprf.r**2*rprf.rho).mean().data[()])
            sigma_mw.append(sigmw)

            # select r = r_M
            rprf = rprof.interp(r=r_M)
            rhoe.append(rprf.rho.data[()])
            rhoavg.append(mcore / (4*np.pi*r_M**3/3))
            Fthm.append(rprf.Fthm.data[()])
            Ftrb.append(rprf.Ftrb.data[()])
            Fcen.append(rprf.Fcen.data[()])
            Fani.append(rprf.Fani.data[()])
            Fgrv.append(rprf.Fgrv.data[()])
    lprops = pd.DataFrame(data = dict(radius=radius, menc=menc, edge_density=rhoe, mean_density=rhoavg,
                                      vinfall=vinfall, sigma_mw=sigma_mw,
                                      Fthm=Fthm, Ftrb=Ftrb, Fcen=Fcen, Fani=Fani, Fgrv=Fgrv),
                          index = cores.index)

    # Attach some attributes
    # Velocity dispersion at t_crit
    if np.isnan(ncrit):
        sigma_r = np.nan
    else:
        sigma_r = lprops.loc[ncrit].sigma_mw
    lprops.attrs['sigma_r'] = sigma_r

    # Free-fall time at t_coll
    lprops.attrs['tff_coll'] = tfreefall(lprops.loc[ncoll].mean_density, s.gconst)

    return lprops


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
               cen=((rprf.vel2_mw**2 + rprf.vel3_mw**2) / rprf.r).where(rprf.r > 0, other=0),
               grv=rprf.gacc1_mw,
               ani=((rprf.dvel2_sq_mw + rprf.dvel3_sq_mw - 2*rprf.dvel1_sq_mw)
                    / rprf.r).where(rprf.r > 0, other=0))
    acc = xr.Dataset(acc)
    acc['dvdt_lagrange'] = acc.thm + acc.trb + acc.grv + acc.cen + acc.ani
    acc['dvdt_euler'] = acc.dvdt_lagrange - acc.adv

    dm = 4*np.pi*rprf.r**2*rprf.rho
    acc['Fadv'] = (dm*acc.adv).cumulative_integrate('r')
    acc['Fthm'] = (dm*acc.thm).cumulative_integrate('r')
    acc['Ftrb'] = (dm*acc.trb).cumulative_integrate('r')
    acc['Fcen'] = (dm*acc.cen).cumulative_integrate('r')
    acc['Fgrv'] = (-dm*acc.grv).cumulative_integrate('r')
    acc['Fani'] = (dm*acc.ani).cumulative_integrate('r')

    return acc


def calculate_observables(s, core, rprf, rmax, method):
    """Calculate observable properties of a core"""

    match method:
        case 'three_d':
            # Method 1: calculate observable properties from radial profile, truncated at rmax

            # Create interpolation functions
            try:
                rho_itp = interp1d(rprf.r.data, rprf.rho.data)
                rfwhm = utils.fwhm(rho_itp, rmax)

                # Mean density inside FWHM radius
                mfwhm = utils.integrate_2d_projected(rho_itp, rfwhm, rmax)
                dfwhm = mfwhm / (4*np.pi*rfwhm**3/3)

                # Central surface density
                dcen = utils.integrate_los(rho_itp, 0, rmax)

                # Projected velocity dispersion within FWHM
                dvel_sq_itp = interp1d(rprf.r.data, rprf.dvel1_sq_mw)
                two_ke = utils.integrate_2d_projected(lambda x: rho_itp(x)*dvel_sq_itp(x), rfwhm, rmax)
                sigma = np.sqrt(two_ke/mfwhm)

#        # Critical mass of TES having rcrit = R_FWHM
#        ts = TESc(p=core.pindex, sigma=sigma)
#        xi_fwhm = utils.fwhm(ts.rho, ts.rmax)
#        m_fwhm = utils.integrate_2d_projected(ts.rho, xi_fwhm, ts.rmax)
#        m_over_xi = m_fwhm / xi_fwhm
#        mcrit = np.pi*s.cs**2/s.gconst*m_over_xi*rfwhm
            except ValueError:
                rfwhm = mfwhm = dfwhm = dcen = sigma = np.nan
            obsprops = dict(radius=rfwhm, mass=mfwhm, mean_density=dfwhm,
                            central_column_density=dcen, velocity_dispersion=sigma)

        case 'iterative':
            # Method 2: calculate observable properties from iterative background subtraction, using
            # the 3D radial profile.

            try:
                # Calculate background subtracted quantities (iterative subtraction)
                rho_itp = interp1d(rprf.r.data, rprf.rho.data)
                rfwhm = utils.fwhm_bgrsub(rho_itp, rmax, method='iterative')
                dcol_bgr = utils.integrate_los(rho_itp, rfwhm, rmax)
                m_bgr = dcol_bgr*np.pi*rfwhm**2
                mfwhm = utils.integrate_2d_projected(rho_itp, rfwhm, rmax) - m_bgr
                dfwhm = mfwhm / (4*np.pi*rfwhm**3/3)

                # Central surface density (background subtracted)
                dcen = utils.integrate_los(rho_itp, 0, rmax) - dcol_bgr

                # Projected velocity dispersion within FWHM
                dvel_sq_itp = interp1d(rprf.r.data, rprf.dvel1_sq_mw)
                two_ke = utils.integrate_2d_projected(lambda x: rho_itp(x)*dvel_sq_itp(x), rfwhm, rmax)
                sigma = np.sqrt(two_ke/mfwhm)
            except ValueError:
                # Because r_obs is found by multiplying two to the HMHW,
                # it can sometimes exceed the maximum radius of the radial profiles.
                # If that happens, we cannot solve for the properties within the
                # FWHM radius.
                rfwhm = mfwhm = dfwhm = dcen = sigma = np.nan
            obsprops = dict(radius=rfwhm, mass=mfwhm, mean_density=dfwhm,
                            central_column_density=dcen, velocity_dispersion=sigma)

        case 'two_d':
            obsprops = dict()

            # Prepare for projection map
            # See `calculate_prj_radial_profile`
            xc, yc, zc = get_coords_node(s, core.leaf_id)
            prj = s.read_prj(core.name)
            xycoordnames = dict(z=['x', 'y'],
                                x=['y', 'z'],
                                y=['z', 'x'])
            xycenters = dict(z=[xc, yc],
                             x=[yc, zc],
                             y=[zc, xc])


            # Simplest background subtraction -- average column in the whole box
            dcol_bgr = s.rho0*s.Lbox

            for i, ax in enumerate(['x', 'y', 'z']):
                dcol = rprf[f'{ax}_Sigma_gas']
                ncrit_list = [10, 20, 30, 50, 100]
                sigma = dict()
                try:
                    # Calculate FWHM quantities
                    rfwhm = utils.fwhm(interp1d(dcol.R.data[()], dcol.data-dcol_bgr),
                                       dcol.R.max()[()], which='column')
                    mfwhm = ((dcol-dcol_bgr)*2*np.pi*dcol.R).sel(R=slice(0, rfwhm)).integrate('R').data[()]
                    dcol_fwhm = dcol.interp(R=rfwhm)
                    mfwhm_bgrsub = ((dcol-dcol_fwhm)*2*np.pi*dcol.R).sel(R=slice(0, rfwhm)).integrate('R').data[()]
                    dfwhm = mfwhm / (4*np.pi*rfwhm**3/3)

                    # Calculate surface-density weighted average projected velocity
                    # dispersion, directly using the 2D map data
                    x1, x2 = xycoordnames[ax]
                    x1c, x2c = xycenters[ax]
                    for ncrit in ncrit_list:
                        w = prj[ax]['Sigma_gas']
                        ds = prj[ax][f'veldisp_nc{ncrit}'].copy(deep=True)
                        ds, new_center = recenter_dataset(ds, {x1:x1c, x2:x2c})
                        ds.coords['R'] = np.sqrt((ds.coords[x1]- new_center[x1])**2
                                                 + (ds.coords[x2] - new_center[x2])**2)
                        sigma[ncrit] = ds.where(ds.R < rfwhm).weighted(w).mean().data[()]

                except ValueError:
                    rfwhm = mfwhm = mfwhm_bgrsub = dfwhm = np.nan
                    for ncrit in ncrit_list:
                        sigma[ncrit] = np.nan

                # Central column density
                dcen = dcol.sel(R=0).data[()] - dcol_bgr

                obsprops[f'{ax}_radius'] = rfwhm
                obsprops[f'{ax}_mass'] = mfwhm
                obsprops[f'{ax}_mass_bgrsub'] = mfwhm_bgrsub
                obsprops[f'{ax}_mean_density'] = dfwhm
                obsprops[f'{ax}_center_column_density'] = dcen
                for ncrit in ncrit_list:
                    obsprops[f'{ax}_velocity_dispersion_nc{ncrit}'] = sigma[ncrit]

    obsprops['num'] = core.name

    return obsprops


def column_density(rcyl, frho, rmax):
    """Calculate column density

    Parameters
    ----------
    rcyl : float
        Cylindrical radius at which the column density is computed
    frho : function
        The function rho(r) that returns the volume density at a given
        spherical radius.
    rmax : float
        The maximum radius to integrate out.

    Returns
    -------
    dcol : float
        Column density.
    """
    def func(z, rcyl):
        r = np.sqrt(rcyl**2 + z**2)
        return frho(r)
    if isinstance(rcyl, np.ndarray):
        dcol = []
        for R in rcyl:
            zmax = np.sqrt(rmax**2 - R**2)
            res, _ = quad(func, 0, zmax, args=(R,), epsrel=1e-2, limit=200)
            dcol.append(2*res)
        dcol = np.array(dcol)
    else:
        zmax = np.sqrt(rmax**2 - rcyl**2)
        res, _ = quad(func, 0, zmax, args=(rcyl,), epsrel=1e-2, limit=200)
        dcol = 2*res
    return dcol


def critical_time(s, pid, method=1):
    cores = s.cores[pid].copy()
    if len(cores) == 0:
        return np.nan
    cores = cores.loc[:cores.attrs['numcoll']]
    rprofs = s.rprofs[pid]

    ncrit = None

    if method in {1, 2}:
        # For these method, check conditions from t_coll, marching backward in time.
        for num, core in cores.sort_index(ascending=False).iterrows():
            if num in cores.index[-2:]:
                if np.isnan(core.critical_radius):
                    msg = f"Critical radius at t_coll - {cores.attrs['numcoll']-num}"
                    msg += f" is NaN for par {pid}, method {method}."
                    msg += f" This may have been caused by negative pindex. Continuing"
                    s.logger.warning(msg)
                    continue
            rprf = rprofs.sel(num=num)

            if method == 1:
                # Critical conditions are satisfied after critical time,
                # throughout the collapse.
                if np.isfinite(core.critical_radius):
                    menc = rprf.menc.interp(r=core.critical_radius).data[()]
                else:
                    menc = np.nan
                cond1 = core.tidal_radius >= core.critical_radius
                cond2 = menc >= core.critical_mass
                cond = cond1 and cond2
                if not cond:
                    ncrit = num + 1
                    break
            elif method == 2:
                # Net force at the critical radius is negative after the
                # critical time, throughout the collapse.
                if np.isfinite(core.critical_radius):
                    fnet = (rprf.Fthm + rprf.Ftrb + rprf.Fcen + rprf.Fani
                            - rprf.Fgrv)
                    fnet = fnet.interp(r=core.critical_radius).data[()]
                else:
                    fnet = np.nan
                if not fnet < 0:
                    ncrit = num + 1
                    break
    elif method in {4, 5, 6, 7}:
        for num, core in cores.sort_index(ascending=True).iterrows():
            if method == 4:
                # Predicted critical time using R_tidal_max
                cond = core.tidal_radius >= core.critical_radius
                if cond:
                    ncrit = num
                    break
            elif method == 5:
                # Predicted critical time using R_tidal_min
                cond = core.leaf_radius >= core.critical_radius
                if cond:
                    ncrit = num
                    break
            elif method == 6:
                # Predicted critical time using R_tidal_avg
                rtidal_avg = 0.5*(core.leaf_radius + core.tidal_radius)
                cond = rtidal_avg >= core.critical_radius
                if cond:
                    ncrit = num
                    break
            elif method == 7:
                rprf = rprofs.sel(num=num)
                # Predicted critical time using R_tidal_avg and Menc
                if np.isfinite(core.critical_radius):
                    menc = rprf.menc.interp(r=core.critical_radius).data[()]
                else:
                    menc = np.nan
                rtidal_avg = 0.5*(core.leaf_radius + core.tidal_radius)
                cond1 = rtidal_avg >= core.critical_radius
                cond2 = menc >= core.critical_mass
                if cond1 and cond2:
                    ncrit = num
                    break

    if ncrit is None or ncrit == cores.index[-1] + 1:
        # If the critical condition is satisfied for all time, or is not
        # satisfied at t_coll, set ncrit to NaN.
        # TODO: we may not want to discard those that the critical condition
        # is satisfied for all times.
        ncrit = np.nan
    return ncrit


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


def get_periodic_distance(pos1, pos2, Lbox, return_axis_distance=False):
    hLbox = 0.5*Lbox
    axis_distance = []
    for x1, x2 in zip(pos1, pos2):
        dst = np.abs(x1-x2)
        dst = Lbox - dst if dst > hLbox else dst
        axis_distance.append(dst)
    axis_distance = np.array(axis_distance)
    dst = np.sqrt((axis_distance**2).sum())
    if return_axis_distance:
        return axis_distance
    else:
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
    rsonic = get_sonic(Mach, Lbox)
    if rho_amb is None:
        rho_amb = s.get_contrast(mfrac)
    rhoc_BE, R_BE, M_BE = get_critical_tes(rhoe=rho_amb, rsonic=np.inf,
                                           pindex=0.5)
    rhoc_TES, R_TES, M_TES = get_critical_tes(rhoe=rho_amb, rsonic=rsonic,
                                              pindex=0.5)
    R_LP_BE = lpradius(M_BE, s.cs, s.gconst)
    R_LP_TES = lpradius(M_TES, s.cs, s.gconst)
    dx_req_LP = R_LP_BE/ncells_min
    dx_req_BE = R_BE/ncells_min
    ncells_req_LP = np.ceil(Lbox/dx_req_LP).astype(int)
    ncells_req_BE = np.ceil(Lbox/dx_req_BE).astype(int)

    print(f"Mach number = {Mach}")
    print("sonic length = {}".format(rsonic))
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
    center : dict
        {x:xc, y:yc} or {x:xc, y:yc, z:zc}, etc.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Recentered dataset.
    tuple
        Position of the new center. This must be the grid coordinates
        closest, but not exactly the same, to (0, 0, 0).
    """
    shift, new_center = {}, {}
    for dim, pos in center.items():
        hNx = ds.sizes[dim] // 2
        coords = ds.coords[dim].data
        dx = coords[1] - coords[0]
        shift[dim] = hNx - np.where(np.isclose(coords, pos, atol=0.1*dx))[0][0]
        new_center[dim] = ds.coords[dim].isel({dim: hNx}).data[()]

    return ds.roll(shift), new_center


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


def test_resolved_core(s, cores, ncells_min):
    """Test if the given core is sufficiently resolved.

    Returns True if the critical radius at t_crit is greater than
    ncells_min*dx.

    Parameters
    ----------
    s : LoadSimCoreFormation
        Object containing simulation metadata
    pid : int
        Particle ID.
    ncells_min : int
        Minimum number of cells to be considered resolved.

    Returns
    -------
    bool
        True if a core is resolved, false otherwise.
    """
    ncrit = cores.attrs['numcrit']
    if np.isnan(ncrit):
        return False
    ncells = cores.loc[ncrit].critical_radius / s.dx
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
    pid : int
        Particle ID.

    Returns
    -------
    bool
        True if a core is isolated, false otherwise.
    """
    ncrit = cores.attrs['numcrit']
    if np.isnan(ncrit):
        return False
    pds = s.load_partab(ncrit)
    pstar = pds[['x1', 'x2', 'x3']]
    core = cores.loc[ncrit]

    nd = core.leaf_id
    pcore = get_coords_node(s, nd)

    dst = np.sqrt(((pstar - pcore)**2).sum(axis=1))

    return (dst > core.tidal_radius).all()


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


def get_evol_norm(vmin=-3, vmid=0, vmax=1):
    """Get a normalization for color coding evolutionary time

    Blue (vmin)  ->  white (vmid)  ->  red (vmax).
    """
    # Color scale
    from matplotlib import colors
    alpha = np.log(0.5) / np.log((vmid - vmin)/(vmax - vmin))

    def _forward(x):
        t = (x - vmin) / (vmax - vmin)
        return t**alpha

    def _inverse(x):
        return vmin + (vmax - vmin)*x**(1/alpha)
    norm = colors.FuncNorm((_forward, _inverse), vmin=vmin, vmax=vmax)
    return norm


def get_evol_cbar(mappable, ax=None, cax=None, ticks=[-3, -1, 0, 0.5, 1]):
    """Get an appropriate color bar for get_evol_norm"""
    import matplotlib.pyplot as plt
    lbl = r'$\dfrac{t - t_\mathrm{crit}}{\Delta t_\mathrm{coll}}$'
    cbar = plt.colorbar(mappable, ax=ax, cax=cax, label=lbl)
    cbar.solids.set(alpha=1)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(ticks)
    cbar.ax.minorticks_off()
    return cbar


def sawtooth(x, xmin, xmax, ymin, ymax):
    """Sawtooth curve

    Linear from [xmin, ymin] to [xmax, ymax] and then periodic elsewhere
    """
    t = ((x - xmax) + (x - xmin)) / (xmax - xmin)
    p = 2
    u = 2*(t / p - np.floor(0.5 + t / p))
    y = 0.5*(ymax - ymin)*u + 0.5*(ymax + ymin)
    return y
