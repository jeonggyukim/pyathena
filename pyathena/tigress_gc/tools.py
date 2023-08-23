from pathlib import Path
from pygc.pot import MHubble, Plummer
import numpy as np
import xarray as xr
from scipy import optimize
from pyathena.tigress_gc import config
from pyathena.util import transform
from pyathena.util import units
from pyathena.classic import cooling

u = units.Units(muH=config.muH)


def calculate_ring_averages(s, num, Rmax, mf_crit=0.9, warmcold=False):
    """Calculate ring masked averages

    Parameters
    ----------
    s : pa.LoadSim object
    num : integer index of simulation output
    warmcold : if True, calculate quantities only for T < 2e4 gas
    """
    # Step 1. Load data
    ds = s.load_vtk(num, id0=False)
    field_list = [f for f in ds.field_list if 'scalar' not in f]
    dat = ds.get_field(field_list)
    dat = add_derived_fields(dat, ['R', 'gz_sg'])
    dat = dat.drop_vars(['velocity1', 'velocity2', 'gravitational_potential'])
    dx, dy, dz = s.domain['dx']
    if warmcold:
        dat = add_derived_fields(dat, 'T')
        # Set switch for warm-cold medium (Theta = 1 for T<2e4; 0 for T>2e4)
        is_wc = xr.where(dat.T.sel(z=0, method='nearest') < config.Twarm, 1, 0)
        # Apply switch
        dat = dat.where(dat.T < config.Twarm, other=0)
        # Save switch
        dat['is_wc'] = is_wc
        dat = dat.drop_vars('T')

    # midplane thermal pressure
    dat['pressure'] = dat.pressure.sel(z=0, method='nearest')

    # midplane turbulent pressure
    rho = dat.density.sel(z=0, method='nearest')
    vz = dat.velocity3.sel(z=0, method='nearest')
    dat['turbulent_pressure'] = rho*vz**2
    dat = dat.drop_vars('velocity3')

    # midplane magnetic stress
    if 'cell_centered_B1' in dat:
        bx = dat.cell_centered_B1.sel(z=0, method='nearest')
        by = dat.cell_centered_B2.sel(z=0, method='nearest')
        bz = dat.cell_centered_B3.sel(z=0, method='nearest')
        dat['magnetic_stress'] = 0.5*(bx**2 + by**2 - bz**2)
        dat = dat.drop_vars(['cell_centered_B1', 'cell_centered_B2', 'cell_centered_B3'])

    # weights
    bul = MHubble(rb=s.par['problem']['R_b'], rhob=s.par['problem']['rho_b'])
    bh = Plummer(Mc=s.par['problem']['M_c'], Rc=s.par['problem']['R_c'])
    dat['gz_ext'] = bul.gz(dat.x, dat.y, dat.z) + bh.gz(dat.x, dat.y, dat.z)
    dat['gz_ext'] = dat.gz_ext.transpose('z','y','x')
    dat['Wself'] = -(dat.density*dat.gz_sg*dz).sel(z=slice(0, s.domain['re'][2])).sum(dim='z')
    dat['Wext'] = -(dat.density*dat.gz_ext*dz).sel(z=slice(0, s.domain['re'][2])).sum(dim='z')
    dat = dat.drop_vars(['density', 'gz_sg', 'gz_ext'])


    fname = Path(s.basedir, "time_averages", "prims.nc")
    dat_tavg = xr.open_dataset(fname)
    dat_tavg = add_derived_fields(dat_tavg, 'surf')
    surf_th, mask = mask_ring_by_mass(s, dat_tavg, Rmax, mf_crit)
    dat = dat.where(mask).mean()
    for f in ['pressure', 'turbulent_pressure']:
        dat[f] /= dat.is_wc
    if 'magnetic_stress' in dat:
        dat['magnetic_stress'] /= dat.is_wc

    area = mask.sum().data[()]*dx*dy
    area_warmcold = area*dat.is_wc.data[()]
    dat = dat.drop_vars('is_wc')
    dat = dat.assign_attrs(dict(time=ds.domain['time'], area=area, area_warmcold=area_warmcold))

    return dat


def calculate_azimuthal_averages(s, num, warmcold=False):
    """Calculate azimuthal averages and write to file

    Parameters
    ----------
    s : pa.LoadSim object
    num : integer index of simulation output
    warmcold : if True, calculate quantities only for T < 2e4 gas

    Description
    -----------
    Calculates following Fields (function of (z,R))
    density            : <rho>
    pressure           : <P>
    turbulent_pressure : <rho v_z^2>
    velocity1          : <v_R>
    velocity2          : <v_phi>
    velocity3          : <v_z>
    helicity           : <v*w>
    mass_flux1         : <rho v_R>
    mass_flux2         : <rho v_phi>
    mass_flux3         : <rho v_z>
    B1                 : <B_R>
    B2                 : <B_phi>
    B3                 : <B_z>
    B_squared1         : <B_R^2>
    B_squared2         : <B_phi^2>
    B_squared3         : <B_z^2>
    Reynolds           : <rho v_R (v_phi - v_circ)>
    Maxwell            : <B_R B_phi>
    is_wc              : <Theta>
    """


    # Step 1. Load data
    ds = s.load_vtk(num, id0=False)
    field_list = [f for f in ds.field_list if f not in
                  ['heat_rate', 'cool_rate', 'specific_scalar[0]',
                   'specific_scalar[1]', 'specific_scalar[2]']]
    dat = ds.get_field(field_list)
    dat = add_derived_fields(dat, ['R', 'gz_sg'])
    dx, dy, dz = s.domain['dx']
    if warmcold:
        dat = add_derived_fields(dat, 'T')
        # Set switch for warm-cold medium (Theta = 1 for T<2e4; 0 for T>2e4)
        is_wc = xr.where(dat.T < config.Twarm, 1, 0)
        # Apply switch
        dat = dat.where(dat.T < config.Twarm, other=0)
        # Save switch
        dat['is_wc'] = is_wc
        dat = dat.drop('T')

    # helicity
    vx, vy, vz = dat.velocity1, dat.velocity2, dat.velocity3
    wx = vz.differentiate('y') - vy.differentiate('z')
    wy = vx.differentiate('z') - vz.differentiate('x')
    wz = vy.differentiate('x') - vx.differentiate('y')
    dat['helicity'] = vx*wx + vy*wy + vz*wz

    # midplane thermal pressure
    dat['pressure'] = dat.pressure.sel(z=0, method='nearest')

    # midplane turbulent pressure
    rho = dat.density.sel(z=0, method='nearest')
    vz = dat.velocity3.sel(z=0, method='nearest')
    dat['turbulent_pressure'] = rho*vz**2

    # midplane magnetic stress
    if 'cell_centered_B1' in dat:
        bx = dat.cell_centered_B1.sel(z=0, method='nearest')
        by = dat.cell_centered_B2.sel(z=0, method='nearest')
        bz = dat.cell_centered_B3.sel(z=0, method='nearest')
        dat['magnetic_stress'] = 0.5*(bx**2 + by**2 - bz**2)

    # transform vector quantities to cylindrical coordinates
    vec = (dat['velocity1'], dat['velocity2'], dat['velocity3'])
    R, vec_cyl = transform.to_cylindrical(vec, (0, 0, 0))
    for i, axis in enumerate([1,2,3]):
        dat[f'velocity{axis}'] = vec_cyl[i]

    if 'cell_centered_B1' in dat:
        vec = (dat['cell_centered_B1'], dat['cell_centered_B2'], dat['cell_centered_B3'])
        R, vec_cyl = transform.to_cylindrical(vec, (0, 0, 0))
        for i, axis in enumerate([1,2,3]):
            dat[f'cell_centered_B{axis}'] = vec_cyl[i]
        dat = dat.rename({'cell_centered_B1':'B1',
                          'cell_centered_B2':'B2',
                          'cell_centered_B3':'B3'})

    # calculate circular velocity
    vcirc = get_circular_velocity(s, dat.x, dat.y)

    # Calculate derived quantities
    dat['Reynolds'] = dat.density*dat.velocity1*(dat.velocity2 - vcirc)
    dat['mass_flux1'] = dat.density*dat.velocity1
    dat['mass_flux2'] = dat.density*dat.velocity2
    dat['mass_flux3'] = dat.density*dat.velocity3
    if 'B1' in dat:
        dat['Maxwell'] = -(dat.B1*dat.B2)
        dat['B_squared1'] = dat.B1**2
        dat['B_squared2'] = dat.B2**2
        dat['B_squared3'] = dat.B3**2

    # weights
    bul = MHubble(rb=s.par['problem']['R_b'], rhob=s.par['problem']['rho_b'])
    bh = Plummer(Mc=s.par['problem']['M_c'], Rc=s.par['problem']['R_c'])
    dat['gz_ext'] = bul.gz(dat.x, dat.y, dat.z) + bh.gz(dat.x, dat.y, dat.z)
    dat['gz_ext'] = dat.gz_ext.transpose('z','y','x')
    dat['Wself'] = -(dat.density*dat.gz_sg*dz).sel(z=slice(0, s.domain['re'][2])).sum(dim='z')
    dat['Wext'] = -(dat.density*dat.gz_ext*dz).sel(z=slice(0, s.domain['re'][2])).sum(dim='z')

    # Radial binning
    nbin = 64
    edges = np.linspace(0, 1000, nbin+1)
    dat = dat.groupby_bins('R', edges).mean()
    dat.attrs = {'time':ds.domain['time']}
    dat = dat.rename({'R_bins':'R'})
    # Calculate midpoint of each bin and use it as coordinate
    dat.update({'R':list(map(lambda x: x.mid, dat.R.data))})
    dat = dat.transpose('z','R')

    if warmcold:
        fmid = {'pressure', 'turbulent_pressure', 'magnetic_stress'}
        excl = {'is_wc', 'Wself', 'Wext'}
        for f in set(dat.data_vars) - fmid - excl:
            dat[f] /= dat.is_wc
        for f in fmid:
            dat[f] /= dat.is_wc.sel(z=0, method='nearest')
        area = (is_wc.sel(z=0, method='nearest')*dx*dy).groupby_bins('R', edges).sum()
        area = area.rename({'R_bins':'R'})
        area.coords['R'] = list(map(lambda x: x.mid, area.R.data))
        dat['area_warmcold'] = area

    return dat


def get_circular_velocity(s, x, y=0, z=0):
    bul = MHubble(rb=s.par['problem']['R_b'], rhob=s.par['problem']['rho_b'])
    bh = Plummer(Mc=s.par['problem']['M_c'], Rc=s.par['problem']['R_c'])
    if 'Omega_p' in s.par['problem']:
        Omega_p = s.par['problem']['Omega_p']
    elif 'Omega_0' in s.par['problem']:
        Omega_p = s.par['problem']['Omega_0']
    else:
        raise ValueError("No pattern speed information in the input file")
    vbul = bul.vcirc(x, y, z)
    vbh = bh.vcirc(x, y, z)
    R = np.sqrt(x**2 + y**2)
    vcirc = np.sqrt(vbul**2 + vbh**2) - R*Omega_p
    return vcirc


def mask_ring_by_mass(s, dat, Rmax, mf_crit=0.9):
    """Create ring mask by applying density threshold and radius cut

    Parameters
    ----------
    s : pa.LoadSim object
    dat : xarray dataset
    Rmax : maximum radius to exclude dust lanes
    mf_crit : mass fraction threshold

    Description
    -----------
    This function generates ring mask by selecting cells whose surface density
    is larger than critical surface density. The critical surface density is determined
    such that the total mass in the ring mask is a certain fraction of the total mass
    inside Rmax. This is intended to be applied to the time-averaged snapshot.
    """

    def _Mabove(surf_th):
        """Return total gas mass above threshold density surf_th."""
        surf = dat.surf
        M = surf.where(surf>surf_th).sum()*s.domain['dx'][0]*s.domain['dx'][1]
        return M.values[()]

    if not 'R' in dat.coords:
        dat = add_derived_fields(dat, 'R')
    R_mask = dat.R < Rmax

    dat = dat.where(R_mask, other=0)
    Mtot = _Mabove(0)
    surf_th = optimize.bisect(lambda x: mf_crit*Mtot-_Mabove(x), 1e1, 1e4)
    mask = dat.surf > surf_th

    mask = mask & R_mask
    return surf_th, mask


def find_snapshot_number(s, t0):
    """Return snapshot number that is closest to t0

    Parameters
    ----------
    s : pa.LoadSim object
    t0 : desired time in Myr
    """
    nl, nu = _bracket_snapshot_number(s, t0)
    tl = s.load_vtk(nl).domain['time']*u.Myr
    tu = s.load_vtk(nu).domain['time']*u.Myr
    offl = abs(tl-t0)
    offu = abs(tu-t0)
    num = nl if offl < offu else nu
    if min(offl, offu) > 0.01:
        print("WARNING: time offset is greater than 0.01 Myr")
    return num


def _bracket_snapshot_number(s, t0):
    """Return snapshot numbers [ns, ns+1] such that t(ns) <= t0 < t(ns+1)

    Parameters
    ----------
    s : pa.LoadSim object
    t0 : desired time in Myr
    """
    a = s.nums[0]
    b = s.nums[-1]
    # initial check
    ta = s.load_vtk(a).domain['time']*u.Myr
    tb = s.load_vtk(b).domain['time']*u.Myr
    if ta==t0:
        return (0,1)
    if (ta-t0)*(tb-t0) > 0:
        raise ValueError("No snapshot with t={} Myr.".format(t0) +
                         "Time at first and last snapshot: {:.2f} Myr, {:.2f} Myr".format(ta,tb))
    # bisection
    while (b - a > 1):
        c = round((a+b)/2)
        ta = s.load_vtk(a).domain['time']*u.Myr
        tb = s.load_vtk(b).domain['time']*u.Myr
        tc = s.load_vtk(c).domain['time']*u.Myr
        if (ta-t0)*(tc-t0) < 0:
            b = c
        else:
            a = c
    return (a, b)


def add_derived_fields(s, dat, fields=[]):
    """Add derived fields in a Dataset

    Parameters
    ----------
    dat    : xarray Dataset of variables
    fields : list containing derived fields to be added.
               ex) ['H', 'surf', 'T']
    """

    def _gz_self():
        phir = dat.gravitational_potential.shift(z=-1, fill_value=np.nan)
        phil = dat.gravitational_potential.shift(z=1, fill_value=np.nan)
        phir.loc[{'z':phir.z[-1]}] = 3*phir.isel(z=-2) - 3*phir.isel(z=-3) + phir.isel(z=-4)
        phil.loc[{'z':phir.z[0]}] = 3*phil.isel(z=1) - 3*phil.isel(z=2) + phil.isel(z=3)
        gz = (phil - phir) / (2*s.dz)
        return gz

    def _gz_ext():
        bul = MHubble(rb=s.par['problem']['R_b'], rhob=s.par['problem']['rho_b'])
        bh = Plummer(Mc=s.par['problem']['M_c'], Rc=s.par['problem']['R_c'])
        gz = (bul.gz(dat.x, dat.y, dat.z)
              + bh.gz(dat.x, dat.y, dat.z)).transpose()
        return gz

    def _weight_self():
        if 'gz_self' not in dat:
            add_derived_fields(s, dat, 'gz_self')

        wself = -(dat.density*dat.gz_self*s.dz
                  ).sel(z=slice(0, s.domain['re'][2])).sum(dim='z')
        return wself

    def _weight_ext():
        if 'gz_ext' not in dat:
            add_derived_fields(s, dat, 'gz_ext')

        wext = -(dat.density*dat.gz_ext*s.dz
                 ).sel(z=slice(0, s.domain['re'][2])).sum(dim='z')
        return wext

    def _fwarm():
        if 'temperature' not in dat:
            add_derived_fields(s, dat, 'temperature')
        fwarm = xr.where(dat.temperature < config.Twarm, True, False)
        return fwarm

    def _turbulent_pressure():
        return dat.density*dat.velocity3**2

    def _temperature():
        cf = cooling.coolftn()
        pok = dat.pressure*u.pok
        t1 = pok/(dat.density*u.muH) # muH = Dcode/mH
        temp = xr.DataArray(cf.get_temp(t1.values), coords=t1.coords,
                            dims=t1.dims)
        return temp

    def _scale_height():
        h = np.sqrt((dat.density*dat.z**2).sum()/dat.density.sum())
        return h

    def _surface_density():
        surf = (dat.density*s.dz).sum(dim='z')
        return surf

    if isinstance(fields, str):
        fields = [fields]

    for f in fields:
        if f not in dat:
            dat[f] = locals()['_'+f]()

#     if 'R' in fields:
#         dat.coords['R'] = np.sqrt(dat.y**2 + dat.x**2)
#
#     if 'phi' in fields:
#         dat.coords['phi'] = np.arctan2(dat.y, dat.x)
